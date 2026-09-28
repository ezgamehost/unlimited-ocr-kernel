//! Read-only safetensors storage. Files must not be modified while mapped.
use crate::error::{OcrError, Result};
use memmap2::{Mmap, MmapOptions};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    sync::Mutex,
};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub enum DType {
    BF16,
    F16,
    F32,
    I64,
    I8,
}

impl DType {
    fn bytes(self) -> usize {
        match self {
            Self::BF16 | Self::F16 => 2,
            Self::F32 => 4,
            Self::I64 => 8,
            Self::I8 => 1,
        }
    }
}

#[derive(Debug, Deserialize)]
struct Descriptor {
    dtype: DType,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub struct Checkpoint {
    mapping: Mmap,
    file: Mutex<File>,
    data_start: usize,
    tensors: BTreeMap<String, Descriptor>,
    metadata: BTreeMap<String, String>,
}

#[derive(Clone, Copy)]
pub struct WeightView<'a> {
    pub dtype: DType,
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
}

impl WeightView<'_> {
    pub fn validate(&self) -> Result<()> {
        let size = self
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .and_then(|n| n.checked_mul(self.dtype.bytes()));
        if size != Some(self.bytes.len()) {
            return Err(OcrError::Shape("weight view byte length".into()));
        }
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.bytes.len() / self.dtype.bytes()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn to_f32(&self) -> Vec<f32> {
        match self.dtype {
            DType::I8 => self.bytes.iter().map(|&v| (v as i8) as f32).collect(),
            DType::BF16 => self
                .bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                .collect(),
            DType::F16 => self
                .bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect(),
            DType::F32 => self
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            DType::I64 => self
                .bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b) as f32)
                .collect(),
        }
    }
}

impl Checkpoint {
    /// The caller must ensure no process truncates or changes `path` during use.
    /// Artifacts should be downloaded atomically to a versioned immutable path.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        if file.metadata()?.len() < 8 {
            return Err(OcrError::Format("missing header length".into()));
        }
        // SAFETY: read-only mapping retained for the lifetime of borrowed views.
        // No typed references are formed from possibly unaligned payload bytes.
        // External file mutation is forbidden by the documented file contract.
        let mapping = unsafe { MmapOptions::new().map(&file)? };
        let header_len = u64::from_le_bytes(mapping[..8].try_into().unwrap());
        let header_len = usize::try_from(header_len)
            .map_err(|_| OcrError::Format("header length overflow".into()))?;
        if header_len > 100_000_000 {
            return Err(OcrError::Format("header exceeds 100 MB".into()));
        }
        let data_start = header_len
            .checked_add(8)
            .filter(|&n| n <= mapping.len())
            .ok_or_else(|| OcrError::Format("truncated header".into()))?;
        let mut header: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&mapping[8..data_start])?;
        let metadata = header
            .remove("__metadata__")
            .map(serde_json::from_value)
            .transpose()?
            .unwrap_or_default();
        let mut tensors = BTreeMap::new();
        for (name, value) in header {
            let descriptor: Descriptor = serde_json::from_value(value)?;
            let count = descriptor
                .shape
                .iter()
                .try_fold(1usize, |a, &b| a.checked_mul(b))
                .and_then(|n| n.checked_mul(descriptor.dtype.bytes()))
                .ok_or_else(|| OcrError::Format(format!("tensor size overflow: {name}")))?;
            let [start, end] = descriptor.data_offsets;
            if end < start || end - start != count || end > mapping.len() - data_start {
                return Err(OcrError::Format(format!("invalid tensor offsets: {name}")));
            }
            tensors.insert(name, descriptor);
        }
        let mut spans: Vec<_> = tensors.values().map(|d| d.data_offsets).collect();
        spans.sort_unstable();
        let mut cursor = 0;
        for [start, end] in spans {
            if start != cursor {
                return Err(OcrError::Format(
                    "payload gaps or overlapping tensors".into(),
                ));
            }
            cursor = end;
        }
        if cursor != mapping.len() - data_start {
            return Err(OcrError::Format("unclaimed trailing payload".into()));
        }
        Ok(Self {
            mapping,
            file: Mutex::new(file),
            data_start,
            tensors,
            metadata,
        })
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }
    pub fn sha256(&self) -> Result<String> {
        // Read the same inode used by the mapping, without faulting every tensor
        // page into the process's mapped working set. Serialize file-offset use
        // so repeated/concurrent checks start at zero and hash the entire file.
        let mut file = self
            .file
            .lock()
            .map_err(|_| OcrError::Format("checkpoint file lock poisoned".into()))?;
        file.seek(SeekFrom::Start(0))?;
        let mut buffer = vec![0u8; 1 << 20];
        let mut remaining = self.mapping.len();
        let mut hash = Sha256::new();
        while remaining > 0 {
            let count = remaining.min(buffer.len());
            file.read_exact(&mut buffer[..count])?;
            hash.update(&buffer[..count]);
            remaining -= count;
        }
        if file.read(&mut buffer[..1])? != 0 {
            return Err(OcrError::Format("checkpoint file size changed".into()));
        }
        Ok(format!("{:x}", hash.finalize()))
    }
    pub fn file_bytes(&self) -> usize {
        self.mapping.len()
    }

    pub fn tensor(&self, name: &str) -> Result<WeightView<'_>> {
        let d = self
            .tensors
            .get(name)
            .ok_or_else(|| OcrError::Format(format!("missing tensor: {name}")))?;
        Ok(WeightView {
            dtype: d.dtype,
            shape: &d.shape,
            bytes: &self.mapping
                [self.data_start + d.data_offsets[0]..self.data_start + d.data_offsets[1]],
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
}
