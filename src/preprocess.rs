use crate::{
    error::{OcrError, Result},
    tensor::Tensor,
};
use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage};
use std::path::Path;
mod crop_ratios;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageMode {
    Base,
    Gundam,
}
pub struct PreparedImage {
    pub global: Tensor,
    pub crops: Vec<Tensor>,
    pub grid: [usize; 2],
}

impl PreparedImage {
    pub fn image_tokens(&self) -> usize {
        273 + if self.crops.is_empty() {
            0
        } else {
            (10 * self.grid[0] + 1) * 10 * self.grid[1]
        }
    }
}

fn coefficients(input: u32, output: u32) -> Vec<(usize, Vec<i32>)> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.);
    let support = 2. * filter_scale;
    (0..output)
        .map(|x| {
            let center = (x as f64 + 0.5) * scale;
            let start = ((center - support + 0.5) as i64).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input as usize);
            let mut weights: Vec<f64> = (start..end)
                .map(|i| {
                    let distance = ((i as f64 - center + 0.5) / filter_scale).abs();
                    if distance < 1. {
                        (1.5 * distance - 2.5) * distance * distance + 1.
                    } else if distance < 2. {
                        ((distance - 5.) * distance + 8.) * distance * (-0.5) + 2.
                    } else {
                        0.
                    }
                })
                .collect();
            let sum: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= sum;
            }
            (
                start,
                weights
                    .into_iter()
                    .map(|w| (w * (1u32 << 22) as f64).round() as i32)
                    .collect(),
            )
        })
        .collect()
}

/// Separable antialiased cubic resize with Pillow-compatible 22-bit coefficients.
pub fn resize_rgb(image: &RgbImage, width: u32, height: u32) -> Result<RgbImage> {
    if image.width() == 0
        || image.height() == 0
        || width == 0
        || height == 0
        || u64::from(width) * u64::from(height) > 100_000_000
        || u64::from(width) * u64::from(image.height()) > 100_000_000
    {
        return Err(OcrError::Resource(
            "invalid or excessive image dimensions".into(),
        ));
    }
    let mut horizontal = if image.width() == width {
        image.clone()
    } else {
        let weights = coefficients(image.width(), width);
        let mut out = RgbImage::new(width, image.height());
        for y in 0..image.height() {
            for x in 0..width {
                for c in 0..3 {
                    let (start, weights) = &weights[x as usize];
                    let sum = weights.iter().enumerate().fold(1i64 << 21, |s, (i, &w)| {
                        s + i64::from(image.get_pixel((*start + i) as u32, y)[c]) * i64::from(w)
                    });
                    out.get_pixel_mut(x, y)[c] = (sum >> 22).clamp(0, 255) as u8;
                }
            }
        }
        out
    };
    if image.height() == height {
        return Ok(horizontal);
    }
    let weights = coefficients(image.height(), height);
    let mut out = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let (start, weights) = &weights[y as usize];
                let sum = weights.iter().enumerate().fold(1i64 << 21, |s, (i, &w)| {
                    s + i64::from(horizontal.get_pixel(x, (*start + i) as u32)[c]) * i64::from(w)
                });
                out.get_pixel_mut(x, y)[c] = (sum >> 22).clamp(0, 255) as u8;
            }
        }
    }
    // Release the horizontal buffer before the caller builds normalized tensors.
    horizontal = RgbImage::new(0, 0);
    drop(horizontal);
    Ok(out)
}

fn normalize(image: &RgbImage) -> Result<Tensor> {
    let data = image
        .as_raw()
        .iter()
        .map(|&v| half::bf16::from_f32((v as f32 / 255. - 0.5) / 0.5).to_f32())
        .collect();
    Tensor::new(
        vec![1, image.height() as usize, image.width() as usize, 3],
        data,
    )
}

pub fn crop_grid(width: u32, height: u32) -> Result<[usize; 2]> {
    if width == 0 || height == 0 {
        return Err(OcrError::Shape("empty image".into()));
    }
    if width <= 640 && height <= 640 {
        return Ok([1, 1]);
    }
    let aspect = width as f64 / height as f64;
    let mut best = [1, 1];
    let mut difference = f64::INFINITY;
    for &(x, y) in crop_ratios::RATIOS {
        let delta = (aspect - x as f64 / y as f64).abs();
        if delta < difference
            || (delta == difference
                && width as f64 * height as f64 > 0.5 * 640. * 640. * x as f64 * y as f64)
        {
            best = [x as usize, y as usize];
            difference = delta;
        }
    }
    Ok(best)
}

pub fn prepare(path: &Path, mode: ImageMode) -> Result<PreparedImage> {
    let mut decoder = ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()
        .map_err(|e| OcrError::Format(e.to_string()))?;
    let (width, height) = decoder.dimensions();
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > 100_000_000 {
        return Err(OcrError::Resource(
            "image exceeds 100 million pixels".into(),
        ));
    }
    let orientation = decoder
        .orientation()
        .map_err(|e| OcrError::Format(e.to_string()))?;
    let mut decoded =
        DynamicImage::from_decoder(decoder).map_err(|e| OcrError::Format(e.to_string()))?;
    decoded.apply_orientation(orientation);
    let image = decoded.to_rgb8();
    let (w, h) = (image.width(), image.height());
    let (rw, rh) = if w > h {
        (1024, (h as f64 / w as f64 * 1024.).round_ties_even() as u32)
    } else {
        ((w as f64 / h as f64 * 1024.).round_ties_even() as u32, 1024)
    };
    let contained = resize_rgb(&image, rw, rh)?;
    let mut padded = RgbImage::from_pixel(1024, 1024, image::Rgb([127, 127, 127]));
    let offset_x = ((1024 - rw) as f64 / 2.).round_ties_even() as i64;
    let offset_y = ((1024 - rh) as f64 / 2.).round_ties_even() as i64;
    image::imageops::replace(&mut padded, &contained, offset_x, offset_y);
    let grid = if mode == ImageMode::Gundam {
        crop_grid(w, h)?
    } else {
        [1, 1]
    };
    let mut crops = Vec::new();
    if grid != [1, 1] {
        let resized = resize_rgb(&image, (grid[0] * 640) as u32, (grid[1] * 640) as u32)?;
        for y in 0..grid[1] {
            for x in 0..grid[0] {
                let crop = image::imageops::crop_imm(
                    &resized,
                    (x * 640) as u32,
                    (y * 640) as u32,
                    640,
                    640,
                )
                .to_image();
                crops.push(normalize(&crop)?);
            }
        }
    }
    Ok(PreparedImage {
        global: normalize(&padded)?,
        crops,
        grid,
    })
}
