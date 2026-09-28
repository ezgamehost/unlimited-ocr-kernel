use crate::{
    error::{OcrError, Result},
    tensor::{Tensor, element_count},
};

fn zeroed_window_storage(shape: &[usize]) -> Result<Vec<f32>> {
    let count = element_count(shape)?;
    let mut data = Vec::new();
    data.try_reserve_exact(count)
        .map_err(|e| OcrError::Resource(format!("window allocation: {e}")))?;
    data.resize(count, 0.);
    Ok(data)
}

pub fn window_partition(x: &Tensor, window: usize) -> Result<Tensor> {
    x.validate()?;
    if x.shape.len() != 4 || window == 0 || x.shape.contains(&0) {
        return Err(OcrError::Shape("window input layout".into()));
    }
    let (batch, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let (ny, nx) = (h.div_ceil(window), w.div_ceil(window));
    let windows = element_count(&[batch, ny, nx])?;
    let shape = vec![windows, window, window, c];
    let mut data = zeroed_window_storage(&shape)?;
    for b in 0..batch {
        for y in 0..h {
            for xx in 0..w {
                let target = ((((b * ny + y / window) * nx + xx / window) * window + y % window)
                    * window
                    + xx % window)
                    * c;
                let source = ((b * h + y) * w + xx) * c;
                data[target..target + c].copy_from_slice(&x.data[source..source + c]);
            }
        }
    }
    Tensor::new(shape, data)
}

pub fn window_unpartition(x: &Tensor, h: usize, w: usize, window: usize) -> Result<Tensor> {
    x.validate()?;
    if x.shape.len() != 4
        || x.shape.contains(&0)
        || h == 0
        || w == 0
        || window == 0
        || x.shape[1] != window
        || x.shape[2] != window
    {
        return Err(OcrError::Shape("window output layout".into()));
    }
    let (ny, nx) = (h.div_ceil(window), w.div_ceil(window));
    let windows = element_count(&[ny, nx])?;
    if !x.shape[0].is_multiple_of(windows) {
        return Err(OcrError::Shape("window batch mismatch".into()));
    }
    let batch = x.shape[0] / windows;
    let c = x.shape[3];
    let shape = vec![batch, h, w, c];
    let mut data = zeroed_window_storage(&shape)?;
    for b in 0..batch {
        for y in 0..h {
            for xx in 0..w {
                let source = ((((b * ny + y / window) * nx + xx / window) * window + y % window)
                    * window
                    + xx % window)
                    * c;
                let target = ((b * h + y) * w + xx) * c;
                data[target..target + c].copy_from_slice(&x.data[source..source + c]);
            }
        }
    }
    Tensor::new(shape, data)
}

fn cubic(x: f32) -> f32 {
    let x = x.abs();
    if x < 1. {
        (1.5 * x - 2.5) * x * x + 1.
    } else if x < 2. {
        ((x - 5.) * x + 8.) * x * (-0.5) + 2.
    } else {
        0.
    }
}

fn axis(source: usize, target: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = source as f32 / target as f32;
    let filter = scale.max(1.);
    (0..target)
        .map(|i| {
            let center = (i as f32 + 0.5) * scale;
            let start = ((center - 2. * filter + 0.5) as isize).max(0) as usize;
            let end = ((center + 2. * filter + 0.5) as usize).min(source);
            let mut weights: Vec<_> = (start..end)
                .map(|j| cubic((j as f32 - center + 0.5) / filter))
                .collect();
            let sum: f32 = weights.iter().sum();
            for w in &mut weights {
                *w /= sum;
            }
            (start, weights)
        })
        .collect()
}

/// PyTorch antialiased bicubic interpolation for learned positional grids (NHWC).
pub fn interpolate_grid(x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    if x.shape.len() != 4 || x.shape[0] != 1 || h == 0 || w == 0 {
        return Err(OcrError::Shape("positional grid dimensions".into()));
    }
    let (ih, iw, c) = (x.shape[1], x.shape[2], x.shape[3]);
    if (ih, iw) == (h, w) {
        return Ok(x.clone());
    }
    let ax = axis(iw, w);
    let ay = axis(ih, h);
    let mut temp = vec![0.; ih * w * c];
    for y in 0..ih {
        for (xx, (start, weights)) in ax.iter().enumerate() {
            for (j, &weight) in weights.iter().enumerate() {
                for channel in 0..c {
                    temp[(y * w + xx) * c + channel] +=
                        weight * x.data[(y * iw + start + j) * c + channel];
                }
            }
        }
    }
    let mut data = vec![0.; h * w * c];
    for (y, (start, weights)) in ay.iter().enumerate() {
        for (j, &weight) in weights.iter().enumerate() {
            for xx in 0..w {
                for channel in 0..c {
                    data[(y * w + xx) * c + channel] +=
                        weight * temp[((start + j) * w + xx) * c + channel];
                }
            }
        }
    }
    Tensor::new(vec![1, h, w, c], data)
}

pub fn interpolate_relative(x: &Tensor, size: usize) -> Result<Tensor> {
    if x.shape.len() != 2 || x.shape.contains(&0) || size == 0 {
        return Err(OcrError::Shape("relative position table".into()));
    }
    if x.shape[0] == size {
        return Ok(x.clone());
    }
    let (old, c) = (x.shape[0], x.shape[1]);
    let mut data = vec![0.; size * c];
    for i in 0..size {
        let position = ((i as f32 + 0.5) * old as f32 / size as f32 - 0.5).max(0.);
        let left = (position as usize).min(old - 1);
        let right = (left + 1).min(old - 1);
        let mix = position - left as f32;
        for j in 0..c {
            data[i * c + j] = (1. - mix) * x.data[left * c + j] + mix * x.data[right * c + j];
        }
    }
    Tensor::new(vec![size, c], data)
}
