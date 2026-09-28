use std::{hint::black_box, time::Instant};
use uocr::cpu::quantized::Int8Matrix;

fn main() {
    for (rows, cols) in [(896usize, 1280usize), (1280, 896), (257, 513)] {
        let matrix = Int8Matrix::from_parts(
            rows,
            cols,
            (0..rows * cols)
                .map(|i| ((i % 255) as i16 - 127) as i8)
                .collect(),
            (0..rows * cols.div_ceil(32))
                .map(|i| (i % 19 + 1) as f32 / 127.)
                .collect(),
        )
        .unwrap();
        for repeat in 0..4 {
            let start = Instant::now();
            for _ in 0..100 {
                black_box(matrix.dequantize().unwrap());
            }
            println!(
                "rows={rows} cols={cols} repeat={repeat} seconds={:.6}",
                start.elapsed().as_secs_f64()
            );
        }
    }
}
