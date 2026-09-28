//! Checked row-major SGEMM, strided SGEMV, and vector exponentials.
use crate::error::{OcrError, Result};

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );
    fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// Matrix-vector product over a column interval of a token-major KV matrix.
pub fn matvec_strided(
    a: &[f32],
    x: &[f32],
    y: &mut [f32],
    rows: usize,
    cols: usize,
    stride: usize,
    transpose: bool,
) -> Result<()> {
    let end = rows
        .checked_sub(1)
        .and_then(|r| r.checked_mul(stride))
        .and_then(|n| n.checked_add(cols));
    let (input, output) = if transpose {
        (rows, cols)
    } else {
        (cols, rows)
    };
    if cols == 0
        || stride < cols
        || end.is_none_or(|n| n > a.len())
        || x.len() != input
        || y.len() != output
    {
        return Err(OcrError::Shape("strided GEMV storage mismatch".into()));
    }
    let convert =
        |n| i32::try_from(n).map_err(|_| OcrError::Shape("GEMV dimension overflow".into()));
    let (rows, cols, stride) = (convert(rows)?, convert(cols)?, convert(stride)?);
    #[cfg(target_os = "macos")]
    {
        // SAFETY: the checked last row endpoint covers every accessed matrix
        // element. Vector lengths match the selected transpose; stride and
        // dimensions fit CBLAS ints. Output is exclusive and cannot alias inputs.
        unsafe {
            cblas_sgemv(
                101,
                if transpose { 112 } else { 111 },
                rows,
                cols,
                1.,
                a.as_ptr(),
                stride,
                x.as_ptr(),
                1,
                0.,
                y.as_mut_ptr(),
                1,
            )
        };
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (rows, cols, stride);
        Err(OcrError::Unsupported("Accelerate requires macOS".into()))
    }
}

#[cfg(target_os = "macos")]
pub fn exp_in_place(values: &mut [f32]) {
    for chunk in values.chunks_mut(i32::MAX as usize) {
        let n = chunk.len() as i32;
        // SAFETY: vForce permits in-place operation. Both pointers address n
        // initialized floats in this exclusive slice, and n fits a C int.
        unsafe { vvexpf(chunk.as_mut_ptr(), chunk.as_ptr(), &n) };
    }
}

pub fn linear(x: &[f32], w: &[f32], y: &mut [f32], m: usize, n: usize, k: usize) -> Result<()> {
    if m.checked_mul(k) != Some(x.len())
        || n.checked_mul(k) != Some(w.len())
        || m.checked_mul(n) != Some(y.len())
    {
        return Err(OcrError::Shape("SGEMM storage mismatch".into()));
    }
    let [m, n, k] = [m, n, k]
        .map(|v| i32::try_from(v).map_err(|_| OcrError::Shape("BLAS dimension overflow".into())));
    let (m, n, k) = (m?, n?, k?);
    #[cfg(target_os = "macos")]
    {
        // SAFETY: checked slice lengths and i32 dimensions; X=[m,k], W=[n,k],
        // Y=[m,n]. Inputs cannot alias the exclusive output. CBLAS writes Y.
        unsafe {
            cblas_sgemm(
                101,
                111,
                112,
                m,
                n,
                k,
                1.0,
                x.as_ptr(),
                k,
                w.as_ptr(),
                k,
                0.0,
                y.as_mut_ptr(),
                n,
            )
        };
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (m, n, k);
        Err(OcrError::Unsupported("Accelerate requires macOS".into()))
    }
}
