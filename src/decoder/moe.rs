use crate::{
    cpu::ops,
    error::{OcrError, Result},
};

/// Stable tie-break by expert id; probabilities are not renormalized by default.
pub fn route(
    logits: &[f32],
    count: usize,
    normalize: bool,
    scale: f32,
) -> Result<Vec<(usize, f32)>> {
    if count == 0 || count > logits.len() || !scale.is_finite() || scale <= 0. {
        return Err(OcrError::Shape("invalid expert routing".into()));
    }
    let mut probabilities = logits.to_vec();
    ops::softmax(&mut probabilities)?;
    let mut indices: Vec<_> = probabilities.into_iter().enumerate().collect();
    indices.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    indices.truncate(count);
    let denominator = if normalize {
        indices.iter().map(|x| x.1).sum::<f32>() + 1e-20
    } else {
        1.
    };
    for (_, weight) in &mut indices {
        *weight = *weight / denominator * scale;
    }
    #[cfg(test)]
    ROUTE_TRACE.with(|trace| {
        if let Some(rows) = trace.borrow_mut().as_mut() {
            rows.push(indices.clone());
        }
    });
    Ok(indices)
}

#[cfg(test)]
type RouteTrace = Vec<Vec<(usize, f32)>>;
#[cfg(test)]
thread_local! {
    static ROUTE_TRACE: std::cell::RefCell<Option<RouteTrace>> = const { std::cell::RefCell::new(None) };
}

/// Capture actual routing on the calling worker; compiled out of the runtime.
#[cfg(test)]
pub(super) fn capture_routes<T>(operation: impl FnOnce() -> T) -> (T, RouteTrace) {
    ROUTE_TRACE.with(|trace| {
        assert!(trace.borrow().is_none(), "nested route capture");
        *trace.borrow_mut() = Some(Vec::new());
    });
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ROUTE_TRACE.with(|trace| {
                trace.borrow_mut().take();
            });
        }
    }
    let _reset = Reset;
    let result = operation();
    let rows = ROUTE_TRACE.with(|trace| trace.borrow_mut().take().expect("route capture active"));
    (result, rows)
}
