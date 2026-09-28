use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
use uocr::{
    cpu::{Backend, CpuContext, ops},
    tensor::Tensor,
};

struct ObservedAllocator;
thread_local! {
    static LARGEST: Cell<Option<usize>> = const { Cell::new(None) };
}

fn observe(bytes: usize) {
    let _ = LARGEST.try_with(|value| {
        if let Some(previous) = value.get() {
            value.set(Some(previous.max(bytes)));
        }
    });
}

// SAFETY: All operations delegate unchanged pointers and layouts to System;
// thread-local accounting does not allocate or inspect the allocated storage.
unsafe impl GlobalAlloc for ObservedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        observe(layout.size());
        // SAFETY: The caller supplies a valid allocation layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        observe(layout.size());
        // SAFETY: The caller supplies a valid allocation layout.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        observe(size);
        // SAFETY: The original pointer/layout and new size come from the caller.
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: Every allocation above belongs to System with this layout.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: ObservedAllocator = ObservedAllocator;

#[test]
fn convolution_does_not_clone_weights_outside_its_scratch_budget() {
    let x = Tensor::new(vec![1, 1, 1, 64], vec![1.; 64]).unwrap();
    let w = Tensor::new(vec![128, 64, 1, 1], vec![1.; 128 * 64]).unwrap();
    let budget = (64 + 128) * 4;
    let mut cpu = CpuContext::new(Backend::Scalar, 1, budget).unwrap();
    LARGEST.with(|value| value.set(Some(0)));
    let result = ops::conv2d(&x, &w, None, 1, 0, &mut cpu);
    let largest = LARGEST.with(|value| value.replace(None).unwrap());
    let result = result.unwrap();
    assert_eq!(result.data, vec![64.; 128]);
    assert!(
        largest <= budget,
        "allocation of {largest} bytes exceeds {budget}-byte budget"
    );
    let mut too_small = CpuContext::new(Backend::Scalar, 1, budget - 1).unwrap();
    assert!(ops::conv2d(&x, &w, None, 1, 0, &mut too_small).is_err());
}
