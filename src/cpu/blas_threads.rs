//! Optional macOS 15+ thread-local BLAS policy, resolved without raising the
//! deployment target of the portable runtime.

use crate::error::{OcrError, Result};
use std::{
    ffi::{c_char, c_void},
    marker::PhantomData,
    rc::Rc,
    sync::OnceLock,
};

#[derive(Clone, Copy)]
struct Api {
    set: unsafe extern "C" fn(u32) -> i32,
    get: unsafe extern "C" fn() -> u32,
}

unsafe extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

fn api() -> Option<Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    *API.get_or_init(|| {
        // SAFETY: Darwin's RTLD_DEFAULT is ((void *) -2); both names are
        // terminated static C strings. Accelerate is linked for this process.
        let (set, get) = unsafe {
            let handle = -2isize as *mut c_void;
            (
                dlsym(handle, c"BLASSetThreading".as_ptr()),
                dlsym(handle, c"BLASGetThreading".as_ptr()),
            )
        };
        if set.is_null() || get.is_null() {
            return None;
        }
        // SAFETY: signatures match Accelerate's thread_api.h: an unsigned-int
        // BLAS_THREADING enum, int setter result, and enum getter result. Symbols
        // live in the process-linked framework and cannot be unloaded here.
        Some(unsafe {
            Api {
                set: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(u32) -> i32>(set),
                get: std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> u32>(get),
            }
        })
    })
}

pub(super) fn available() -> bool {
    api().is_some()
}

pub(super) struct Guard {
    api: Api,
    previous: u32,
    // Policy is thread-local: the guard must never move to another thread.
    _not_send: PhantomData<Rc<()>>,
}

pub(super) fn enter() -> Result<Option<Guard>> {
    let Some(api) = api() else { return Ok(None) };
    // SAFETY: exact resolved SDK signatures; 1 is SINGLE_THREADED. The setting
    // affects only this calling thread, not other requests or worker pools.
    let previous = unsafe { (api.get)() };
    if unsafe { (api.set)(1) } != 0 {
        return Err(OcrError::Unsupported(
            "single-threaded BLAS policy unavailable".into(),
        ));
    }
    Ok(Some(Guard {
        api,
        previous,
        _not_send: PhantomData,
    }))
}

impl Drop for Guard {
    fn drop(&mut self) {
        // SAFETY: guard is !Send, so this restores the same thread's previously
        // accepted policy, including during unwinding or nested calls.
        unsafe {
            (self.api.set)(self.previous);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_thread_guard_restores_nested_and_unwound_policy() {
        let Some(api) = api() else {
            eprintln!("BLAS thread API unavailable; guard test skipped");
            return;
        };
        eprintln!("BLAS thread API available; testing restoration");
        // SAFETY: dynamically resolved SDK function with its exact C signature.
        let original = unsafe { (api.get)() };
        {
            let _outer = enter().unwrap();
            assert_eq!(unsafe { (api.get)() }, 1);
            {
                let _inner = enter().unwrap();
                assert_eq!(unsafe { (api.get)() }, 1);
            }
            assert_eq!(unsafe { (api.get)() }, 1);
        }
        assert_eq!(unsafe { (api.get)() }, original);
        let caught = std::panic::catch_unwind(|| {
            let _guard = enter().unwrap();
            panic!("exercise unwinding");
        });
        assert!(caught.is_err());
        assert_eq!(unsafe { (api.get)() }, original);
    }
}
