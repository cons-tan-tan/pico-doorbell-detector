use core::{ffi::c_char, marker::PhantomData};
use portable_atomic::{AtomicBool, Ordering};

use crate::{config, frontend::FEATURES};

#[repr(C, align(16))]
struct AlignedModel(
    [u8; include_bytes!(concat!(env!("DOORBELL_MODEL_DIR"), "/model.tflite")).len()],
);
static MODEL: AlignedModel = AlignedModel(*include_bytes!(concat!(
    env!("DOORBELL_MODEL_DIR"),
    "/model.tflite"
)));
static TAKEN: AtomicBool = AtomicBool::new(false);

unsafe extern "C" {
    fn doorbell_init(
        data: *const u8,
        len: usize,
        input_scale: f32,
        input_zero: i32,
        output_scale: f32,
        output_zero: i32,
    ) -> i32;
    fn doorbell_invoke(input: *const i8, len: usize, output: *mut i8, output_len: usize) -> i32;
    fn doorbell_arena_used() -> usize;
}

pub struct Model {
    _not_send_or_sync: PhantomData<*mut ()>,
}

impl Model {
    pub fn take() -> Result<Self, i32> {
        if TAKEN.swap(true, Ordering::AcqRel) {
            return Err(-1);
        }
        // The model and C++ arena live for the program lifetime; TAKEN excludes reentry.
        let status = unsafe {
            doorbell_init(
                MODEL.0.as_ptr(),
                MODEL.0.len(),
                config::INPUT_SCALE,
                config::INPUT_ZERO_POINT,
                config::OUTPUT_SCALE,
                config::OUTPUT_ZERO_POINT,
            )
        };
        if status != 0 {
            return Err(status);
        }
        Ok(Self {
            _not_send_or_sync: PhantomData,
        })
    }

    pub fn predict(&mut self, features: &[i8; FEATURES]) -> Result<[i8; 3], i32> {
        let mut output = [0; 3];
        // Exclusive access and fixed-size buffers keep the foreign pointers valid during Invoke.
        let status = unsafe {
            doorbell_invoke(
                features.as_ptr(),
                features.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        if status == 0 { Ok(output) } else { Err(status) }
    }

    pub fn arena_used(&self) -> usize {
        unsafe { doorbell_arena_used() }
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn doorbell_log(message: *const c_char) {
    // C++ passes a terminated stack buffer and does not retain it after this callback.
    if let Ok(text) = unsafe { core::ffi::CStr::from_ptr(message) }.to_str() {
        log::warn!("TFLM: {}", text);
    }
}
