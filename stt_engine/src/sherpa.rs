use std::sync::atomic::{AtomicBool, Ordering};

const MAX_SUPPORT_TOKENS: usize = 2048;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SherpaHandle {
    pub recognizer: *const (),
    pub stream: *const (),
}

unsafe impl Send for SherpaHandle { }
unsafe impl Sync for SherpaHandle { }

#[link(name = "sherpa")]
extern "C" {
    pub fn sherpa_init(
        tokens: *const ::std::os::raw::c_char,
        encoder: *const ::std::os::raw::c_char,
        decoder: *const ::std::os::raw::c_char,
        joiner: *const ::std::os::raw::c_char) -> SherpaHandle;
    pub fn sherpa_transcribe(
            handle: SherpaHandle,
            result: *mut ::std::os::raw::c_char,
            samples: *const f32,
            len: ::std::os::raw::c_int);
    pub fn sherpa_reset(handle: SherpaHandle);
    pub fn sherpa_close(handle: SherpaHandle);
}

#[repr(C)]
#[derive(Debug)]
pub(crate) struct Sherpa {
    busy: AtomicBool,
}

impl Sherpa {
    pub(crate) fn new() -> Self {
        Self { busy: AtomicBool::new(false) }
    }

    pub(crate) fn is_busy(&self) -> bool {
        self.busy.load(Ordering::SeqCst)
    }

    pub(crate) fn set_busy(&self, available: bool) {
        self.busy.store(available, Ordering::SeqCst);
    }
}

impl Sherpa {
    pub fn init(&self, tokens: &str, encoder: &str, decoder: &str, joiner: &str) -> SherpaHandle {
        let tokens_cstr = ::std::ffi::CString::new(tokens).unwrap();
        let encoder_cstr = ::std::ffi::CString::new(encoder).unwrap();
        let decoder_cstr = ::std::ffi::CString::new(decoder).unwrap();
        let joiner_cstr = ::std::ffi::CString::new(joiner).unwrap();

        unsafe {
            sherpa_init(tokens_cstr.as_ptr(), encoder_cstr.as_ptr(), decoder_cstr.as_ptr(), joiner_cstr.as_ptr())
        }
    }

    pub fn transcribe(&self, handle: SherpaHandle, samples: &[f32]) -> String {
        let len = samples.len() as i32;
        let mut result_buf = [0_u8; MAX_SUPPORT_TOKENS];
        let result_ptr = result_buf.as_mut_ptr() as *mut ::std::os::raw::c_char;  

        unsafe { sherpa_transcribe(handle, result_ptr, samples.as_ptr(), len) };

        let c_str = unsafe { ::std::ffi::CStr::from_ptr(result_ptr) };
        c_str.to_str().unwrap_or("").to_string() // Error handling should be more robust in production code
    }

    pub fn reset(&self, handle: SherpaHandle) {
        unsafe { sherpa_reset(handle) };
    }

    pub fn close(&self, handle: SherpaHandle) {
        unsafe { sherpa_close(handle) };
    }
}