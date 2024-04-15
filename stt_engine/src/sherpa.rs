use derive_new::new;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SherpaHandle {
    pub recognizer: *const (),
    pub stream: *const (),
}

#[link(name = "sherpa")]
extern "C" {
    pub fn sherpa_init(tokens: *const ::std::os::raw::c_char,
                       encoder: *const ::std::os::raw::c_char,
                       decoder: *const ::std::os::raw::c_char,
                       joiner: *const ::std::os::raw::c_char) -> SherpaHandle;
    pub fn sherpa_transcribe(handle: SherpaHandle, samples: *const f32, len: ::std::os::raw::c_int) -> *const ::std::os::raw::c_char;
    pub fn sherpa_close(handle: SherpaHandle);
}

#[repr(C)]
#[derive(Debug, Copy, Clone, new)]
pub struct Sherpa;

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
        let result = unsafe { sherpa_transcribe(handle, samples.as_ptr(), len) };
        let c_str = unsafe { ::std::ffi::CStr::from_ptr(result) };
        c_str.to_str().unwrap_or("").to_string() // Error handling should be more robust in production code
    }

    pub fn close(&self, handle: SherpaHandle) {
        unsafe { sherpa_close(handle) };
    }
}