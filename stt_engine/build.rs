fn main() {
    println!("cargo:rustc-link-search=native=./");
    println!("cargo:rustc-link-lib=dylib=sherpa");
    println!("cargo:rustc-link-arg=-Wl,-rpath,./");
}