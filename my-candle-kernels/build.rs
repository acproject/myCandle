use std::io::Write;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    cuda::set_include_dir();
    let (write, kernel_paths) = cuda::build_ptx();
}