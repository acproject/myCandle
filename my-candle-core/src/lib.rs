//! ML framework for Rust
//!
//! ```rust
//! use candle_core::{Tensor, DType, Device};
//! # use candle_core::Error;
//! # fn main() -> Result<(), Error>{
//!
//! let a = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
//! let b = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((3, 4))?;
//!
//! let c = a.matmul(&b)?;
//! # Ok(())}
//! ```
//!
//! ## Features
//!
//! - Simple syntax (looks and like PyTorch)
//! - CPU and Cuda backends (and M1 support)
//! - Enable serverless (CPU) small and fast deployments
//! - Model training
//! - Distributed computing (NCCL).
//! - Models out of the box (Llama, Whisper, Falcon, ...)
//!
//! ## FAQ
//!
//! - Why Candle?
//!
//! Candle stems from the need to reduce binary size in order to *enable serverless*
//! possible by making the whole engine smaller than PyTorch very large library volume
//!
//! And simply *removing Python* from production workloads.
//! Python can really add overhead in more complex workflows and the [GIL](https://www.backblaze.com/blog/the-python-gil-past-present-and-future/) is a notorious source of headaches.
//!
//! Rust is cool, and a lot of the HF ecosystem already has Rust crates [safetensors](https://github.com/huggingface/safetensors) and [tokenizers](https://github.com/huggingface/tokenizers)

#[cfg(feature = "accelerate")]
mod accelerate;
pub mod backend;
pub mod backprop;
mod conv;
mod convert;
pub mod cpu;
pub mod cpu_backend;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
#[cfg(feature = "cudnn")]
pub mod cudnn;
mod device;
pub mod display;
mod dtype;
mod dummy_cuda_backend;
pub mod error;
mod indexer;
pub mod layout;
#[cfg(feature = "mkl")]
mod mkl;
pub mod npy;
mod op;
pub mod quantized;
pub mod safetensors;
pub mod shape;
mod storage;
mod strided_index;
mod tensor;
pub mod utils;
mod variable;

pub use cpu_backend::CpuStorage;
pub use device::{Device, DeviceLocation};
pub use dtype::{DType, FloatDType, IntDType, WithDType};
pub use error::{Error, Result};
pub use indexer::IndexOp;
pub use layout::Layout;
pub use op::{CustomOp1, CustomOp2, CustomOp3};
pub use shape::{Shape, D};
pub use storage::Storage;
pub use strided_index::{StridedBlocks, StridedIndex};
pub use tensor::{Tensor, TensorId};
pub use variable::Var;

#[cfg(feature = "cuda")]
pub use cuda_backend::{CudaDevice, CudaStorage};

#[cfg(not(feature = "cuda"))]
pub use dummy_cuda_backend::{CudaDevice, CudaStorage};

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;


pub trait ToUsize2 {
    fn to_usize2(self) -> (usize, usize);
}

impl ToUsize2 for usize {
    fn to_usize2(self) -> (usize, usize) {
        (self, self)
    }
}

impl ToUsize2 for (usize, usize) {
    fn to_usize2(self) -> (usize, usize) {
        self
    }
}

pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

impl Module for quantized::QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}

impl<T: Fn(&Tensor) -> Result<Tensor>> Module for T {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self(xs)
    }
}