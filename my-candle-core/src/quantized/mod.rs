

use crate::{  Tensor};
use crate::device::Device;
use crate::shape::Shape;
use crate::error::{Result,Error};

pub mod k_quants;
pub mod ggml_file;

pub use k_quants::GgmlType;

pub struct QTensor {
    data: Box<dyn QuantizedType>,
    shape:Shape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlDType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}


impl GgmlDType {
    pub(crate) fn from_u32(u: u32) ->Result<Self> {
        let dtype = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            _ => crate::bail!("unknown dtype for tensor {u}"),
        };
        Ok(dtype)
    }

    fn type_size(&self) -> usize {
        use k_quants::*;
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2K => std::mem::size_of::<BlockQ2K>(),
            Self::Q3K => std::mem::size_of::<BlockQ3K>(),
            Self::Q4K => std::mem::size_of::<BlockQ4K>(),
            Self::Q5K => std::mem::size_of::<BlockQ5K>(),
            Self::Q6K => std::mem::size_of::<BlockQ6K>(),
            Self::Q8K => std::mem::size_of::<BlockQ8K>(),
        }
    }

    fn blck_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::Q4_0 => k_quants::QK4_0,
            Self::Q4_1 => k_quants::QK4_1,
            Self::Q5_0 => k_quants::QK5_0,
            Self::Q5_1 => k_quants::QK5_1,
            Self::Q8_0 => k_quants::QK8_0,
            Self::Q8_1 => k_quants::QK8_1,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => k_quants::QK_K,
        }
    }
}

pub trait QuantizedType: Send + Sync {
    fn dtype(&self) -> GgmlDType;
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()>;
    fn to_float(&self, ys: &mut [f32]) -> Result<()>;
}

impl<T: k_quants::GgmlType + Send + Sync> QuantizedType for Vec<T> {
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        k_quants::matmul(mkn, lhs, self.as_slice(), dst)
    }

    fn dtype(&self) -> GgmlDType {
        T::DTYPE
    }

    fn to_float(&self, ys: &mut [f32]) -> Result<()> {
        T::to_float(self.as_slice(), ys)
    }
}

impl std::fmt::Debug for QTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "QTensor[{:?}; {:?}]", self.shape, self.dtype())
    }
}

impl QTensor {
    pub fn new<S: Into<Shape>, T: k_quants::GgmlType + Send + Sync + 'static>(
        data: Vec<T>,
        shape: S,
    ) -> Self {
        Self {
            data: Box::new(data),
            shape: shape.into(),
        }
    }

    pub fn dtype(&self) -> GgmlDType {
        self.data.dtype()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        let mut f32_data = vec![0f32; self.shape.elem_count()];
        self.data.to_float(&mut f32_data)?;
        Tensor::from_vec(f32_data, &self.shape, device)
    }

    pub fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        self.data.matmul_t(mkn, lhs, dst)
    }
}

#[derive(Debug, Clone)]
pub struct QMatMul(std::sync::Arc<QTensor>);

impl QMatMul {
    pub fn new(qtensor: std::sync::Arc<QTensor>) -> Self {
        Self(qtensor)
    }
}

impl crate::CustomOp1 for QMatMul {
    fn name(&self) -> &'static str {
        "qmatmul"
    }

    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        let (k, n) = self.0.shape.dims2()?;
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!(
                "input tensor {layout:?} incompatible with {:?}",
                self.0.shape
            )
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let storage = storage.as_slice::<f32>()?;
        let storage =
            &storage[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
        let mut dst_storage = vec![0f32; dst_shape.elem_count()];
        self.0.matmul_t(
            (dst_shape.elem_count() / n, k, n),
            storage,
            &mut dst_storage,
        )?;
        Ok((crate::CpuStorage::F32(dst_storage), dst_shape))
    }
}

