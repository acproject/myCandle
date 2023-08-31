use crate::backend::BackendDevice;
use crate::cpu_backend::{CpuDevice, CpuStorage};
use crate::dtype::{DType, FloatDType, WithDType};
use crate::error::{Error, Result};
use crate::cuda_backend::CudaDevice;
use crate::shape::Shape;
use crate::storage::Storage;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DeviceLocation {
    Cpu,
    Cuda {gpu_id: usize},
}

#[derive(Debug, Clone)]
pub enum  Device {
    Cpu,
    Cuda(CudaDevice),
}

pub trait NdArray {
    fn shape(&self) -> Result<Shape>;

    fn to_cpu_storage(&self) -> CpuStorage;
}

impl<S: WithDType> NdArray for S {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(&[*self])
    }
}

impl<S: WithDType, const N: usize> NdArray for &[S; N] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(self.as_slice())
    }
}

impl<S: WithDType> NdArray for &[S] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(self)
    }
}

impl<S: WithDType, const N: usize, const M: usize> NdArray for &[[S; N]; M] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((M, N)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage_owned(self.concat())
    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize> NdArray
for &[[[S; N3]; N2]; N1]
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let mut vec = Vec::with_capacity(N1 * N2 * N3);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                vec.extend(self[i1][i2])
            }
        }
        S::to_cpu_storage_owned(vec)
    }
}

impl Device {
    pub fn new_cuda(ordinal: usize) -> Result<Self> {
        Ok(Self::Cuda(crate::cuda_backend::CudaDevice::new(ordinal)?))
    }

    pub fn same_device(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Cpu, Self::Cpu) => true,
            (Self::Cuda(lhs), Self::Cuda(rhs)) => lhs.same_device(rhs),
            _ => false,
        }
    }

    pub fn location(&self) -> DeviceLocation {
        match self {
            Self::Cpu => DeviceLocation::Cpu,
            Self::Cuda(device) => device.location(),
        }
    }

    pub fn is_cpu(&self) -> bool {
        match self {
            Self::Cpu => true,
            Self::Cuda(_) => false,
        }
    }

    pub fn is_cuda(&self) -> bool {
        match self {
            Self::Cpu => false,
            Self::Cuda(_) => true,
        }
    }

    pub fn cuda_if_available(ordinal: usize) -> Result<Self> {
        if crate::utils::cuda_is_available() {
            Self::new_cuda(ordinal)
        } else {
            Ok(Self::Cpu)
        }
    }

    pub(crate) fn rand_uniform_f64(
        &self,
        lo: f64,
        up: f64,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.rand_uniform(shape, dtype, lo, up)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                let storage = device.rand_uniform(shape, dtype, lo, up)?;
                Ok(Storage::Cuda(storage))
            }
        }
    }

    pub(crate) fn rand_uniform<T: FloatDType>(
        &self,
        lo: T,
        up: T,
        shape: &Shape,
    ) -> Result<Storage> {
        self.rand_uniform_f64(lo.to_f64(), up.to_f64(), shape, T::DTYPE)
    }

    pub(crate) fn rand_normal_f64(
        &self,
        mean: f64,
        std: f64,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.rand_normal(shape, dtype, mean, std)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                let storage = device.rand_normal(shape, dtype, mean, std)?;
                Ok(Storage::Cuda(storage))
            }
        }
    }

    pub(crate) fn rand_normal<T: FloatDType>(
        &self,
        mean: T,
        std: T,
        shape: &Shape,
    ) -> Result<Storage> {
        self.rand_normal_f64(mean.to_f64(), std.to_f64(), shape, T::DTYPE)
    }

    pub(crate) fn ones(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.ones_impl(shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                let storage = device.ones_impl(shape, dtype)?;
                Ok(Storage::Cuda(storage))
            }
        }
    }

    pub(crate) fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.zeros_impl(shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                let storage = device.zeros_impl(shape, dtype)?;
                Ok(Storage::Cuda(storage))
            }
        }
    }

    pub(crate) fn storage<A: NdArray>(&self, array: A) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(array.to_cpu_storage())),
            Device::Cuda(device) => {
                let storage = array.to_cpu_storage();
                let storage = device.storage_from_cpu_storage(&storage)?;
                Ok(Storage::Cuda(storage))
            }
        }
    }

    pub(crate) fn storage_owned<S: WithDType>(&self, data: Vec<S>) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(S::to_cpu_storage_owned(data))),
            Device::Cuda(device) => {
                let storage = S::to_cpu_storage_owned(data);
                let storage = device.storage_from_cpu_storage(&storage)?;
                Ok(Storage::Cuda(storage))
            }
        }
    }
}