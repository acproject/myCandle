use std::path::Path;
use std::collections::HashMap;
use std::borrow::Cow;
use crate::dtype::{DType, FloatDType, WithDType};
use crate::tensor::Tensor;
use crate::device::Device;
use crate::error::{Result, Error};
use crate::shape::Shape;

use safetensors::tensor as st;
use safetensors::tensor::SafeTensors;

impl From<DType> for st::Dtype {
    fn from(value: DType) -> Self {
        match value {
            DType::U8 => st::Dtype::U8,
            DType::U32 => st::Dtype::U32,
            DType::BF16 => st::Dtype::BF16,
            DType::F16 => st::Dtype::F16,
            DType::F32 => st::Dtype::F32,
            DType::F64 => st::Dtype::F64,
        }
    }
}

impl TryFrom<st::Dtype> for DType {
    type Error = Error;
    fn try_from(value: st::Dtype) -> Result<Self> {
        match value {
            st::Dtype::U8 => Ok(DType::U8),
            st::Dtype::U32 => Ok(DType::U32),
            st::Dtype::BF16 => Ok(DType::BF16),
            st::Dtype::F16 => Ok(DType::F16),
            st::Dtype::F32 => Ok(DType::F32),
            st::Dtype::F64 => Ok(DType::F64),
            dtype => Err(Error::UnsupportedSafeTensorDtype(dtype)),
        }
    }
}

impl st::View for Tensor {
    fn dtype(&self) -> st::Dtype {
        self.dtype().into()
    }
    fn shape(&self) -> &[usize] {
        self.shape().dims()
    }

    fn data(&self) -> Cow<[u8]> {
        // This copies data from GPU to CPU.
        // TODO: Avoid the unwrap here.
        Cow::Owned(convert_back(self).unwrap())
    }

    fn data_len(&self) -> usize {
        let n: usize = self.shape().elem_count();
        let bytes_per_element = self.dtype().size_in_bytes();
        n * bytes_per_element
    }
}

impl st::View for &Tensor {
    fn dtype(&self) -> st::Dtype {
        (*self).dtype().into()
    }
    fn shape(&self) -> &[usize] {
        self.dims()
    }

    fn data(&self) -> Cow<[u8]> {
        // This copies data from GPU to CPU.
        // TODO: Avoid the unwrap here.
        Cow::Owned(convert_back(self).unwrap())
    }

    fn data_len(&self) -> usize {
        let n: usize = self.dims().iter().product();
        let bytes_per_element = (*self).dtype().size_in_bytes();
        n * bytes_per_element
    }
}

impl Tensor {
    pub fn save_safetensors<P: AsRef<std::path::Path>>(
        &self,
        name: &str,
        filename: P,
    ) -> Result<()> {
        let data = [(name, self.clone())];
        Ok(st::serialize_to_file(data, &None, filename.as_ref())?)
    }
}

fn convert_slice<T: WithDType>(data: &[u8], shape: &[usize], device: &Device) -> Result<Tensor> {
    let size_in_bytes = T::DTYPE.size_in_bytes();
    let elem_count = data.len() / size_in_bytes;
    if (data.as_ptr() as usize) % size_in_bytes == 0 {
        // SAFETY This is safe because we just checked that this
        // was correctly aligned.
        let data: &[T] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, elem_count) };
        Tensor::from_slice(data, shape, device)
    } else {
        // XXX: We need to specify `T` here, otherwise the compiler will infer u8 because of the following cast
        // Making this vector too small to fit a full f16/f32/f64 weights, resulting in out-of-bounds access
        let mut c: Vec<T> = Vec::with_capacity(elem_count);
        // SAFETY: We just created c, so the allocated memory is necessarily
        // contiguous and non overlapping with the view's data.
        // We're downgrading the `c` pointer from T to u8, which removes alignment
        // constraints.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), c.as_mut_ptr() as *mut u8, data.len());
            c.set_len(elem_count)
        }
        Tensor::from_slice(&c, shape, device)
    }
}

fn convert_slice_with_cast<T: Sized + Copy, U: WithDType, F: Fn(T) -> Result<U>>(
    data: &[u8],
    shape: &[usize],
    device: &Device,
    conv: F,
) -> Result<Tensor> {
    let size_in_bytes = std::mem::size_of::<T>();
    let elem_count = data.len() / size_in_bytes;
    if (data.as_ptr() as usize) % size_in_bytes == 0 {
        // SAFETY This is safe because we just checked that this
        // was correctly aligned.
        let data: &[T] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, elem_count) };
        let data = data.iter().map(|t| conv(*t)).collect::<Result<Vec<_>>>()?;
        Tensor::from_vec(data, shape, device)
    } else {
        // XXX: We need to specify `T` here, otherwise the compiler will infer u8 because of the following cast
        // Making this vector too small to fit a full f16/f32/f64 weights, resulting in out-of-bounds access
        let mut c: Vec<T> = Vec::with_capacity(elem_count);
        // SAFETY: We just created c, so the allocated memory is necessarily
        // contiguous and non overlapping with the view's data.
        // We're downgrading the `c` pointer from T to u8, which removes alignment
        // constraints.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), c.as_mut_ptr() as *mut u8, data.len());
            c.set_len(elem_count)
        }
        let c = c.into_iter().map(conv).collect::<Result<Vec<_>>>()?;
        Tensor::from_vec(c, shape, device)
    }
}

fn convert_with_cast_<T: Sized + Copy, U: WithDType, F: Fn(T) -> Result<U>>(
    view: &st::TensorView<'_>,
    device: &Device,
    conv: F,
) -> Result<Tensor> {
    convert_slice_with_cast::<T, U, F>(view.data(), view.shape(), device, conv)
}

fn convert_<T: WithDType>(view: &st::TensorView<'_>, device: &Device) -> Result<Tensor> {
    convert_slice::<T>(view.data(), view.shape(), device)
}

fn convert_back_<T: WithDType>(mut vs: Vec<T>) -> Vec<u8> {
    let size_in_bytes = T::DTYPE.size_in_bytes();
    let length = vs.len() * size_in_bytes;
    let capacity = vs.capacity() * size_in_bytes;
    let ptr = vs.as_mut_ptr() as *mut u8;
    // Don't run the destructor for Vec<T>
    std::mem::forget(vs);
    // SAFETY:
    //
    // Every T is larger than u8, so there is no issue regarding alignment.
    // This re-interpret the Vec<T> as a Vec<u8>.
    unsafe { Vec::from_raw_parts(ptr, length, capacity) }
}

pub trait Load {
    fn load(&self, device: &Device) -> Result<Tensor>;
}

impl<'a> Load for st::TensorView<'a> {
    fn load(&self, device: &Device) -> Result<Tensor> {
        convert(self, device)
    }
}

impl Tensor {
    pub fn from_raw_buffer(
        data: &[u8],
        dtype: DType,
        shape: &[usize],
        device: &Device,
    ) -> Result<Self> {
        match dtype {
            DType::U8 => convert_slice::<u8>(data, shape, device),
            DType::U32 => convert_slice::<u32>(data, shape, device),
            DType::BF16 => convert_slice::<half::bf16>(data, shape, device),
            DType::F16 => convert_slice::<half::f16>(data, shape, device),
            DType::F32 => convert_slice::<f32>(data, shape, device),
            DType::F64 => convert_slice::<f64>(data, shape, device),
        }
    }
}

fn convert(view: &st::TensorView<'_>, device: &Device) -> Result<Tensor> {
    match view.dtype() {
        st::Dtype::U8 => convert_::<u8>(view, device),
        st::Dtype::U16 => {
            let conv = |x| Ok(u32::from(x));
            convert_with_cast_::<u16, u32, _>(view, device, conv)
        }
        st::Dtype::U32 => convert_::<u32>(view, device),
        st::Dtype::BF16 => convert_::<half::bf16>(view, device),
        st::Dtype::F16 => convert_::<half::f16>(view, device),
        st::Dtype::F32 => convert_::<f32>(view, device),
        st::Dtype::F64 => convert_::<f64>(view, device),
        st::Dtype::I32 => {
            let conv = |x| {
                u32::try_from(x)
                    .map_err(|_| Error::Msg(format!("out of bounds value for u32: {x}")))
            };
            convert_with_cast_::<i32, u32, _>(view, device, conv)
        }
        st::Dtype::I64 => {
            let conv = |x| {
                u32::try_from(x)
                    .map_err(|_| Error::Msg(format!("out of bounds value for u32: {x}")))
            };
            convert_with_cast_::<i64, u32, _>(view, device, conv)
        }
        dtype => Err(Error::UnsupportedSafeTensorDtype(dtype)),
    }
}

fn convert_back(tensor: &Tensor) -> Result<Vec<u8>> {
    // TODO: This makes an unnecessary copy when the tensor is on the cpu.
    let tensor = tensor.flatten_all()?;
    match tensor.dtype() {
        DType::U8 => Ok(convert_back_::<u8>(tensor.to_vec1()?)),
        DType::U32 => Ok(convert_back_::<u32>(tensor.to_vec1()?)),
        DType::F16 => Ok(convert_back_::<half::f16>(tensor.to_vec1()?)),
        DType::BF16 => Ok(convert_back_::<half::bf16>(tensor.to_vec1()?)),
        DType::F32 => Ok(convert_back_::<f32>(tensor.to_vec1()?)),
        DType::F64 => Ok(convert_back_::<f64>(tensor.to_vec1()?)),
    }
}

pub fn load<P: AsRef<Path>>(filename: P, device: &Device) -> Result<HashMap<String, Tensor>> {
    let data = std::fs::read(filename.as_ref())?;
    load_buffer(&data[..], device)
}

pub fn load_buffer(data: &[u8], device: &Device) -> Result<HashMap<String, Tensor>> {
    let st = safetensors::SafeTensors::deserialize(data)?;
    st.tensors()
        .into_iter()
        .map(|(name, view)| Ok((name, view.load(device)?)))
        .collect()
}

pub fn save<K: AsRef<str> + Ord + std::fmt::Display, P: AsRef<Path>>(
    tensors: &HashMap<K, Tensor>,
    filename: P,
) -> Result<()> {
    Ok(st::serialize_to_file(tensors, &None, filename.as_ref())?)
}

pub struct MmapedFile {
    path: std::path::PathBuf,
    inner: memmap2::Mmap,
}

impl MmapedFile {
    /// Creates a wrapper around a memory mapped file from which you can retrieve
    /// tensors using [`MmapedFile::deserialize`]
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`memmap2::MmapOptions`].
    pub unsafe fn new<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let p = p.as_ref();
        let file = std::fs::File::open(p).map_err(|e| Error::from(e).with_path(p))?;
        let inner = memmap2::MmapOptions::new()
            .map(&file)
            .map_err(|e| Error::from(e).with_path(p))?;
        Ok(Self {
            inner,
            path: p.to_path_buf(),
        })
    }

    pub fn deserialize(&self) -> Result<SafeTensors<'_>> {
        let st = safetensors::SafeTensors::deserialize(&self.inner)
            .map_err(|e| Error::from(e).with_path(&self.path))?;
        Ok(st)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn save_single_tensor() {
        let t = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        t.save_safetensors("t", "t.safetensors").unwrap();
        let bytes = std::fs::read("t.safetensors").unwrap();
        assert_eq!(bytes, b"@\0\0\0\0\0\0\0{\"t\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}       \0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0");
        std::fs::remove_file("t.safetensors").unwrap();
    }

    #[test]
    fn save_load_multiple_tensors() {
        let t = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        let u = Tensor::zeros((1, 2), DType::F32, &Device::Cpu).unwrap();
        let map: HashMap<_, _> = [("t", t), ("u", u)].into_iter().collect();
        save(&map, "multi.safetensors").unwrap();

        let weights = load("multi.safetensors", &Device::Cpu).unwrap();
        assert_eq!(weights.get("t").unwrap().dims(), &[2, 2]);
        assert_eq!(weights.get("u").unwrap().dims(), &[1, 2]);
        let bytes = std::fs::read("multi.safetensors").unwrap();
        assert_eq!(bytes, b"x\0\0\0\0\0\0\0{\"t\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"u\":{\"dtype\":\"F32\",\"shape\":[1,2],\"data_offsets\":[16,24]}}      \0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0");
        std::fs::remove_file("multi.safetensors").unwrap();
    }
}