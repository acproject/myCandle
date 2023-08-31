//!
//! 主要的类型都定义在了这里，作为一个专用的DSL，定义自己的内容是非常有必要的
//! 同时这里提供了对于DSL中专用的类型的一些简单操作，比如：从字符串中去转
//!
use half::{bf16, f16};
use crate::cpu_backend::CpuStorage;
use Error::{Error,Result};
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DType {
    U8,
    U32,
    BF16,
    F16,
    F32,
    F64,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DTypeParseError;

impl std::str::FromStr for DType {
    type Err = DTypeParseError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            // core::result::Result::Ok
            "u8" => Ok(Self::U8),
            "u32" => Ok(Self::U32),
            "bf16" => Ok(Self::BF16),
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            // core::result::Result::Err
            _ => Err(DTypeParseError),
        }
    }
}

impl DType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U32 => "u32",
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::U8 => 1,
            DType::U32 => 4,
            DType::BF16 => 2,
            DType::F16 => 2,
            DType::F32 => 4,
            DType::F64 => 8,
        }
    }
}

pub trait WithDType:
    Sized
    + Copy
    + num_traits::NumAssign
    + std::cmp::PartialOrd
    + std::fmt::Display
    + 'static
    + Send
    + Sync
    + crate::cpu::kernels::VecOps
{
    const DTYPE:DType;

    fn from_f64(v:f64) -> Self;
    fn to_f64(self) -> f64;

    fn to_cpu_storage_owmed(data: Vec<Self>) -> CpuStorage;

    fn to_cpu_storage(data: &[Self]) -> CpuStorage {
        Self::to_cpu_storage_owned(data.to_vec())
    }

    fn cpu_storage_as_slice(s: &CpuStorage) -> Result<&[Self]>;
    fn cpu_storage_data(s: CpuStorage) -> Result<Vec<Self>>;
}

/**
item ——一个项（item），像一个函数，结构体，模块等。
block ——一个块 （block）（即一个语句块或一个表达式，由花括号所包围）
stmt —— 一个语句（statement）
pat ——一个模式（pattern）
expr —— 一个表达式（expression）
ty ——一个类型（type）
ident—— 一个标识符（indentfier）
path —— 一个路径（path）（例如，foo，::std::mem::replace，transmute::<_, int>，...）
meta —— 一个元数据项；位于#[...]和#![...]属性
tt——一个词法树
vis——一个可能为空的Visibility限定词
*/
/**
这段代码是一个宏定义的示例，它定义了一个名为with_dtype的宏。
这个宏有四个参数：$ty表示数据类型，$dtype表示更具体的数据类型标识，
$from_f64表示从f64到指定类型的转换方法，$to_f64表示从指定类型到f64的转换方法。

通过使用这个宏，
可以方便地定义不同数据类型之间的转换方法。
它可以用于在代码中创建更通用的数据操作，
同时允许不同的数据类型使用不同的具体转换方法。
*/
macro_rules! with_dtype {
    ($ty:ty, $dtype:ident, $from_f64:expr, $to_f64:expr) => {
        impl WithDType for $ty {
            const DTYPE: DType = DType::$dtype;

              fn from_f64(v: f64) -> Self {
                $from_f64(v)
            }

            fn to_f64(self) -> f64 {
                $to_f64(self)
            }

             fn to_cpu_storage_owned(data: Vec<Self>) -> CpuStorage {
                CpuStorage::$dtype(data)
            }

            fn cpu_storage_data(s: CpuStorage) -> Result<Vec<Self>> {
                match s {
                    CpuStorage::$dtype(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }

            fn cpu_storage_as_slice(s: &CpuStorage) -> Result<&[Self]> {
                match s {
                    CpuStorage::$dtype(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }
        }
    };
}



with_dtype!(u8, U8, |v: f64| v as u8, |v: u8| v as f64);
with_dtype!(u32, U32, |v: f64| v as u32, |v: u32| v as f64);
with_dtype!(f16, F16, f16::from_f64, f16::to_f64);
with_dtype!(bf16, BF16, bf16::from_f64, bf16::to_f64);
with_dtype!(f32, F32, |v: f64| v as f32, |v: f32| v as f64);
with_dtype!(f64, F64, |v: f64| v, |v: f64| v);

pub trait IntDType: WithDType {
    fn is_true(&self) -> bool;
    
    fn as_usize(&self) -> usize;
}

impl IntDType for u32 {
    fn is_true(&self) -> bool {
        *self != 0
    }

    fn as_usize(&self) -> usize {
        *self as usize
    }
}

pub trait FloatDType: WithDType {}

impl FloatDType for f16 {}
impl FloatDType for bf16 {}
impl FloatDType for f32 {}
impl FloatDType for f64 {}
