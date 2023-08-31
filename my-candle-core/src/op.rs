//!
//! 在这里有DSL中已经用到的具体操作，也是通用的操作
//!
use half::{bf16, f16};
use num_traits::float::Float;
use crate::layout::Layout;
use crate::shape::Shape;
use crate::error::{Error, Result};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Min,
    Max,
    ArgMin,
    ArgMax,
}

impl ReduceOp {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::ArgMax => "argmax",
            Self::ArgMin => "argmin",
            Self::Min => "min",
            Self::Max => "max",
            Self::Sum => "sum",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Mul,
    Sub,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Exp,
    Log,
    Sin,
    Cos,
    Abs,
    Neg,
    Recip,
    Sqr,
    Sqrt,
    Gelu,
    Relu,
}

#[derive(Clone)]
pub enum Op {
    Binary(Tensor, Tensor, BinaryOp),
    Unary(Tensor, UnaryOp),
    Cmp(Tensor, CmpOp),
    // The third argument is the reduced shape with `keepdim=true`.
    Reduce(Tensor, ReduceOp, Vec<usize>),
    Matmul(Tensor, Tensor),
    Gather(Tensor, Tensor, usize),
    ScatterAdd(Tensor, Tensor, Tensor, usize),
    IndexSelect(Tensor, Tensor, usize),
    IndexAdd(Tensor, Tensor, Tensor, usize),
    WhereCond(Tensor, Tensor, Tensor),

    #[allow(dead_code)]
    Conv1D {
        arg: Tensor,
        kernel: Tensor,
        padding: usize,
        stride: usize,
    },

    #[allow(dead_code)]
    Conv2D {
        arg: Tensor,
        kernel: Tensor,
        padding: usize,
        stride: usize,
    },

    AvgPool2D {
        arg: Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    },

    MaxPool2D {
        arg: Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    },

    UpsampleNearest2D(Tensor),

    Cat(Vec<Tensor>, usize),

    #[allow(dead_code)] // add is currently unused.
    Affine {
        arg: Tensor,
        mul: f64,
        add: f64,
    },
    ToDType(Tensor),
    Copy(Tensor),
    Broadcast(Tensor),
    Narrow(Tensor, usize, usize, usize),
    Reshape(Tensor),
    ToDevice(Tensor),
    Transpose(Tensor, usize, usize),
    Elu(Tensor, f64),
    CustomOp1(Tensor, std::sync::Arc<Box<dyn CustomOp1>>),
    CustomOp2(Tensor, Tensor, std::sync::Arc<Box<dyn CustomOp2>>),
    CustomOp3(Tensor, Tensor, Tensor, std::sync::Arc<Box<dyn CustomOp3>>),
}

pub trait CustomOp1: Send + Sync {
    // Box<dyn> does not support const yet, so use a function to get the name.
    fn name(&self) -> &'static str;

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)>;

    /// The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cuda_fwd(&self, _storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        Err(Error::Cuda(
            format!("no cuda implementation for {}", self.name()).into(),
        ))
    }

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    fn bwd(&self, _arg: &Tensor, _res: &Tensor, _grad_res: &Tensor) -> Result<Option<Tensor>> {
        Err(Error::BackwardNotSupported { op: self.name() })
    }
}

pub trait CustomOp2: Send + Sync {
    fn name(&self) -> &'static str;

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)>;

    /// The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cuda_fwd(
        &self,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        Err(Error::Cuda(
            format!("no cuda implementation for {}", self.name()).into(),
        ))
    }

    fn bwd(
        &self,
        _arg1: &Tensor,
        _arg2: &Tensor,
        _res: &Tensor,
        _grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        Err(Error::BackwardNotSupported { op: self.name() })
    }
}

pub trait CustomOp3: Send + Sync {
    fn name(&self) -> &'static str;

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)>;

    /// The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cuda_fwd(
        &self,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        Err(Error::Cuda(
            format!("no cuda implementation for {}", self.name()).into(),
        ))
    }

    fn bwd(
        &self,
        _arg1: &Tensor,
        _arg2: &Tensor,
        _arg3: &Tensor,
        _res: &Tensor,
        _grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        Err(Error::BackwardNotSupported { op: self.name() })
    }
}

pub trait UnaryOpT {
    const NAME: &'static str;
    const KERNEL: &'static str;
    const V: Self;
    fn bf16(v1: bf16) -> bf16;
    fn f16(v1: f16) -> f16;
    fn f32(v1: f32) -> f32;
    fn f64(v1: f64) -> f64;
    fn u8(v1: u8) -> u8;
    fn u32(v1: u32) -> u32;

    // There is no very good way to represent optional function in traits so we go for an explicit
    // boolean flag to mark the function as existing.
    const BF16_VEC: bool = false;
    fn bf16_vec(_xs: &[bf16], _ys: &mut [bf16]) {}
    const F16_VEC: bool = false;
    fn f16_vec(_xs: &[f16], _ys: &mut [f16]) {}
    const F32_VEC: bool = false;
    fn f32_vec(_xs: &[f32], _ys: &mut [f32]) {}
    const F64_VEC: bool = false;
    fn f64_vec(_xs: &[f64], _ys: &mut [f64]) {}
}

pub trait BinaryOpT {
    const NAME: &'static str;
    const KERNEL: &'static str;
    const V: Self;
    fn bf16(v1: bf16, v2: bf16) -> bf16;
    fn f16(v1: f16, v2: f16) -> f16;
    fn f32(v1: f32, v2: f32) -> f32;
    fn f64(v1: f64, v2: f64) -> f64;
    fn u8(v1: u8, v2: u8) -> u8;
    fn u32(v1: u32, v2: u32) -> u32;

    const BF16_VEC: bool = false;
    fn bf16_vec(_xs1: &[bf16], _xs2: &[bf16], _ys: &mut [bf16]) {}
    const F16_VEC: bool = false;
    fn f16_vec(_xs1: &[f16], _xs2: &[f16], _ys: &mut [f16]) {}
    const F32_VEC: bool = false;
    fn f32_vec(_xs1: &[f32], _xs2: &[f32], _ys: &mut [f32]) {}
    const F64_VEC: bool = false;
    fn f64_vec(_xs1: &[f64], _xs2: &[f64], _ys: &mut [f64]) {}
    const U8_VEC: bool = false;
    fn u8_vec(_xs1: &[u8], _xs2: &[u8], _ys: &mut [u8]) {}
    const U32_VEC: bool = false;
    fn u32_vec(_xs1: &[u32], _xs2: &[u32], _ys: &mut [u32]) {}
}

pub(crate) struct Add;
pub(crate) struct Div;
pub(crate) struct Mul;
pub(crate) struct Sub;
pub(crate) struct Exp;
pub(crate) struct Log;
pub(crate) struct Sin;
pub(crate) struct Cos;
pub(crate) struct Abs;
pub(crate) struct Neg;
pub(crate) struct Recip;
pub(crate) struct Sqr;
pub(crate) struct Sqrt;
pub(crate) struct Gelu;
pub(crate) struct Relu;

macro_rules! bin_op {
    ($op:ident, $name: literal, $e: expr, $f32_vec: ident, $f64_vec: ident) => {
        impl BinaryOpT for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("b", $name);
            const V: Self = $op;
            #[inline(always)]
            fn bf16(v1: bf16, v2: bf16) -> bf16 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f16(v1: f16, v2: f16) -> f16 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f32(v1: f32, v2: f32) -> f32 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f64(v1: f64, v2: f64) -> f64 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn u8(v1: u8, v2: u8) -> u8 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn u32(v1: u32, v2: u32) -> u32 {
                $e(v1, v2)
            }

            #[cfg(feature = "mkl")]
            const F32_VEC: bool = true;
            #[cfg(feature = "mkl")]
            const F64_VEC: bool = true;
            #[cfg(feature = "mkl")]
            #[inline(always)]
            fn f32_vec(xs1: &[f32], xs2: &[f32], ys: &mut [f32]) {
                crate::mkl::$f32_vec(xs1, xs2, ys)
            }
            #[cfg(feature = "mkl")]
            #[inline(always)]
            fn f64_vec(xs1: &[f64], xs2: &[f64], ys: &mut [f64]) {
                crate::mkl::$f64_vec(xs1, xs2, ys)
            }

            #[cfg(feature = "accelerate")]
            const F32_VEC: bool = true;
            #[cfg(feature = "accelerate")]
            const F64_VEC: bool = true;
            #[cfg(feature = "accelerate")]
            #[inline(always)]
            fn f32_vec(xs1: &[f32], xs2: &[f32], ys: &mut [f32]) {
                crate::accelerate::$f32_vec(xs1, xs2, ys)
            }
            #[cfg(feature = "accelerate")]
            #[inline(always)]
            fn f64_vec(xs1: &[f64], xs2: &[f64], ys: &mut [f64]) {
                crate::accelerate::$f64_vec(xs1, xs2, ys)
            }
        }
    };
}

bin_op!(Add, "add", |v1, v2| v1 + v2, vs_add, vd_add);
bin_op!(Sub, "sub", |v1, v2| v1 - v2, vs_sub, vd_sub);
bin_op!(Mul, "mul", |v1, v2| v1 * v2, vs_mul, vd_mul);
bin_op!(Div, "div", |v1, v2| v1 / v2, vs_div, vd_div);

macro_rules! unary_op {
    ($op: ident, $name: literal, $a: ident, $e: expr) => {
        impl UnaryOpT for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("u", $name);
            const V: Self = $op;
            #[inline(always)]
            fn bf16($a: bf16) -> bf16 {
                $e
            }
            #[inline(always)]
            fn f16($a: f16) -> f16 {
                $e
            }
            #[inline(always)]
            fn f32($a: f32) -> f32 {
                $e
            }
            #[inline(always)]
            fn f64($a: f64) -> f64 {
                $e
            }
            #[inline(always)]
            fn u8(_: u8) -> u8 {
                todo!("no unary function for u8")
            }
            #[inline(always)]
            fn u32(_: u32) -> u32 {
                todo!("no unary function for u32")
            }
        }
    };

    ($op: ident, $name: literal, $a: ident, $e: expr, $f32_vec:ident, $f64_vec:ident) => {
        impl UnaryOpT for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("u", $name);
            const V: Self = $op;
            #[inline(always)]
            fn bf16($a: bf16) -> bf16 {
                $e
            }
            #[inline(always)]
            fn f16($a: f16) -> f16 {
                $e
            }
            #[inline(always)]
            fn f32($a: f32) -> f32 {
                $e
            }
            #[inline(always)]
            fn f64($a: f64) -> f64 {
                $e
            }
            #[inline(always)]
            fn u8(_: u8) -> u8 {
                todo!("no unary function for u8")
            }
            #[inline(always)]
            fn u32(_: u32) -> u32 {
                todo!("no unary function for u32")
            }

            #[cfg(feature = "mkl")]
            const F32_VEC: bool = true;
            #[cfg(feature = "mkl")]
            const F64_VEC: bool = true;
            #[cfg(feature = "mkl")]
            #[inline(always)]
            fn f32_vec(xs: &[f32], ys: &mut [f32]) {
                crate::mkl::$f32_vec(xs, ys)
            }
            #[cfg(feature = "mkl")]
            #[inline(always)]
            fn f64_vec(xs: &[f64], ys: &mut [f64]) {
                crate::mkl::$f64_vec(xs, ys)
            }

            #[cfg(feature = "accelerate")]
            const F32_VEC: bool = true;
            #[cfg(feature = "accelerate")]
            const F64_VEC: bool = true;
            #[cfg(feature = "accelerate")]
            #[inline(always)]
            fn f32_vec(xs: &[f32], ys: &mut [f32]) {
                crate::accelerate::$f32_vec(xs, ys)
            }
            #[cfg(feature = "accelerate")]
            #[inline(always)]
            fn f64_vec(xs: &[f64], ys: &mut [f64]) {
                crate::accelerate::$f64_vec(xs, ys)
            }
        }
    };
}

unary_op!(Exp, "exp", v, v.exp(), vs_exp, vd_exp);
unary_op!(Log, "log", v, v.ln(), vs_ln, vd_ln);
unary_op!(Sin, "sin", v, v.sin(), vs_sin, vd_sin);
unary_op!(Cos, "cos", v, v.cos(), vs_cos, vd_cos);
unary_op!(Abs, "abs", v, v.abs());
unary_op!(Neg, "neg", v, -v);
unary_op!(Recip, "recip", v, v.recip());
unary_op!(Sqr, "sqr", v, v * v, vs_sqr, vd_sqr);
unary_op!(Sqrt, "sqrt", v, v.sqrt(), vs_sqrt, vd_sqrt);

/// `gelu` operation
/// <https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions>
impl UnaryOpT for Gelu {
    const NAME: &'static str = "gelu";
    const V: Self = Gelu;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        bf16::from_f32_const(0.5)
            * v
            * (bf16::ONE
            + bf16::tanh(
            (bf16::from_f32_const(2.0) / bf16::PI).sqrt()
                * v
                * (bf16::ONE + bf16::from_f32_const(0.044715) * v * v),
        ))
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        f16::from_f32_const(0.5)
            * v
            * (f16::ONE
            + f16::tanh(
            (f16::from_f32_const(2.0) / f16::PI).sqrt()
                * v
                * (f16::ONE + f16::from_f32_const(0.044715) * v * v),
        ))
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        0.5 * v
            * (1.0
            + f32::tanh((2.0f32 / std::f32::consts::PI).sqrt() * v * (1.0 + 0.044715 * v * v)))
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        0.5 * v
            * (1.0
            + f64::tanh((2.0f64 / std::f64::consts::PI).sqrt() * v * (1.0 + 0.044715 * v * v)))
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        0
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        0
    }
    const KERNEL: &'static str = "ugelu";

    #[cfg(feature = "mkl")]
    const F32_VEC: bool = true;

    #[cfg(feature = "mkl")]
    #[inline(always)]
    fn f32_vec(xs: &[f32], ys: &mut [f32]) {
        crate::mkl::vs_gelu(xs, ys)
    }

    #[cfg(feature = "mkl")]
    const F64_VEC: bool = true;

    #[cfg(feature = "mkl")]
    #[inline(always)]
    fn f64_vec(xs: &[f64], ys: &mut [f64]) {
        crate::mkl::vd_gelu(xs, ys)
    }
}

impl UnaryOpT for Relu {
    const NAME: &'static str = "relu";
    const KERNEL: &'static str = "urelu";
    const V: Self = Relu;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v.max(bf16::ZERO)
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v.max(f16::ZERO)
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v.max(0f32)
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v.max(0f64)
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        v
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        v
    }
}

/// `BackpropOp` is a wrapper around `Option<Op>`. The main goal is to ensure that dependencies are
/// properly checked when creating a new value
#[derive(Clone)]
pub struct BackpropOp(Option<Op>);

impl BackpropOp {
    pub(crate) fn none() -> Self {
        BackpropOp(None)
    }

    pub(crate) fn new1(arg: &Tensor, f: impl Fn(Tensor) -> Op) -> Self {
        let op = if arg.track_op() {
            Some(f(arg.clone()))
        } else {
            None
        };
        Self(op)
    }

    pub(crate) fn new2(arg1: &Tensor, arg2: &Tensor, f: impl Fn(Tensor, Tensor) -> Op) -> Self {
        let op = if arg1.track_op() || arg2.track_op() {
            Some(f(arg1.clone(), arg2.clone()))
        } else {
            None
        };
        Self(op)
    }

    pub(crate) fn new3(
        arg1: &Tensor,
        arg2: &Tensor,
        arg3: &Tensor,
        f: impl Fn(Tensor, Tensor, Tensor) -> Op,
    ) -> Self {
        let op = if arg1.track_op() || arg2.track_op() || arg3.track_op() {
            Some(f(arg1.clone(), arg2.clone(), arg3.clone()))
        } else {
            None
        };
        Self(op)
    }

    pub(crate) fn new<A: AsRef<Tensor>>(args: &[A], f: impl Fn(Vec<Tensor>) -> Op) -> Self {
        let op = if args.iter().any(|arg| arg.as_ref().track_op()) {
            let args: Vec<Tensor> = args.iter().map(|arg| arg.as_ref().clone()).collect();
            Some(f(args))
        } else {
            None
        };
        Self(op)
    }
}

impl std::ops::Deref for BackpropOp {
    type Target = Option<Op>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
