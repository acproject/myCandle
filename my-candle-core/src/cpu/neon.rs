/// 这段代码是一个实现了Trait Cpu的结构体CurrentCpu的定义。Cpu是一个泛型trait，涉及到向量化计算的操作。根据不同的目标架构，分别引入了`core::arch::arm::*`和`core::arch::aarch64::*`模块。
///
/// 这段代码定义了一系列的常量和方法。其中，`STEP`表示向量的步长，`EPR`表示每个寄存器可以容纳的元素个数，`ARR`表示向量长度除以EPR的值。这里用常量的形式定义这些值，是为了在后面的代码中使用。
///
/// `CurrentCpu`提供了一些静态方法，包括`reduce_one`，用于对向量进行求和操作。根据不同目标架构的不同，`reduce_one`的具体实现有所区别，分别是`vaddvq_f32`和`vgetq_lane_f32`。
///
/// 接下来是对Cpu trait的实现，定义了与向量化计算相关的方法。其中，包括对单个向量和多个向量进行运算的方法，以及加载和存储数据的方法。通过使用`unsafe`关键字，可以直接调用原生的计算指令，提高运算速度。最后，定义了一个`vec_reduce`方法，用于将多个向量进行归约操作。
///
/// 这段代码的目的是抽象出一个通用的向量化计算trait，并根据不同的目标架构提供不同的实现。这样可以方便地在不同的架构上进行向量化计算。

use super::Cpu;
#[cfg(target_arch = "arm")]
use core::arch::arm::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

pub struct CurrentCpu {}

const STEP: usize = 16;
const EPR: usize = 4;
const ARR: usize = STEP / EPR;

impl CurrentCpu {
    #[cfg(target_arch = "aarch64")]
    unsafe fn reduce_one(x: float32x4_t) -> f32 {
        vaddvq_f32(x)
    }

    #[cfg(target_arch = "arm")]
    unsafe fn reduce_one(x: float32x4_t) -> f32 {
        vgetq_lane_f32(x, 0) + vgetq_lane_f32(x, 1) + vgetq_lane_f32(x, 2) + vgetq_lane_f32(x, 3)
    }
}

impl Cpu<ARR> for CurrentCpu {
    type Unit = float32x4_t;

    type Array = [float32x4_t; ARR];
}