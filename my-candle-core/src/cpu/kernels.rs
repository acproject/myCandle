pub trait VecOps: num_traits::NumAssign + Copy {
    /// Dot-product of two vectors.
    ///
    /// # Safety
    ///
    /// the length of 'lhs' and 'rhs' have to be at least 'len', 'res' has to point to a valid
    /// element.

    #[inline(always)]
    unsafe fn vec_dot(lhs: *const Self, rhs: *const Self, res: *mut Self, len: usize) {
        *res = Self::zero();
        for i in 0..len {
            *res += *lhs.add(i) * *rhs.add(i)
        }
    }

    /// Sum of all elements in a vector.
    ///
    /// # Safety
    /// The length of `xs` must be at least `len`. `res` has to point to a valid
    /// element.
    #[inline(always)]
    unsafe fn vec_reduce_sum(xs: *const Self, res: *mut Self, len:usize) {
        *res = Self::zero();
        for i in 0..len {
            *res += *xs.add(i)
        }
    }
}

impl VecOps for f32 {

    #[inline(always)]
    unsafe  fn vec_dot(lhs: *const Self, rhs: *const Self, res: *mut Self, len: usize) {
        super::vec_dot_f32(lhs, rhs, res, len)
    }

    #[inline(always)]
    unsafe fn vec_reduce_sum(xs: *const Self, res: *mut Self, len: usize) {
        super::vec_sum(xs, res, len)
    }
}

impl VecOps for half::f16 {
    #[inline(always)]
    unsafe fn vec_dot(lhs: *const Self, rhs: *const Self,  res: *mut Self, len: usize) {
        let mut res_f32 = 0f32;
        super::vec_dot_f16(lhs, rhs, &mut res_f32, len);
        *res = half::f16::from_f32(res_f32);
    }
}

impl VecOps for f64 {}

impl VecOps for half::bf16 {}

impl VecOps for u8 {}

impl VecOps for u32 {}


/**
这段代码定义了一个函数 par_for_each，用于并行地执行一个函数。具体来说：

#[inline(always)] 是一个属性注解，用于提示编译器优化内联函数调用。
pub fn par_for_each(n_threads: usize, func: impl Fn(usize) + Send + Sync)
是函数的签名，它接受两个参数：n_threads 表示要启动的线程数，func 则是一个函数闭包，接受一个 usize 参数并返回 ()。
函数首先检查 n_threads 是否为 1，如果是，则直接调用 func，传入参数 0。

如果 n_threads 不是 1，则使用 rayon::scope 创建一个新的作用域。rayon 是一个并行计算库，
它提供了一种在作用域内创建并发任务的方式。在这个作用域内，使用 for 循环创建 n_threads 个任务。

每个任务中的闭包会获取 func 的不可变借用，并使用 move 关键字获取 thread_idx 的所有权。
然后，通过 spawn 方法创建一个线程，传入闭包作为参数。闭包中会调用 func，传入对应的 thread_idx 作为参数。

这样就可以并行地执行 func 函数，并使用不同的线程（任务）调用函数的不同参数。
*/
#[inline(always)]
pub fn par_for_each(n_threads: usize, func: impl Fn(usize) + Send + Sync) {
    if n_threads == 1 {
        func(0)
    } else {
        rayon::scope(|s| {
            for thread_idx in 0..n_threads {
                let func = &func;
                s.spawn(move |_| func(thread_idx));
            }
        })
    }
}

#[inline(always)]
pub fn par_range(lo: usize, up: usize, n_threads: usize, func: impl Fn(usize) + Send + Sync) {
    if n_threads == 1 {
        for i in lo..up {
            func(i)
        }
    } else {
        rayon::scope(|s| {
            for thread_idx in 0..n_threads {
                let func = &func;
                s.spawn(move |_| {
                    for i in (thread_idx..up).step_by(n_threads) {
                        func(i)
                    }
                });
            }
        })
    }
}
