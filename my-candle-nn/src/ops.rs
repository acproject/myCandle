use my_candle_core::{Tensor, Result};
/// Applies the softmax function to the input tensor, rescaling the element so that elements on
/// a slice of fixed index on dimension `dim` are between 0 and 1 and sum to 1.
///
/// ```rust
/// use my_candle_core::{Tensor, Device};
/// let  a = Tensor::new(&[[0f32, 1., 0., 1.], [-2. ,2.,3., -3.]], &Device::Cpu)?;
///
pub fn log_softmax<D: my_candle_core::shape::Dim>(xs: &Tensor, d: D) -> Result<Tensor> {
    let d = d.to_index(xs.shape(), "log-softmax")?;
    let max = xs.max_keepdim(d)?;
    let diff = xs.broadcast_sub(&max)?;
    let sum_exp = diff.exp()?.sum_keepdim(d)?;
    let log_sm = diff.broadcast_sub(&sum_exp.log()?)?;
    Ok(log_sm)
}

pub fn softmax<D: my_candle_core::shape::Dim>(xs:&Tensor, dim: D) -> Result<Tensor> {
    let dim = dim.to_index(xs.shape(), "softmax")?;
    let max = xs.max_keepdim(dim)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(dim)?;
    num.broadcast_div(&den)
}

pub fn silu(xs:Tensor)-> Result<Tensor> {
    xs / (xs.neg()?.exp()? + 1.0)?
}

pub fn sigmoid(xs:&Tensor) -> Result<Tensor> {
    (xs.neg()?.exp()? + 1.0)?.recip()
}