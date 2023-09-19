//! Linear layer
//!
//! This layer applies a linear transformation to the incoming data, `y = x@w.t() + b`.
//! The bias is optional. The `forward` method can be used to apply the layer, it supports input
//! with a batch dimension (so of shape `(b_sz, in_c)`) or without (of shape `(in_c,)`), the
//! output has shape `(b_sz, out_c)` and `(out_c,)` respectively.
//!
//! ```rust
//! use my_candle_core::{Tensor, Device::Cpu};
//! use my_candle_nn::Linear;
//! # fn main() -> my_candle_core::Result<()> {
//!
//! let w = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Cpu)?;
//! let layer = Linear::new(w, None); // Use no bias.
//! let xs = Tensor::new(&[[10f32, 100.]], &Cpu)?;
//! let ys = layer.forward(&xs)?;
//! assert_eq!(ys.to_vec2::<f32>()?, &[[210.0, 430.0, 650.0]]);
//! # Ok(()) }
//! ```

use my_candle_core::{Result, Tensor};

#[derive(Debug)]
pub struct Linear {
    weight:Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight:Tensor, bias:Option<Tensor>) -> Self {
        Self {
            weight,
            bias,
        }
    }

    pub fn forward(&self, x:&Tensor) -> Result<Tensor> {
        let w = match x.dims() {
            &[bszie, _, _] => self.weight.broadcast_left(bszie)?.t()?,
            _ => self.weight.t()?,
        };

        let x = x.matmul(&w)?;
        match *self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

/// Create or initialize a new linear layer.
///
/// This uses some default names for weight and biases, namely `"weight"` and `"bias"`.
pub fn linear(in_dim: usize, out_dim: usize, vs: crate::VarBuilder) -> Result<Linear> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vs.get_or_init((out_dim, in_dim), "weight", init_ws)?;
    let bound = 1. / (in_dim as f64).sqrt();
    let init_bs = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vs.get_or_init(out_dim,"bias", init_bs)?;
    Ok(Linear::new(ws, Some(bs)))
}

pub fn linear_no_bias(in_dim:usize, out_dim:usize, vs:crate::VarBuilder) -> Result<Linear> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vs.get_or_init((out_dim, in_dim), "weight", init_ws)?;
    Ok(Linear::new(ws,None))
}