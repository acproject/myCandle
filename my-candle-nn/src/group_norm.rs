use my_candle_core::{DType, Result, Tensor};

#[derive(Debug)]
pub struct GroupNorm {
    weight:Tensor,
    bias:Tensor,
    eps: f64,
    num_channels:usize,
    num_groups:usize,
}

impl GroupNorm {
    pub fn new(weight:Tensor,
               bias:Tensor,
               num_channels:usize,
               num_groups:usize, eps: f64,) -> Result<Self> {
        if num_channels % num_groups != 0 {
            my_candle_core::bail!(
                "GroupNorm: num_groups ({num_groups}) must divide num_channels ({num_channels})"
            )
        }
        Ok(Self {
            weight,
            bias,
            eps,
            num_channels,
            num_groups,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_shape = x.dims();
        if x_shape.len() <= 2 {
            my_candle_core::bail!("input rank for GroupNorm should be at least 3");
        }
        let (b_sz, n_channels) = (x_shape[0], x_shape[1]);
        let hidden_size = x_shape[2..].iter().product::<usize>() * n_channels / self.num_groups;
        if n_channels != self.num_channels {
            my_candle_core::bail!(
                "unexpected num-channels in GroupNorm ({n_channels} <> {}",
                self.num_channels
            )
        }
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };

        let x = x.reshape((b_sz, self.num_groups, hidden_size))?;
        let x = x.to_dtype(internal_dtype)?;
        let mean_x = (x.sum_keepdim(2)? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        let norm_x = (x.sqr()?.sum_keepdim(2)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let mut w_dims = vec![1; x_shape.len()];
        w_dims[1] = n_channels;
        let weight = self.weight.reshape(w_dims.clone())?;
        let bias = self.bias.reshape(w_dims)?;
        x_normed
            .to_dtype(x_dtype)?
            .reshape(x_shape)?
            .broadcast_mul(&weight)?
            .broadcast_add(&bias)
    }
}

pub fn group_nore(
    num_groups:usize,
    num_channels: usize,
    eps: f64,
    vb:crate::var_builder::VarBuilder,
) -> Result<GroupNorm> {
    let weight = vb.get_or_init(num_channels, "weight", crate::init::Init::Const(1.))?;
    let bias = vb.get_or_init(num_channels, "bias", crate::init::Init::Const(0.))?;
    GroupNorm::new(weight, bias, num_channels, num_groups, eps)
}