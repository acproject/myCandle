use my_candle_core::Tensor;

pub enum Activation {
    Gelu,
    Relu,
    Elu(f64),
}

impl Activation {
    pub fn forward(&self, xs: &Tensor) -> my_candle_core::Result<Tensor> {
        match self {
            Activation::Gelu => xs.gelu(),
            Activation::Relu => xs.relu(),
            &Activation::Elu(alpha) => xs.elu(alpha),
        }
    }
}