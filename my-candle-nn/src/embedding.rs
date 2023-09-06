use my_candle_core::{Result, Tensor};

#[derive(Debug)]
pub struct Embedding {
    embeddings: Tensor,
    hidden_size: usize,
}

impl Embedding {
    pub fn new(embedding:Tensor, hidden_size: usize) -> Self {
        Self {
            embeddings,
            hidden_size,
        }
    }

    pub fn embeddings(&self) -> &Tensor { &self.embeddings }

    pub fn forward(&self, indexes:&Tensor) -> Result<Tensor> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = self.embeddings.index_select(&indexes, 0)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}

pub fn embedding(in_size: usize, out_size: usize, vb:crate::var_builder::VarBuilder) -> Result<Embedding> {
    let embedding = vb.get_or_init(
        (in_size, out_size),
        "weight",
        crate::init::Init::Randn {
            mean:0.,
            stdev: 1.,
        },
    )?;
    Ok(Embedding::new(embedding, out_size))
}