use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use safetensors::SafeTensors;
use my_candle_core::{Device, DType, Shape, Tensor, Var};
use crate::init::Init;
use my_candle_core::Result;

#[derive(Clone)]
pub struct VarMap {
    data:Arc<Mutex<HashMap<String, Var>>>,
}

#[derive(Clone)]
pub struct VarBuilder<'a> {
    data: Arc<TensorData<'a>>,
    path:Vec<String>,
}

struct TensorData<'a> {
    tensors: Tensors<'a>,
    pub dtype:DType,
    pub device:Device,
}

enum Tensors<'a> {
    SafeTensorWithRouting {
        routing:HashMap<String, usize>,
        safetensors:Vec<SafeTensors<'a>>,
    },
    Npz(my_candle_core::npy::NpzTensors),
    TensorMap(HashMap<String, Tensor>),
    Zeros,
    VarMap(VarMap),
}

impl<'a> VarBuilder<'a> {
    pub fn get_or_init<S: Into<Shape>>(
        &self,
        s: S,
        tensor_name: &str,
        init: Init,
    ) -> Result<Tensor> {
        let data = self.data.as_ref();
        match &self.data.tensors {
            Tensors::VarMap(varmap) => {
                let path = self.path(tensor_name);
                varmap.get(s, &path, init, data.dtype, &data.device)
            }
            _ => self.get(s, tensor_name),
        }
    }
}