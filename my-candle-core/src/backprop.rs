use crate::op::{BinaryOp, Op, ReduceOp, UnaryOp};
use crate::{Error, Result, Tensor, TensorId};
use std::collections::HashMap;
use num_traits::real::Real;

// arg has been reduced to node via reduce_dims, expand it back to arg.
// This has to handle keepdims.
fn broadcast_back(arg: &Tensor, node: &Tensor, reduced_dims: &[usize]) -> Result<Tensor> {
    if arg.rank() == node.rank() {
        // keepdim = true
        node.broadcast_as(arg.shape())
    } else {
        // keepdim = false
        // first expand the reduced dims.
        node.reshape(reduced_dims)?.broadcast_as(arg.shape())
    }
}

impl Tensor {
    /// Return all the nodes that lead to this value in a topologically sorted vec, the first
    /// elements having dependencies on the latter ones, e.g. the first element if any is the
    /// argument.
    /// This assumes that the op graph is a DAG.
    fn sorted_nodes(&self) -> Vec<&Tensor> {
        // The vec of sorted nodes is passed as an owned value rather than a mutable reference
        // to get around some lifetime limitations.
        fn walk<'a>(
            node: &'a Tensor,
            nodes: Vec<&'a Tensor>,
            already_seen: &mut HashMap<TensorId, bool>,
        ) -> (bool, Vec<&'a Tensor>) {
            if let Some(&tg) = already_seen.get(&node.id()) {
                return (tg, nodes);
            }
            let mut track_grad = false;
            let mut nodes = if node.is_variable() {
                // Do not call recursively on the "leaf" nodes.
                track_grad = true;
                nodes
            } else if let Some(op) = node.op() {
                match op {
                    Op::IndexAdd(t1, t2, t3, _)
                    | Op::ScatterAdd(t1, t2, t3, _)
                    | Op::CustomOp3(t1, t2, t3, _)
                    | Op::WhereCond(t1, t2, t3) => {
                        let (tg, nodes) = walk(t1, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(t2, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(t3, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    Op::Conv1D {
                        arg: lhs,
                        kernel: rhs,
                        ..
                    }
                    | Op::Conv2D {
                        arg: lhs,
                        kernel: rhs,
                        ..
                    }
                    | Op::CustomOp2(lhs, rhs, _)
                    | Op::Binary(lhs, rhs, _)
                    | Op::Gather(lhs, rhs, _)
                    | Op::IndexSelect(lhs, rhs, _)
                    | Op::Matmul(lhs, rhs) => {
                        let (tg, nodes) = walk(lhs, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(rhs, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    Op::Cat(args, _) => args.iter().fold(nodes, |nodes, arg| {
                        let (tg, nodes) = walk(arg, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }),
                    Op::Affine { arg, mul, .. } => {
                        if *mul == 0. {
                            nodes
                        } else {
                            let (tg, nodes) = walk(arg, nodes, already_seen);
                            track_grad |= tg;
                            nodes
                        }
                    }
                    Op::Reshape(node)
                    | Op::UpsampleNearest2D(node)
                    | Op::AvgPool2D { arg: node, .. }
                    | Op::MaxPool2D { arg: node, .. }
                    | Op::Copy(node)
                    | Op::Broadcast(node)
                    | Op::Cmp(node, _)
                    | Op::Reduce(node, _, _)
                    | Op::ToDType(node)
                    | Op::ToDevice(node)
                    | Op::Transpose(node, _, _)
                    | Op::Narrow(node, _, _, _)
                    | Op::Unary(node, _)
                    | Op::Elu(node, _)
                    | Op::CustomOp1(node, _) => {
                        let (tg, nodes) = walk(node, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                }
            } else {
                nodes
            };
            already_seen.insert(node.id(), track_grad);
            if track_grad {
                nodes.push(node);
            }
            (track_grad, nodes)
        }
        let (_tg, mut nodes) = walk(self, vec![], &mut HashMap::new());
        nodes.reverse();
        nodes
    }

    pub fn backward(&self) -> Result<GradStore> {
        let sorted_nodes = self.sorted_nodes();
        let mut grads = GradStore::new();
        grads.insert(self, self.ones_like()?.contiguous()?);
        for node in sorted_nodes.iter() {
            if node.is_variable() {
                continue;
            }
            let grad = grads.remove(node).unwrap();
            // TODO: We should perform all these operations in place (or at least not track the
            // whole graph). The only drawback would be if we wanted to support grad of grad but
            // this is out of scope.
            if let Some(op) = node.op() {
                match op {
                    Op::Binary(lhs, rhs, BinaryOp::Add) => {
                        let lhs_sum_grad = grads.or_insert(lhs)?;
                        *lhs_sum_grad = lhs_sum_grad.add(&grad)?;
                        let rhs_sum_grad = grads.or_insert(rhs)?;
                        *rhs_sum_grad = rhs_sum_grad.add(&grad)?;
                    }
                    Op::Binary(lhs, rhs, BinaryOp::Sub) => {
                        let lhs_sum_grad = grads.or_insert(lhs)?;
                        *lhs_sum_grad = lhs_sum_grad.add(&grad)?;
                        let rhs_sum_grad = grads.or_insert(rhs)?;
                        *rhs_sum_grad = rhs_sum_grad.sub(&grad)?;
                    }
                    Op::Binary(lhs, rhs, BinaryOp::Mul) => {
                        let lhs_grad = grad.mul(rhs)?;
                        let lhs_sum_grad = grads.or_insert(lhs)?;
                        *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;
                        let rhs_grad = grad.mul(lhs)?;
                        let rhs_sum_grad = grads.or_insert(rhs)?;
                        *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                    }
                    Op::Binary(lhs, rhs, BinaryOp::Div) => {
                        let lhs_grad = grad.div(rhs)?;
                        let lhs_sum_grad = grads.or_insert(lhs)?;
                        *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;
                        let rhs_grad = grad.mul(lhs)?.div(&rhs.sqr()?)?;
                        let rhs_sum_grad = grads.or_insert(rhs)?;
                        *rhs_sum_grad = rhs_sum_grad.sub(&rhs_grad)?;
                    }
                    Op::WhereCond(pred, t, f) => {
                        let zeros = grad.zeros_like()?;
                        let t_sum_grad = grads.or_insert(t)?;
                        let t_grad = pred.where_cond(&grad, &zeros)?;
                        *t_sum_grad = t_sum_grad.add(&t_grad)?;
                        let f_sum_grad = grads.or_insert(f)?;
                        let f_grad = pred.where_cond(&zeros, &grad)?;
                        *f_sum_grad = f_sum_grad.add(&f_grad)?;
                    }
                    Op::Conv1D { .. } => Err(Error::BackwardNotSupported { op: "conv1d" })?,
                    Op::Conv2D { .. } => Err(Error::BackwardNotSupported { op: "conv2d" })?,
                    Op::AvgPool2D { .. } => Err(Error::BackwardNotSupported { op: "avg-pool2d" })?,
                    Op::MaxPool2D { .. } => Err(Error::BackwardNotSupported { op: "max-pool2d" })?,
                    Op::UpsampleNearest2D { .. } => Err(Error::BackwardNotSupported {
                        op: "upsample-nearest2d",
                    })?,
                    Op::Gather(arg, indexes, dim) => {
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.scatter_add(indexes, &grad, *dim)?;
                    }
                    Op::ScatterAdd(init, indexes, src, dim) => {
                        let init_sum_grad = grads.or_insert(init)?;
                        *init_sum_grad = init_sum_grad.add(&grad)?;

                        let src_grad = grad.gather(indexes, *dim)?;
                        let src_sum_grad = grads.or_insert(src)?;
                        *src_sum_grad = src_sum_grad.add(&src_grad)?;
                    }
                    Op::IndexAdd(init, indexes, src, dim) => {
                        let init_sum_grad = grads.or_insert(init)?;
                        *init_sum_grad = init_sum_grad.add(&grad)?;

                        let src_grad = grad.index_select(indexes, *dim)?;
                        let src_sum_grad = grads.or_insert(src)?;
                        *src_sum_grad = src_sum_grad.add(&src_grad)?;
                    }
                    Op::IndexSelect(arg, indexes, dim) => {
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.index_add(indexes, &grad, *dim)?;
                    }
                    Op::Matmul(lhs, rhs) => {
                        // Skipping checks, the op went ok, we can skip
                        // the matmul size checks for now.

                        let lhs_grad = grad.matmul(&rhs.t()?)?;
                        let lhs_sum_grad = grads.or_insert(lhs)?;
                        *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;

                        let rhs_grad = lhs.t()?.matmul(&grad)?;
                        let rhs_sum_grad = grads.or_insert(rhs)?;
                        *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                    }
                    Op::Cat(args, dim) => {
                        let mut start_idx = 0;
                        for arg in args {
                            let len = arg.dims()[*dim];
                            let arg_grad = grad.narrow(*dim, start_idx, len)?;
                            let sum_grad = grads.or_insert(arg)?;
                            *sum_grad = sum_grad.add(&arg_grad)?;
                            start_idx += len;
                        }
                    }
                    Op::Broadcast(arg) => {
                        let arg_dims = arg.dims();
                        let node_dims = node.dims();
                        // The number of dims that have been inserted on the left.
                        let left_dims = node_dims.len() - arg_dims.len();
                        let mut sum_dims: Vec<usize> = (0..left_dims).collect();
                        for (dim, (node_dim, arg_dim)) in node_dims[left_dims..]
                            .iter()
                            .zip(arg_dims.iter())
                            .enumerate()
                        {
                            if node_dim != arg_dim {
                                sum_dims.push(dim + left_dims)
                            }
                        }

                        let mut arg_grad = grad.sum_keepdim(sum_dims.as_slice())?;
                        for _i in 0..left_dims {
                            arg_grad = arg_grad.squeeze(0)?
                        }
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad.broadcast_as(sum_grad.dims())?)?;
                    }
                    Op::Reduce(arg, ReduceOp::Sum, reduced_dims) => {
                        let grad = broadcast_back(arg, &grad, reduced_dims)?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&grad)?;
                    }
                    Op::Cmp(_args, _) => {}
                    Op::Reduce(arg, ReduceOp::Max, reduced_dims) => {
                        let node = broadcast_back(arg, node, reduced_dims)?;
                        let grad = broadcast_back(arg, &grad, reduced_dims)?;
                        let grad = node.eq(arg)?.to_dtype(grad.dtype())?.mul(&grad)?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&grad.broadcast_as(sum_grad.dims())?)?;
                    }
                    Op::Reduce(arg, ReduceOp::Min, reduced_dims) => {
                        let node = broadcast_back(arg, node, reduced_dims)?;
                        let grad = broadcast_back(arg, &grad, reduced_dims)?;
                        let grad = node.eq(arg)?.to_dtype(grad.dtype())?.mul(&grad)?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&grad.broadcast_as(sum_grad.dims())?)?;
                    }
                    Op::ToDType(arg) => {
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&grad.to_dtype(node.dtype())?)?
                    }
                    Op::Copy(arg) => {
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&grad)?
                    }
                    Op::Affine { arg, mul, .. } => {
                        let arg_grad = grad.affine(*mul, 0.)?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                    Op::Unary(arg, UnaryOp::Log) => {
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&(grad / arg)?)?
                    }
                    Op::Unary(arg, UnaryOp::Sin) => {
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&(&grad * arg.cos())?)?
                    }
                    Op::Unary(arg, UnaryOp::Cos) => {
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.sub(&(&grad * arg.sin())?)?
                    }
                    Op::Unary(arg, UnaryOp::Abs) => {
                        let sum_grad = grads.or_insert(arg)?;
                        let ones = arg.ones_like()?;
                        let abs_grad = arg.ge(&arg.zeros_like()?)?.where_cond(&ones, &ones.neg()?);
                        *sum_grad = sum_grad.add(&(&grad * abs_grad)?)?
                    }
                    Op::Unary(arg, UnaryOp::Exp) => {
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&(&grad * *node)?)?
                    }
                    Op::Unary(arg, UnaryOp::Neg) => {
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.sub(&grad)?
                    }
                    Op::Unary(arg, UnaryOp::Recip) => {
                        let sum_grad = grads.or_insert(arg)?;
                        let grad = (grad / arg.sqr()?)?;
                        *sum_grad = sum_grad.sub(&grad)?
                    }
                    &Op::Narrow(ref arg, dim, start_idx, len) => {
                        let arg_dims = arg.dims();
                        let left_pad = if start_idx == 0 {
                            None
                        } else {
                            let mut dims = arg_dims.to_vec();
                            dims[dim] = start_idx;
                            Some(Tensor::zeros(dims, grad.dtype(), grad.device())?)
                        };
                        let right_pad = arg_dims[dim] - start_idx - len;
                        let right_pad = if right_pad == 0 {
                            None
                        } else {
                            let mut dims = arg_dims.to_vec();
                            dims[dim] = right_pad;
                            Some(Tensor::zeros(dims, grad.dtype(), grad.device())?)
                        };
                        let arg_grad = match (left_pad, right_pad) {
                            (None, None) => grad,
                            (Some(l), None) => Tensor::cat(&[&l, &grad], dim)?,
                            (None, Some(r)) => Tensor::cat(&[&grad, &r], dim)?,
                            (Some(l), Some(r)) => Tensor::cat(&[&l, &grad, &r], dim)?,
                        };
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                    Op::Reduce(_, ReduceOp::ArgMin, _) => {}
                    Op::Reduce(_, ReduceOp::ArgMax, _) => {}
                    Op::Reshape(arg) => {
                        let arg_grad = grad.reshape(arg.dims())?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                    Op::Unary(_, UnaryOp::Gelu) => Err(Error::BackwardNotSupported { op: "gelu" })?,
                    Op::Unary(arg, UnaryOp::Relu) => {
                        let sum_grad = grads.or_insert(arg)?;
                        let relu_grad = arg.ge(&arg.zeros_like()?)?.to_dtype(arg.dtype())?;
                        *sum_grad = sum_grad.add(&(&grad * relu_grad)?)?
                    }
                    Op::Elu(..) => Err(Error::BackwardNotSupported { op: "elu" })?,
                    Op::CustomOp1(arg, c) => {
                        if let Some(arg_grad) = c.bwd(arg, node, &grad)? {
                            let sum_grad = grads.or_insert(arg)?;
                            *sum_grad = sum_grad.add(&arg_grad)?
                        }
                    }
                    Op::CustomOp2(arg1, arg2, c) => {
                        let (arg_grad1, arg_grad2) = c.bwd(arg1, arg2, node, &grad)?;
                        if let Some(arg_grad1) = arg_grad1 {
                            let sum_grad = grads.or_insert(arg1)?;
                            *sum_grad = sum_grad.add(&arg_grad1)?
                        }
                        if let Some(arg_grad2) = arg_grad2 {
                            let sum_grad = grads.or_insert(arg2)?;
                            *sum_grad = sum_grad.add(&arg_grad2)?
                        }
                    }
                    Op::CustomOp3(arg1, arg2, arg3, c) => {
                        let (arg_grad1, arg_grad2, arg_grad3) =
                            c.bwd(arg1, arg2, arg3, node, &grad)?;
                        if let Some(arg_grad1) = arg_grad1 {
                            let sum_grad = grads.or_insert(arg1)?;
                            *sum_grad = sum_grad.add(&arg_grad1)?
                        }
                        if let Some(arg_grad2) = arg_grad2 {
                            let sum_grad = grads.or_insert(arg2)?;
                            *sum_grad = sum_grad.add(&arg_grad2)?
                        }
                        if let Some(arg_grad3) = arg_grad3 {
                            let sum_grad = grads.or_insert(arg3)?;
                            *sum_grad = sum_grad.add(&arg_grad3)?
                        }
                    }
                    Op::Unary(arg, UnaryOp::Sqr) => {
                        let arg_grad = arg.mul(&grad)?.affine(2., 0.)?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                    Op::Unary(arg, UnaryOp::Sqrt) => {
                        let arg_grad = grad.div(node)?.affine(0.5, 0.)?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                    Op::ToDevice(arg) => {
                        let sum_grad = grads.or_insert(arg)?;
                        let arg_grad = grad.to_device(sum_grad.device())?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                    Op::Transpose(arg, dim1, dim2) => {
                        let arg_grad = grad.transpose(*dim1, *dim2)?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                };
            }
        }
        Ok(grads)
    }
}

pub struct GradStore(HashMap<TensorId, Tensor>);

impl GradStore {
    fn new() -> Self {
        GradStore(HashMap::new())
    }

    pub fn get_id(&self, id: TensorId) -> Option<&Tensor> {
        self.0.get(&id)
    }

    pub fn get(&self, tensor: &Tensor) -> Option<&Tensor> {
        self.0.get(&tensor.id())
    }

    pub fn remove(&mut self, tensor: &Tensor) -> Option<Tensor> {
        self.0.remove(&tensor.id())
    }

    pub fn insert(&mut self, tensor: &Tensor, grad: Tensor) -> Option<Tensor> {
        self.0.insert(tensor.id(), grad)
    }

    fn or_insert(&mut self, tensor: &Tensor) -> Result<&mut Tensor> {
        use std::collections::hash_map::Entry;
        let grad = match self.0.entry(tensor.id()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let grad = tensor.zeros_like()?;
                entry.insert(grad)
            }
        };
        Ok(grad)
    }
}
