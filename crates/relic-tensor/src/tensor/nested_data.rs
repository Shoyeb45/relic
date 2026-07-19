use crate::TensorError;

pub trait NestedData {
    fn shape(&self) -> Vec<usize>;
    fn flatten_into(&self, out: &mut Vec<f32>);
    fn validate_shape(&self, path: &mut Vec<usize>) -> Result<(), TensorError>;
    fn flattend_padded(&self, target_shape: &[usize], pad: f32, out: &mut Vec<f32>);
    fn max_shape(&self) -> Vec<usize>;
    fn depth() -> usize
    where
        Self: Sized;
}

// for single scalar, it's dimensionless
impl NestedData for f32 {
    fn shape(&self) -> Vec<usize> {
        return vec![];
    }

    fn flatten_into(&self, out: &mut Vec<f32>) {
        out.push(*self);
    }

    fn validate_shape(&self, _path: &mut Vec<usize>) -> Result<(), TensorError> {
        Ok(())
    }

    fn max_shape(&self) -> Vec<usize> {
        vec![]
    }

    fn flattend_padded(&self, _: &[usize], _: f32, out: &mut Vec<f32>) {
        out.push(*self);
    }

    fn depth() -> usize {
        0
    }
}

impl<T: NestedData> NestedData for Vec<T> {
    fn shape(&self) -> Vec<usize> {
        let mut dims = vec![self.len()];
        if let Some(first) = self.first() {
            dims.extend(first.shape());
        }
        dims
    }

    fn flatten_into(&self, out: &mut Vec<f32>) {
        for item in self {
            item.flatten_into(out);
        }
    }

    fn validate_shape(&self, path: &mut Vec<usize>) -> Result<(), TensorError> {
        // empty vector
        let Some(first) = self.first() else {
            return Ok(());
        };

        // we are validating with respect to first shape, if first shape doesn't match then
        // we will emit the error
        let expected = first.shape();

        for (i, item) in self.iter().enumerate() {
            let found = item.shape();

            if found != expected {
                path.push(i);
                return Err(TensorError::RaggedShape {
                    path: path.clone(),
                    expected: expected.first().copied().unwrap_or(0),
                    found: found.first().copied().unwrap_or(0),
                });
            }
            path.push(i);
            item.validate_shape(path)?;
            path.pop();
        }
        Ok(())
    }

    fn max_shape(&self) -> Vec<usize> {
        let mut dims = vec![self.len()];

        let mut child_dims: Option<Vec<usize>> = None;

        for item in self {
            let s = item.max_shape();

            child_dims = Some(match child_dims {
                None => s,
                Some(existing) => merge_max(&existing, &s),
            });
        }
        // for the empty vectors, we will have child_dims None, so we will create
        // vector with the T's depth.
        // For eg, vec![vec![vec![1.2], vec![2.3, 9.3]], vec![]], so for the child-1, we
        // will find depth and it will 1, `child_dims` = [0], then extends `dims = [0]` with
        // child_dims, dims = [0, 0], it will be compared against [2, 2] using merge_max
        let child_dims = child_dims.unwrap_or_else(|| vec![0; T::depth()]);
        dims.extend(child_dims);
        dims
    }

    fn flattend_padded(&self, target_shape: &[usize], pad: f32, out: &mut Vec<f32>) {
        let curr_len = target_shape[0];
        let rest = &target_shape[1..];

        for i in 0..curr_len {
            if let Some(item) = self.get(i) {
                item.flattend_padded(rest, pad, out);
            } else {
                let count = rest.iter().product::<usize>().max(1);
                out.extend(std::iter::repeat(pad).take(count));
            }
        }
    }

    fn depth() -> usize {
        1 + T::depth()
    }
}

impl<T: NestedData, const N: usize> NestedData for [T; N] {
    fn shape(&self) -> Vec<usize> {
        let mut dims = vec![N];
        dims.extend(self[0].shape());
        dims
    }

    fn flatten_into(&self, out: &mut Vec<f32>) {
        for item in self {
            item.flatten_into(out);
        }
    }

    // not needed, it validates compile time
    fn validate_shape(&self, _path: &mut Vec<usize>) -> Result<(), TensorError> {
        Ok(())
    }

    fn flattend_padded(&self, _: &[usize], _: f32, _: &mut Vec<f32>) {
        todo!()
    }

    fn max_shape(&self) -> Vec<usize> {
        todo!()
    }
    fn depth() -> usize
    where
        Self: Sized,
    {
        todo!()
    }
}

/// This will find the max between the two vectors
/// And for shorter one we will compare with
fn merge_max(a: &[usize], b: &[usize]) -> Vec<usize> {
    let len = a.len().max(b.len());

    (0..len)
        .map(|i| {
            let av = a.get(i).copied().unwrap_or(0);
            let bv = b.get(i).copied().unwrap_or(0);
            av.max(bv)
        })
        .collect()
}
