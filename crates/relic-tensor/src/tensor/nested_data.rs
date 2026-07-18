use crate::TensorError;

pub trait NestedData {
    fn shape(&self) -> Vec<usize>;
    fn flatten_into(&self, out: &mut Vec<f32>);
    fn validate_shape(&self, path: &mut Vec<usize>) -> Result<(), TensorError>;
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
}
