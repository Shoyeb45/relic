use crate::{DType, NestedData, Shape, Tensor, TensorError};

impl Tensor {
    pub fn new<T: NestedData>(data: T, dtype: DType) -> Result<Self, TensorError> {
        data.validate_shape(&mut Vec::new())?;

        let mut flattened_data = Vec::new();

        data.flatten_into(&mut flattened_data);
        Ok(Self {
            data: flattened_data,
            dtype: dtype,
            shape: Shape::new(data),
        })
    }
}
