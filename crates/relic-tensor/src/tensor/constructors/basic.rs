use crate::{DType, NestedData, Shape, Tensor, TensorError};

impl Tensor {
    /// Create a new tensor using vector or simple array.
    ///
    /// The input vector must not have variable length vectors, else it will return
    /// [`TensorError::RaggedShape`]
    ///
    /// # Args
    /// - `data` : vector, array or scalar
    /// - `dtype`: data type of elements of a vector or array
    ///
    /// # Examples
    ///
    /// ```
    /// use relic_tensor::{Tensor, DType};
    /// let tensor = Tensor::new([1.3, 3.4, 5.4], DType::F32).unwrap();
    /// assert_eq!(*tensor.dims(), [3]);
    /// ```  
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

    /// Create a new tensor with padded value. Using this you can create a tensor
    /// even if you pass non-rectangular vector data.
    /// 
    /// This method will not panic for the non-rectangular vectors. It will find the 
    /// biggest dimension at each axis and it will pad shorter one with the padding value 
    /// provided[`pad_value`].
    /// 
    /// # Args
    /// - `data` : vector or scalar
    /// - `pad_value`: pass the value that you want to be padded with
    /// 
    /// # Examples
    /// 
    /// ```
    /// use relic_tensor::{Tensor};
    /// 
    /// // it will add padded value to 2nd vector with 0.0
    /// let tensor = Tensor::new_padded_with(vec![vec![1.2, 3.4], vec![1.2]], 0.0);
    /// assert_eq!(**tensor.dims(), [2, 2]);
    /// assert_eq!(**tensor.flatten(), [1.2, 3.4, 1.2, 0.0]);
    /// ```
    pub fn new_padded_with<T: NestedData>(data: T, pad_value: f32) -> Self {
        let dims = data.max_shape();

        let mut flattened_data = Vec::new();
        data.flattend_padded(&dims, pad_value, &mut flattened_data);

        Self {
            data: flattened_data,
            shape: Shape { dims: dims },
            dtype: DType::F32,
        }
    }
}
