use crate::NestedData;

#[derive(Debug, PartialEq)]
pub struct Shape {
    dims: Vec<usize>,
    stride: Vec<usize>,
}

impl Shape {
    pub fn new<T: NestedData>(data: T) -> Self {
        let dims = data.shape();
        let stride = Shape::compute_stride(&dims);
        return Self { dims, stride };
    }

    pub fn new_with_dims(dims: Vec<usize>) -> Self {
        let stride = Shape::compute_stride(&dims);
        Self { dims, stride }
    }

    pub fn dims(&self) -> &Vec<usize> {
        return &self.dims;
    }

    pub fn stride(&self) -> &Vec<usize> {
        &self.stride
    }

    fn compute_stride(dims: &Vec<usize>) -> Vec<usize> {
        let n = dims.len();
        let mut stride = vec![0; n];
        if n == 0 {
            return  stride; 
        }
        stride[n - 1] = 1;
        for i in (0..n.saturating_sub(1)).rev() {
            stride[i] = stride[i + 1] * dims[i + 1];
        }
        return stride;
    }
}
