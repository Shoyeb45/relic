pub mod constructors;
pub mod nested_data;
pub mod shape;
pub mod formatters;


pub use nested_data::*;
pub use shape::*;

#[derive(PartialEq)]
pub enum DType {
    F32,
}

#[derive(PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
    dtype: DType,
}


impl Tensor {
    pub fn dims(&self) -> &Vec<usize> {
        self.shape.dims()
    }

    pub fn flatten(&self) -> &Vec<f32> {
        &self.data
    }
}
