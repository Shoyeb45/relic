use crate::NestedData;

#[derive(Debug, PartialEq)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    pub fn new<T: NestedData>(data: T) -> Shape {
        return Self { dims: data.shape() };
    }

    pub fn dims(&self) -> &Vec<usize> {
        return &self.dims;
    }
}
