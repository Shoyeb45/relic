
pub struct Tensor {
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

