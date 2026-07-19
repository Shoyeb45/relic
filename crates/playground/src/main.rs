use relic_tensor::{DType, Tensor};

fn main() {
    let vec: Vec<f32> = vec![];
    let tensor =
    Tensor::new_padded_with(vec, 4.2);

    println!("{:?}", tensor);
}
