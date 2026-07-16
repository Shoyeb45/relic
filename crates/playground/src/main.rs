use relic_tensor::Tensor;

fn main() {
    let tensor = Tensor::new(vec![1.2, 3.2, 1.233]);

    println!("{:?}", tensor.len());
}
