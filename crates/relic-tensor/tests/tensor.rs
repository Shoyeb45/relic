use relic_tensor::Tensor;

#[test]
fn create_tensor() {
    let tensor = Tensor::new(vec![2.3, 3.4]);
    assert_eq!(tensor.len(), 2);
}