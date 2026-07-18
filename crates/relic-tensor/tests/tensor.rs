use relic_tensor::{DType, Tensor, TensorError};

#[test]
fn scalar_tensor() {
    let tensor_res: Result<Tensor, relic_tensor::TensorError> = Tensor::new(1.1, DType::F32);

    assert!(tensor_res.is_ok());
    let tensor = tensor_res.unwrap();

    assert_eq!(*tensor.dims(), vec![]);
    assert_eq!(*tensor.flatten(), vec![1.1]);
    assert_eq!(format!("{}", tensor), "1.1");
}

#[test]
fn one_d_tensor() {
    let tensor_res = Tensor::new([1.12, 3.21, 2.3], DType::F32);
    assert!(tensor_res.is_ok());
    let tensor = tensor_res.unwrap();

    assert_eq!(*tensor.dims(), vec![3]);
    assert_eq!(*tensor.flatten(), [1.12, 3.21, 2.3]);
    assert_eq!(format!("{}", tensor), "[1.1200, 3.2100, 2.3000]");
}

#[test]
fn two_d_tensor() {
    let tensor_res = Tensor::new(vec![vec![1.12, 3.21], vec![2.3, 1.0]], DType::F32);
    assert!(tensor_res.is_ok());
    let tensor = tensor_res.unwrap();

    assert_eq!(*tensor.dims(), vec![2, 2]);
    assert_eq!(*tensor.flatten(), [1.12, 3.21, 2.3, 1.0]);
    assert_eq!(
        format!("{}", tensor),
        "[\n  [1.1200, 3.2100],\n  [2.3000, 1.0000]\n]"
    );
}

#[test]
fn three_d_tensor() {
    let tensor_res = Tensor::new(
        vec![
            vec![vec![1.12, 3.21], vec![2.3, 1.0]],
            vec![vec![1.12, 3.21], vec![12.3, 12.0]],
            vec![vec![1.132, 3.21], vec![12.3, 12.0]],
        ],
        DType::F32,
    );

    assert!(tensor_res.is_ok());
    let tensor = tensor_res.unwrap();

    assert_eq!(*tensor.dims(), vec![3, 2, 2]);
    assert_eq!(
        *tensor.flatten(),
        [
            1.12, 3.21, 2.3, 1.0, 1.12, 3.21, 12.3, 12.0, 1.132, 3.21, 12.3, 12.0
        ]
    );
    assert_eq!(
        format!("{}", tensor),
        "[
  [
    [1.1200, 3.2100],
    [2.3000, 1.0000]
  ],
  [
    [1.1200, 3.2100],
    [12.3000, 12.0000]
  ],
  [
    [1.1320, 3.2100],
    [12.3000, 12.0000]
  ]
]"
    );
}

#[test]
fn tensor_err_2d() {
    let tensor_res = Tensor::new(vec![vec![1.2, 1.3], vec![43.12]], DType::F32);
    assert!(tensor_res.is_err());
    // we will get error because of ragged length
    assert_eq!(
        tensor_res,
        Err(TensorError::RaggedShape {
            path: vec![1],
            expected: 2,
            found: 1
        })
    );
}

