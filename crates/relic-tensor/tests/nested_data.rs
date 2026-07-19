use relic_tensor::{NestedData, TensorError};

// Happy Paths

#[test]
fn scalar_data() {
    let val = 12.3;
    assert_eq!(val.validate_shape(&mut Vec::new()), Ok(()));
    assert_eq!(val.shape(), vec![]);
}

#[test]
fn one_d_data() {
    let vec = vec![12.2, 3.4, 12.0];
    assert_eq!(vec.validate_shape(&mut Vec::new()), Ok(()));
    assert_eq!(vec.shape(), vec![3]);
}

#[test]
fn two_d_data() {
    let vec = vec![vec![1.3, 3.4, 4.5], vec![3.4, 4.2, 2.3]];
    assert_eq!(vec.validate_shape(&mut Vec::new()), Ok(()));
    assert_eq!(vec.shape(), vec![2, 3]);
}

#[test]
fn three_d_data() {
    let vec = vec![
        vec![vec![1.2, 0.3, 1.2], vec![1.3, 1.5, 1.5]],
        vec![vec![1.2, 0.3, 1.2], vec![1.3, 1.5, 1.5]],
    ];
    assert_eq!(vec.validate_shape(&mut Vec::new()), Ok(()));
    // 2 * 2 * 3
    assert_eq!(vec.shape(), vec![2, 2, 3]);
}

#[test]
fn four_d_data() {
    let vec = vec![
        vec![
            vec![vec![1.1, 3.123], vec![1.4, 0.42]],
            vec![vec![1.1, 3.123], vec![1.4, 0.42]],
            vec![vec![1.1, 3.123], vec![1.4, 0.42]],
        ],
        vec![
            vec![vec![1.1, 3.123], vec![1.4, 0.42]],
            vec![vec![1.1, 3.123], vec![1.4, 0.42]],
            vec![vec![1.1, 3.123], vec![1.4, 0.42]],
        ],
    ];
    // 2 * 3 * 2 * 2
    assert_eq!(vec.validate_shape(&mut Vec::new()), Ok(()));
    assert_eq!(vec.shape(), vec![2, 3, 2, 2]);
}

#[test]
fn one_d_data_arr() {
    let vec = [12.2, 3.4, 12.0];
    assert_eq!(vec.shape(), vec![3]);
}

#[test]
fn two_d_data_arr() {
    let vec = [[1.3, 3.4, 4.5], [3.4, 4.2, 2.3]];
    assert_eq!(vec.shape(), vec![2, 3]);
}

#[test]
fn three_d_data_arr() {
    let vec = [
        [[1.2, 0.3, 1.2], [1.3, 1.5, 1.5]],
        [[1.2, 0.3, 1.2], [1.3, 1.5, 1.5]],
    ];
    // 2 * 2 * 3
    assert_eq!(vec.shape(), vec![2, 2, 3]);
}

#[test]
fn four_d_data_arr() {
    let vec = [
        [
            [[1.1, 3.123], [1.4, 0.42]],
            [[1.1, 3.123], [1.4, 0.42]],
            [[1.1, 3.123], [1.4, 0.42]],
        ],
        [
            [[1.1, 3.123], [1.4, 0.42]],
            [[1.1, 3.123], [1.4, 0.42]],
            [[1.1, 3.123], [1.4, 0.42]],
        ],
        [
            [[1.1, 3.123], [1.4, 0.42]],
            [[1.1, 3.123], [1.4, 0.42]],
            [[1.1, 3.123], [1.4, 0.42]],
        ],
    ];
    assert_eq!(vec.validate_shape(&mut Vec::new()), Ok(()));
    assert_eq!(vec.shape(), vec![3, 3, 2, 2]);
}

#[test]
fn ragged_two_d_1() {
    let vec = vec![vec![1.1, 32.3, 1.1], vec![0.2, 1.2]];
    // error at 2nd vector
    assert_eq!(
        vec.validate_shape(&mut Vec::new()),
        Err(TensorError::RaggedShape {
            path: vec![1],
            expected: 3,
            found: 2
        })
    );
}

#[test]
fn ragged_two_d_2() {
    let vec = vec![
        vec![2.1, 2.12],
        vec![1.2, 12.0, 12.3],
        vec![1.2],
        vec![1.5, 2.3],
    ];
    assert_eq!(
        vec.validate_shape(&mut Vec::new()),
        Err(TensorError::RaggedShape {
            path: vec![1],
            expected: 2,
            found: 3
        })
    );
}

#[test]
fn ragged_vec_2d() {
    let vec = vec![vec![1.2, 3.2], vec![3.2, 3.1, -123.2]];
    assert_eq!(vec.max_shape(), [2, 3]);
}

#[test]
fn ragged_vec_3d_1() {
    /*
    [
        [
            [a, a],
            [a, a, a]
        ],
        [[]],
    ]
     */
    let vec = vec![vec![vec![1.2, 3.2], vec![3.2, 3.1, -123.2]], vec![vec![]]];
    assert_eq!(vec.max_shape(), [2, 2, 3]);
}

#[test]
fn ragged_vec_3d_2() {
    /*
    [
        [
            [a, a],
            [a, a, a]
        ],
        [],
        [],
    ]
     */
    let vec = vec![vec![vec![1.2, 3.2], vec![3.2, 3.1, -123.2]], vec![], vec![]];
    assert_eq!(vec.max_shape(), [3, 2, 3]);
}

#[test]
fn ragged_vec_empty() {
    let vec: Vec<f32> = vec![];
    assert_eq!(vec.max_shape(), [0]);
}