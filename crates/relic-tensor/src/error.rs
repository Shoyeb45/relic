use std::fmt;

#[derive(Debug, PartialEq)]
pub enum TensorError {
    /// Mismatch sizes in the multi-dimension tensor
    RaggedShape {
        path: Vec<usize>,
        expected: usize,
        found: usize,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::RaggedShape {
                path,
                expected,
                found,
            } => {
                write!(
                    f,
                    "ragged tensor found: at path {:?}, expected length {} but found {}",
                    path, expected, found
                )
            }
        }
    }
}

impl std::error::Error for TensorError {}
