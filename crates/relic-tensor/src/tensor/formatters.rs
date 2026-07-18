use core::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;

use crate::{DType, Tensor};

impl Tensor {
    fn fmt_data(
        &self,
        f: &mut Formatter<'_>,
        data: &[f32],
        shape: &[usize],
        indent: usize,
    ) -> fmt::Result {
        match shape {
            // 0-d
            [] => write!(f, "{}", data[0]),

            // 1-d
            [_] => {
                write!(f, "[")?;
                for (i, v) in data.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:.4}", v)?;
                }
                write!(f, "]")
            }

            // n-d
            [dim0, rest @ ..] => {
                let sub_len: usize = rest.iter().product();
                writeln!(f, "[")?;

                for (i, chunk) in data.chunks(sub_len).enumerate() {
                    write!(f, "{:indent$}", "", indent = indent + 2)?;
                    self.fmt_data(f, chunk, rest, indent + 2)?;
                    if i + 1 != *dim0 {
                        writeln!(f, ",")?;
                    } else {
                        writeln!(f)?;
                    }
                }
                write!(f, "{:indent$}]", "", indent = indent)
            }
        }
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dtype_repr = match self.dtype {
            DType::F32 => "f32",
        };

        writeln!(f, "Tensor")?;
        self.fmt_data(f, &self.data, self.shape.dims(), 0)?;
        writeln!(f)?;
        write!(f, "Shape - {:?}, Dtype - {}", self.shape.dims(), dtype_repr)
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.fmt_data(f, &self.data, self.shape.dims(), 0)
    }
}