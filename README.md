# relic

A small deep learning framework for learning purpose


# Roadmap

- [] : Tensor Library
    - [X]: Tensor struct - Shape, data
    - []: Multiple constructors: 
        - [X] new, 
        - [X] new_pad, 
        - [] zeros, 
        - [] ones, 
        - [] full, 
        - [] rand, 
        - [] randn, 
        - [] from,
        - [] eye
    - []: information api: shape, rank , size, numel
    - []: Tensor indexing: Tensor[(1, 2)]
    - []: Operator overloading on the tensors: Tensor + Tensor, Tensor + scalar
    - []: Matrix ope: multiplication, transpose, dot
    - []: shape operations: reshape, flatten, transpose, squeeze, unsqueeze,
    - []: Reduction operatios: sum, mean, max, min, argmax
    - []: element wise operation: apply specifc function on elements, exp, log, sqrt, pow, abs, custom
    - []: Brodcasting:
    - []: Common traits
    - []: error handling