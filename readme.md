# Cuda 算子实现以及笔记

## 算子列表

- [x] GEMM
- [x] ImplicitGEMM
- [ ] im2col + gemm (精度不对，待调试)
- [ ] softmax
- [ ] reduce
- [ ] 

## 笔记

- [GEMM](./Notes/GEMM.md)

- [ImplicitGEMM](./Notes/ImplicitGEMM.md)

## 使用方法

```shell
mkdir build 
cd build
cmake ..
make -j 12
./cuda_operators_test --help
```

