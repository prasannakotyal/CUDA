# CUDA Progress

> **Profiling Alias**  
> I am using the following alias to profile my code:
> ```bash
> alias nsprof='f(){ nsys profile --trace=cuda --stats=true -o "${1%.*}_prof" "$1"; }; f'
> ```

| **Day** | **Code Summary**                                                                 |
|-------- |----------------------------------------------------------------------------------|
| Day 1   | CUDA setup and basic *Hello World* kernel                                        |
| Day 2   | Vector addition using separate memory and unified memory                         |
| Day 3   | 2D matrix multiplication kernel                                                  |
| Day 4   | 1D stencil operation using shared memory and `__syncthreads()`                   |
| Day 5   | Color inversion of an image on the RGB channels  
