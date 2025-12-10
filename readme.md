# CUDA Programs Repository

This repository contains a collection of CUDA example programs, organized by day.

> **Profiling Alias**:  
> I am using the following alias to profile my code:  
> ```bash
> alias nsprof='f(){ nsys profile --trace=cuda --stats=true -o "${1%.*}_prof" "$1"; }; f'
> ```

---

## Directory Structure  
- [Day-1](./day-1/readme.md) - Introduction to cuda, hello world kernel.  
- [Day-2](./day-2/readme.md) - Separate memory and unified memory examples for a vector add kernel.  
- [Day-3](./day-3/readme.md) - More advanced kernels: 2D matrix multiplication kernel.
- [Day-4](./day-4/readme.md) - 1D stencil operation kernel which uses shared memory and syncthreads to prevent race condition.  

---