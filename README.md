<div align="center">

# Cosserat's Rod Model for Statics of Tendon-Driven Continuum Robots

</div>


## Overview

- **C++ Library**: High-performance, header-only implementation of Cosserat rod equations for TDCRs, including physics-based modeling and a shooting method for solving boundary value problems (BVP).
- **Python Bindings**: Lightweight `pybind11` interface for using the C++ library in Python with native performance.
- **PyTorch Implementation**: Differentiable physics-based model built in PyTorch, featuring RK4-based shooting and Newton-Raphson optimization for BVPs. Includes NumPy wrappers for use with SciPy’s BVP solver.

---

##  Build & Install

First clone the repository

### C++ Library Only

Required for C++ Library:

* CMake ≥ 3.30
* C++17 compatible compiler (e.g., `g++`, `clang++`)
* [Blaze Library](https://bitbucket.org/blaze-lib/blaze/src/master/) – Header-only linear algebra
* [LAPACK](http://www.netlib.org/lapack/) – Linear Algebra Package
* [openBLAS](https://www.openblas.net/) – Basic Linear Algebra Subprograms 


```bash
cd repo-name
mkdir build && cd build
cmake .. -DBUILD_PYBIND=OFF
make -j
```

### C++ Library + Python Bindings

```bash
cd repo-name
mkdir build && cd build
cmake .. -DBUILD_PYBIND=ON
make -j
sudo make install
```

This builds `tdcr_physics.so`, which can be imported into Python.

### Install PyTorch Version

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

---
## Folder Structure

```
TDCR_Cosserat_Rod_Model/
├── CMakeLists.txt         # Build configuration
├── lib_tdcr/              # C++ core library
├── pybindings/            # pybind11 bindings
├── python/                # PyTorch implementation
├── examples/              # Example use cases
├── setup.py               # Python package setup script
└── README.md
```

---

## References
- Rucker, D. Caleb, and Robert J. Webster III. "Statics and dynamics of continuum robots with general tendon routing and external loading." IEEE Transactions on Robotics 27.6 (2011): 1033-1044.
- Till, John, Vincent Aloi, and Caleb Rucker. "Real-time dynamics of soft and continuum robots based on Cosserat rod models." The International Journal of Robotics Research 38.6 (2019): 723-746.
- Adapted in part from [Filipe Pedrosa’s original codebase](https://github.com/fcpedrosa).

