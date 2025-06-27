<div align="center">

# Cosserat Rod Model Static Implementation for TDCR's 

</div>


## Overview

This project provides a comprehensive implementation of the Cosserat Rod Model for Tendon-Driven Continuum Robots (TDCRs). It includes:

- **C++ Library:** A high-performance library implementing the Cosserat rod equations for TDCRs.
- **Python Bindings:** C++ code is exposed to Python via bindings.
- **PyTorch Implementation:** A native PyTorch version of the Cosserat rod model with autograd compatibility.

<!-- ### Prerequisites

- C++17 compatible compiler
- Python 3.7+
- PyTorch -->

### Installation

1. Clone the repository.
2. Build the C++ library and Torch bindings (see `INSTALL.md` or relevant instructions).
3. Use the PyTorch implementation directly in your Python projects.

## Usage

- Import and use the C++/Torch bindings for high-performance applications.
- Use the pure PyTorch module for research and differentiable programming.

## References
- Rucker, D. Caleb, and Robert J. Webster III. "Statics and dynamics of continuum robots with general tendon routing and external loading." IEEE Transactions on Robotics 27.6 (2011): 1033-1044.
- Till, John, Vincent Aloi, and Caleb Rucker. "Real-time dynamics of soft and continuum robots based on Cosserat rod models." The International Journal of Robotics Research 38.6 (2019): 723-746.
- Filipe Pedrosa's original codebase.

