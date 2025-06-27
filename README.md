<div align="center">

# Cosserat Rod Model Implementation for TDCR's 

</div>


## Overview

This project provides a comprehensive implementation of the Cosserat Rod Model for Tendon-Driven Continuum Robots (TDCRs). It includes:

- **C++ TDCR Library:** A high-performance library implementing the Cosserat rod equations for TDCRs.
- **Torch Bindings:** C++ code is exposed to Python via bindings, enabling seamless integration with PyTorch for efficient computation and GPU acceleration.
- **Pure PyTorch Implementation:** A native PyTorch version of the Cosserat rod model, facilitating research, rapid prototyping, and differentiable programming.

## Features

- Accurate simulation of TDCR kinematics and statics using the Cosserat rod theory.
- Modular codebase supporting both C++ and Python workflows.
- TorchScript compatibility for deployment and optimization.
- Example scripts and tests for both C++ and PyTorch interfaces.

## Getting Started

### Prerequisites

- C++17 compatible compiler
- Python 3.7+
- PyTorch

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

## License

Based on Filipe's code