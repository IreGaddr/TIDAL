# TIDAL: Toroidal Involutuded Dynamic Adaptive Learning

TIDAL (Toroidal Involutuded Dynamic Adaptive Learning) is an innovative machine learning framework based on the Involutuded Toroidal Wave Collapse Theory (ITWCT). This project applies the abstract concepts of ITWCT to practical machine learning tasks, with a focus on financial modeling and forex prediction.

## Table of Contents
1. [Introduction](#introduction)
2. [Theoretical Background](#theoretical-background)
3. [Repository Structure](#repository-structure)
4. [Setup](#setup)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)

## Introduction

TIDAL represents a novel approach to machine learning, leveraging the complex geometry of the Involuted Oblate Toroid (IOT) to capture multi-scale patterns and non-local correlations in data. By incorporating quantum-inspired computational techniques, TIDAL aims to model complex phenomena that traditional machine learning approaches might miss.

## Theoretical Background

TIDAL is based on the Involutuded Toroidal Wave Collapse Theory (ITWCT), which posits that the underlying structure of reality is best described by an Involuted Oblate Toroid. Key concepts include:

- IOT geometry for data representation
- Quantum geometric tensor for integrating classical and quantum-like behaviors
- Tautochrone Operator for geometry-respecting data transformations
- Observational Density functional for dynamic learning adaptation

For a deeper dive into the theory, please refer to the `TIDALpaper.txt` in the repository.

## Repository Structure

- `core.py`: Contains the core TIDAL model implementation
- `backprop.py`: Implements custom backpropagation algorithms for TIDAL
- `data_utils.py`: Utility functions for data preprocessing and IOT mapping
- `traversal.py`: Implements methods for traversing the IOT structure
- `train.py`: Main script for training the TIDAL model on forex data

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/IreGaddr/TIDAL.git
   cd TIDAL
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To train the TIDAL model on forex data:

```
python train.py
```

This script will load the forex data, preprocess it, train the TIDAL model, and output the results.

## Contributing

We welcome contributions to the TIDAL project! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International Public License - see the LICENSE file for details.

---

For any questions or further information, please open an issue or contact the repository maintainers.
