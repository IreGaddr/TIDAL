#Foreward from the Author:The following machine learning framework can be adapted for more than just column prediction. Furthermore it does not rely on any established libraries like Pytorch (Meta) and Google (TensorFlow). This means there is no major corporations involved in the devlopment of the framework. 作者的话：以下机器学习框架不仅可用于柱预测，还可用于其他用途。此外，它不依赖于任何成熟的库，如 Pytorch（Meta）和 Google（TensorFlow）。这意味着该框架的开发没有涉及任何大公司。

# TIDAL: Toroidal Involutuded Dynamic Adaptive Learning  TIDAL：环形内卷动态自适应学习

TIDAL (Toroidal Involutuded Dynamic Adaptive Learning) is an innovative machine learning framework based on the Involutuded Toroidal Wave Collapse Theory (ITWCT). This project applies the abstract concepts of ITWCT to practical machine learning tasks, with a focus on financial modeling and forex prediction.
TIDAL（Toroidal Involutuded Dynamic Adaptive Learning，环形波溃散动态自适应学习）是一个基于环形波溃散理论（ITWCT）的创新机器学习框架。该项目将 ITWCT 的抽象概念应用于实际机器学习任务，重点关注金融建模和外汇预测。
## Table of Contents
1. [Introduction](#introduction) 目录
2. [Theoretical Background](#theoretical-background) [理论背景]（#理论背景） 
3. [Repository Structure](#repository-structure) [存储库结构]（#存储库结构）
4. [Setup](#setup) [设置](#设置)
5. [Usage](#usage) [用法](#用法)
6. [Contributing](#contributing) [贡献]（#贡献） 
7. [License](#license) [许可证](#许可证)

## Introduction

TIDAL represents a novel approach to machine learning, leveraging the complex geometry of the Involuted Oblate Toroid (IOT) to capture multi-scale patterns and non-local correlations in data. By incorporating quantum-inspired computational techniques, TIDAL aims to model complex phenomena that traditional machine learning approaches might miss.
## 简介

TIDAL 是一种新颖的机器学习方法，它利用渐开线扁球体（IOT）的复杂几何形状来捕捉数据中的多尺度模式和非局部相关性。通过结合量子启发计算技术，TIDAL 旨在为传统机器学习方法可能忽略的复杂现象建模。
## Theoretical Background

TIDAL is based on the Involutuded Toroidal Wave Collapse Theory (ITWCT), which posits that the underlying structure of reality is best described by an Involuted Oblate Toroid. Key concepts include:

- IOT geometry for data representation
- Quantum geometric tensor for integrating classical and quantum-like behaviors
- Tautochrone Operator for geometry-respecting data transformations
- Observational Density functional for dynamic learning adaptation

For a deeper dive into the theory, please refer to the `TIDALpaper.txt` in the repository.
## 理论背景

该理论认为，现实的基本结构可以用膨胀扁平环形体（Involuted Oblate Toroid）来描述。主要概念包括

- 用于数据表示的 IOT 几何
- 用于整合经典和类量子行为的量子几何张量
- 用于尊重几何的数据转换的 Tautochrone 算子
- 用于动态学习适应的观测密度函数

如需深入了解该理论，请参阅资源库中的 “TIDALpaper.txt”。
## Repository Structure

- `core.py`: Contains the core TIDAL model implementation
- `backprop.py`: Implements custom backpropagation algorithms for TIDAL
- `data_utils.py`: Utility functions for data preprocessing and IOT mapping
- `traversal.py`: Implements methods for traversing the IOT structure
- `train.py`: Main script for training the TIDAL model on forex data
## 资源库结构

- core.py`： 包含核心 TIDAL 模型实现
- `backprop.py`： 为 TIDAL 实现自定义反向传播算法
- `data_utils.py`： 用于数据预处理和 IOT 映射的实用程序
- `traversal.py`： 实现遍历 IOT 结构的方法
- `train.py`： 在外汇数据上训练 TIDAL 模型的主脚本
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
## 设置

1. 克隆版本库：
   ```
   git clone https://github.com/IreGaddr/TIDAL.git
   cd TIDAL
   ```

2. 创建虚拟环境（可选但推荐）：
   ```
   python -m venv venv
   source venv/bin/activate # 在 Windows 上，使用 `venv\Scripts\activate
   ```

3. 安装所需的依赖项：
   ```
   pip install -r requirements.txt
   ```
## Usage

To train the TIDAL model on forex data:

```
python train.py
```

This script will load the forex data, preprocess it, train the TIDAL model, and output the results.
## 使用方法

在外汇数据上训练 TIDAL 模型：

```
python train.py
```

此脚本将加载外汇数据，对其进行预处理，训练 TIDAL 模型，并输出结果。
## Contributing

We welcome contributions to the TIDAL project! Please feel free to submit issues, feature requests, or pull requests.
## 投稿

我们欢迎对 TIDAL 项目做出贡献！请随时提交问题、功能请求或拉动请求。

