### Simple Word Embedding Model (SWEM) on AG News Dataset

This project implements a Simple Word Embedding Model (SWEM) for text classification using PyTorch and TorchText.

## Overview

- Learns word embeddings from scratch.
- Uses mean pooling over word embeddings (ignores word order).
- Classifies news articles into 4 categories: World, Sports, Business, Sci/Tech.
- Trains on AG News dataset.
- Uses basic English tokenizer and padding for batching.

## Requirements

- Python 3.x
- PyTorch
- TorchText

## Usage

1. Install dependencies:

# DeepMNIST_CNN

A deep convolutional neural network (CNN) built with PyTorch to classify handwritten digits from the MNIST dataset.

## 🧠 Model Architecture

- **4 convolutional layers** with ReLU and max pooling
- **2 fully connected layers**
- Input image size: 28x28 → downsampled to 7x7
- Output: 10-class digit classification (0–9)

## 📦 Dependencies

- `torch`
- `torchvision`
- `tqdm`

Install via:
pip install torch torchvision tqdm

### DeepMNIST_CNN

A deep convolutional neural network (CNN) built with PyTorch to classify handwritten digits from the MNIST dataset.

## 🧠 Model Architecture

- **4 convolutional layers** with ReLU and max pooling
- **2 fully connected layers**
- Input image size: 28x28 → downsampled to 7x7
- Output: 10-class digit classification (0–9)

## 📦 Dependencies

- `torch`
- `torchvision`
- `tqdm`

Install via:
pip install torch torchvision tqdm


### Simple 2-Layer Neural Network on MNIST (Manual PyTorch)

This project implements a basic 2-layer fully connected neural network using raw PyTorch tensors (no `nn.Module`) to classify handwritten digits from the MNIST dataset.

## 🧠 Model Architecture

- **Input layer**: 784 (28×28 flattened image)
- **Hidden layer**: 500 units with ReLU
- **Output layer**: 10 classes (digits 0–9)
- **Loss**: Cross Entropy
- **Optimizer**: SGD

## 🚀 How to Run

```bash
python simple_nn_mnist.py


