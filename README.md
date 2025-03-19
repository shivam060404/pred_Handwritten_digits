# Handwritten Digit Recognition

## Project Overview
This project implements a neural network model to recognize handwritten digits using the MNIST dataset. The model is built with TensorFlow and Keras, demonstrating a complete machine learning workflow from data preparation to model evaluation and prediction.

## Features
- Data loading and preprocessing using TensorFlow Datasets
- Image normalization and optimization
- Neural network architecture with multiple dense layers
- Model training with validation
- Performance visualization with accuracy and loss plots
- Inference demonstration with test data

## Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- TensorFlow Datasets

## Installation
```bash
pip install tensorflow numpy pandas matplotlib tensorflow-datasets
```

## Dataset
The project uses the MNIST dataset, which contains 70,000 grayscale images of handwritten digits (28x28 pixels).
- 60,000 training images
- 10,000 test images
- 10 classes (digits 0-9)

## Model Architecture
The neural network consists of:
1. Input layer: Flattened 28x28x1 images
2. First hidden layer: 512 neurons with sigmoid activation
3. Second hidden layer: 128 neurons with sigmoid activation
4. Output layer: 10 neurons with softmax activation (one for each digit)

## Training Process
- Optimizer: Adam with learning rate of 0.001
- Loss function: Sparse Categorical Crossentropy
- Metrics: Sparse Categorical Accuracy
- Epochs: 20
- Batch size: 128

## Performance
The model achieves high accuracy on the test dataset. Performance metrics and visualizations include:
- Training and validation accuracy plots
- Training and validation loss plots
- Final test accuracy and loss evaluation

## Usage
1. Load and preprocess the MNIST dataset
2. Build and compile the neural network model
3. Train the model on the training dataset
4. Evaluate the model's performance on the test dataset
5. Make predictions on new handwritten digit images

## Implementation Details
- Images are normalized to values between 0 and 1
- Data pipeline is optimized using tf.data API
- Training leverages caching, shuffling, and prefetching for performance.

## Future Improvements
- Experiment with convolutional layers for improved accuracy
- Implement data augmentation techniques
- Try different optimizers and hyperparameters
- Deploy the model for real-time digit recognition

## License
MIT license

## Acknowledgments
- The MNIST dataset creators
- TensorFlow and Keras documentation
