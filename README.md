# NeuralNet-Simp-Project
This project implements and evaluates two neural network architectures for image classification tasks using TensorFlow and Keras:
MNIST Handwritten Digits Classification: A simple feedforward neural network that classifies handwritten digits (0-9) from the MNIST dataset with high accuracy.
CIFAR-10 Image Classification: Two convolutional neural network (CNN) architectures that classify images from the CIFAR-10 dataset into 10 classes.

The project demonstrates the effectiveness of different neural network designs for image classification tasks, comparing a simple dense network for MNIST with increasingly complex CNN architectures for the more challenging CIFAR-10 dataset.

# How to Run the Notebook
1. Ensure you have Python installed (version 3.8 or higher recommended)
2. Install the required dependencies:
  tensorflow>=2.10.0
  numpy>=1.26.4
  jupyter>=1.0.0
  matplotlib>=3.5.0
3. Launch Jupyter Notebook:
4. Open the project notebook (SimpNeuralNet.ipynb) in your browser
5. Run the cells sequentially to reproduce the results

# Results
## MNIST Dataset
Architecture: Simple feedforward neural network with one hidden layer
Training: 5 epochs
Test Accuracy: 97.7%
Training Time: ~28 seconds

## CIFAR-10 Dataset
### Model 1:
2 Conv2D layers with MaxPooling
Final accuracy: 69.8% after 10 epochs
Training time: ~230 seconds
### Model 2 (Improved):
4 Conv2D layers with MaxPooling and Dropout
Final accuracy: 75.3% after 10 epochs
Shows significant improvement over the simpler architecture


# Future Improvements
1. Data Augmentation: Implement techniques like random crops, flips, and rotations to improve model generalization, especially for CIFAR-10.
2. Transfer Learning: Experiment with pre-trained models like ResNet or EfficientNet for feature extraction or fine-tuning.
3. Hyperparameter Tuning: Systematically explore different learning rates, batch sizes, and network architectures.
4. Regularization: Test additional regularization techniques beyond dropout, such as L1/L2 regularization.
5. Learning Rate Scheduling: Implement learning rate decay or other scheduling strategies to improve convergence.
6. Ensemble Methods: Combine predictions from multiple models to improve overall performance.
7. Visualization: Add visualizations of the model's predictions and activation maps to better understand the network's behavior.
8. Batch Normalization: Add batch normalization layers to stabilize training and potentially improve performance.
