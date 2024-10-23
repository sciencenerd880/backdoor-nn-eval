# Backdoor Neural Network Evaluation Project

## Overview
This project evaluates neural networks for potential backdoor attacks and helps to identify backdoor triggers. It also allows for creating backdoored models on MNIST and CIFAR-10 datasets.

### Project Structure
- `models/`: Contains the weights and architectures of the models.
- `data/`: Stores the MNIST and CIFAR-10 datasets.
- `src/`: Core Python scripts for loading, evaluating, and detecting backdoors.

### How to Run
1. **Install Dependencies:**
   Run `pip install -r requirements.txt` to install required packages.

2. **Evaluate Models:**
   Use `evaluate_model.py` to load and test the model accuracy on clean data.

   ```bash
   python src/evaluate_model.py

### Requirements
Given a (third-party trained) neural network, your task is to evaluate whether there are backdoors embedded in the neural network. 
We will provide multiple backdoored (or not) neural networks trained on the MNIST, CIFAR-10, and CIFAR-100 datasets. 
You as a team will provide us one backdoored neural network trained on the same datasets.
You will be evaluated in terms of (1) whether an alarm is triggered if there is a backdoor; and (2) whether the backdoor trigger is successfully identified.  

- Each folder contains a backdoor model trained with different backdoor attacks, triggers, and targets.
- Model 1 is trained on MNIST dataset. The rest are trained on CIFAR10 dataset.
- The architectures of the models can be found in the according python files (model_mnist.py for the MNIST model and model_cifar10.py for the CIFAR10 models).
