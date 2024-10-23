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
   python src/evaluate_model.py --model model1 --dataset mnist
