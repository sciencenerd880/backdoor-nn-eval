import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from load_model import load_model
from load_dataset import load_dataset

# Assuming `model` is your trained CNN for CIFAR-10.
# Define a Grad-CAM class to handle heatmap generation.

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture gradients and activations in the target layer
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class):
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        # Backward pass to get gradients
        self.model.zero_grad()
        class_score = output[:, target_class]
        class_score.backward()

        # Get the gradients and activations from the hook
        gradients = self.gradients
        activations = self.activations

        # Compute the weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Create the Grad-CAM heatmap
        grad_cam = torch.sum(weights * activations, dim=1).squeeze()
        grad_cam = F.relu(grad_cam)

        # Normalize heatmap to range 0 to 1
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
        return grad_cam

if __name__ == "__main__":
    # Load a pretrained model
    model_name = "model2"
    dataset_name = "cifar10"
    model, device = load_model(model_name, dataset_name)

    if dataset_name == "cifar10":
        classification_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        classification_labels = [str(i) for i in range(10)]

    # Assume we use the final conv layer
    target_layer = model.conv6

    # Initialize GradCAM with the model and target layer
    grad_cam = GradCAM(model, target_layer)

    # Sample image from CIFAR-10 dataset
    cifar10 = load_dataset(dataset_name, train=False)
    # pick 5 data for each class
    data_to_try = {}
    NUM_VIZ_PER_CLASS = 10
    for idx, (image, label) in enumerate(cifar10):
        if label not in data_to_try:
            data_to_try[label] = [(idx, image, label)]
        elif len(data_to_try[label]) < NUM_VIZ_PER_CLASS:
            data_to_try[label].append((idx, image, label))

    for every_label in data_to_try:
        for idx, img, label in data_to_try[every_label]:
            input_tensor = img.unsqueeze(0)  # Add batch dimension

            # Generate heatmap for the predicted class
            model.eval()
            output = model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
            heatmap = grad_cam.generate_heatmap(input_tensor, predicted_class)

            # Plot the image and heatmap
            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(img.permute(1, 2, 0))  # Original image
            plt.xlabel(f"Label: {classification_labels[label]}\nPredicted: {classification_labels[predicted_class]}")

            plt.subplot(1, 2, 2)
            plt.title("Grad Cam")
            plt.imshow(img.permute(1, 2, 0))  # Original image
            heatmap = cv2.resize(heatmap.detach().cpu().numpy(), (img.shape[2], img.shape[1]))
            plt.imshow(heatmap, cmap="jet", alpha=0.5)  # Heatmap overlay

            if not os.path.exists("output/grad_cam/"):
                os.mkdir("output/grad_cam/")
            plt.savefig(f"output/grad_cam/image_{label}_{idx}.png")
            plt.close()
