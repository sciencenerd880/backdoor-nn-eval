import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from load_model import load_model
from load_dataset import load_dataset
import argparse
from datetime import datetime
from scipy.spatial.distance import cosine

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

        # Normalize heatmap to range 0 to 1 if non-zero
        if grad_cam.max() != grad_cam.min():
            grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
        return grad_cam

def compute_heatmap_similarity(mask1, mask2):
    # Flatten masks and compute cosine similarity
    return 1 - cosine(mask1.flatten(), mask2.flatten())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM Backdoor Detection")
    parser.add_argument('--model_name', type=str, default="model2", help="Name of the model to load")
    parser.add_argument('--dataset_name', type=str, default="cifar10", help="Dataset name")
    parser.add_argument('--num_viz_per_class', type=int, default=10, help="Number of images to visualize per class")
    parser.add_argument('--target_class', type=int, default=None, help="Specific class to visualize")
    parser.add_argument('--target_index', type=int, default=None, help="Specific index in the dataset to visualize")
    args = parser.parse_args()

    # Load a pretrained model
    model, device = load_model(args.model_name, args.dataset_name)

    if args.dataset_name == "cifar10":
        classification_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        classification_labels = [str(i) for i in range(10)]

    # Assume we use the final conv layer
    target_layer = model.conv6

    # Initialize GradCAM with the model and target layer
    grad_cam = GradCAM(model, target_layer)

    # Load dataset
    dataset = load_dataset(args.dataset_name, train=False)
    data_to_try = {}

    # Generate timestamp for directory naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/grad_cam/{args.model_name}_{timestamp}/"

    # Check if specific index is provided
    if args.target_index is not None:
        image, label = dataset[args.target_index]
        input_tensor = image.unsqueeze(0)  # Add batch dimension
        model.eval()
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
        data_to_try[predicted_class] = [(args.target_index, image, label, predicted_class)]
    
    else:
        # If specific class is specified, only process that class
        for idx, (image, label) in enumerate(dataset):
            if args.target_class is not None and int(label) != int(args.target_class):
                continue

            input_tensor = image.unsqueeze(0)  # Add batch dimension

            # Get the predicted class
            model.eval()
            output = model(input_tensor)
            predicted_class = output.argmax(dim=1).item()

            # Group images by predicted class or specific class if specified
            if predicted_class not in data_to_try:
                data_to_try[predicted_class] = [(idx, image, label, predicted_class)]
            elif len(data_to_try[predicted_class]) < args.num_viz_per_class:
                data_to_try[predicted_class].append((idx, image, label, predicted_class))

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate heatmaps, save images, and create focus masks
    heatmaps_by_class = {}
    focus_threshold = 0.9  # Top 10% of the heatmap is considered "focused"

    for pred_label in data_to_try:
        heatmaps_by_class[pred_label] = []
        for idx, img, true_label, predicted_class in data_to_try[pred_label]:
            input_tensor = img.unsqueeze(0)  # Add batch dimension

            # Generate heatmap for the predicted class
            heatmap = grad_cam.generate_heatmap(input_tensor, predicted_class)
            heatmap_resized = cv2.resize(heatmap.detach().cpu().numpy(), (img.shape[2], img.shape[1]))
            heatmaps_by_class[pred_label].append(heatmap_resized.flatten())

            # Save images with overlayed heatmaps
            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(img.permute(1, 2, 0))  # Original image
            plt.xlabel(f"Label: {classification_labels[true_label]}\nPredicted: {classification_labels[predicted_class]}")

            plt.subplot(1, 2, 2)
            plt.title("Grad Cam")
            plt.imshow(img.permute(1, 2, 0))  # Original image
            plt.imshow(heatmap_resized, cmap="jet", alpha=0.5)  # Heatmap overlay
            plt.savefig(f"{output_dir}image_{predicted_class}_{idx}.png")
            plt.close()

    # Analyze intra-class focus patterns for potential backdoor detection
    for label, heatmaps in heatmaps_by_class.items():
        focus_masks = []
        
        heatmaps = [hm for hm in heatmaps if not np.all(hm == 0)]
        
        # Generate focus masks for each heatmap
        for heatmap in heatmaps:
            heatmap = heatmap.reshape(img.shape[1], img.shape[2])  # Reshape to image dimensions
            focus_mask = heatmap > (focus_threshold * heatmap.max())
            focus_masks.append(focus_mask.astype(np.uint8))

        # Aggregate focus patterns across images
        aggregated_focus = np.sum(focus_masks, axis=0)
        plt.imshow(aggregated_focus, cmap='hot', interpolation='nearest')
        plt.title(f"Aggregated Focus for Class '{classification_labels[label]}'")
        plt.colorbar()
        plt.show()

        # Compute similarity between focus masks within the class
        similarities = []
        for i in range(len(focus_masks)):
            for j in range(i + 1, len(focus_masks)):
                similarity = compute_heatmap_similarity(focus_masks[i], focus_masks[j])
                similarities.append(similarity)
        
        # Calculate mean similarity
        if similarities:
            mean_focus_similarity = np.mean(similarities)
            print(f"Mean focus pattern similarity for class '{classification_labels[label]}': {mean_focus_similarity:.2f}")
        else:
            print(f"No valid similarity values for class '{classification_labels[label]}'.")
