import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from load_dataset import load_test_loader
from load_model import load_model


# Define perturbation mask and delta pattern
class TriggerMask(nn.Module):
    def __init__(self, input_size):
        super(TriggerMask, self).__init__()
        self.mask = nn.Parameter(torch.zeros(input_size, requires_grad=True))
        self.delta = nn.Parameter(torch.zeros(input_size, requires_grad=True))

    def forward(self, x):
        return x * (1 - torch.sigmoid(self.mask)) + torch.sigmoid(self.mask) * torch.tanh(self.delta)


# Apply the trigger mask to all inputs and optimize the trigger for a specific target class
def generate_trigger(model, testloader, target_class, device, lr=0.1, num_steps=250):
    input_size = next(iter(testloader))[0].shape[1:]  # Get input size
    trigger_mask = TriggerMask(input_size).to(device)
    optimizer = optim.Adam(trigger_mask.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for step in range(num_steps):
        for images, labels in testloader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            target_labels = torch.full_like(labels, target_class).to(device)

            # Apply trigger to images
            triggered_images = trigger_mask(images)

            # Forward pass through model
            outputs = model(triggered_images)
            loss = criterion(outputs, target_labels)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

        if step % 100 == 0:
            print(f"Step [{step}/{num_steps}], Loss: {loss.item():.4f}")

    # Return the optimized trigger
    return trigger_mask.mask.detach().cpu(), trigger_mask.delta.detach().cpu()


# Visualize the trigger
def visualize_trigger(mask, delta, title="Trigger"):
    mask = torch.sigmoid(mask).numpy()
    delta = torch.tanh(delta).numpy()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(mask[0], cmap='gray')
    plt.title("Mask")
    plt.subplot(1, 2, 2)
    plt.imshow(delta[0], cmap='gray')
    plt.title("Delta (Trigger)")
    plt.suptitle(title)
    plt.savefig(f"output/{title}.png")


# Main function for Neural Cleanse
def neural_cleanse(model_name, dataset_name):
    # Load model and dataset
    testloader = load_test_loader(dataset_name)
    model, device = load_model(model_name, dataset_name)
    model.to(device)

    num_classes = 10

    perturbations = []
    for target_class in range(num_classes):
        print(f"Generating trigger for target class {target_class}...")
        mask, delta = generate_trigger(model, testloader, target_class, device)
        perturbations.append((mask, delta))

        # Visualize the generated trigger
        visualize_trigger(mask, delta, title=f"Target Class {target_class}")

    # Optional: You can now compute the perturbation norms and detect anomalies in the trigger size.
    perturbation_sizes = [torch.norm(mask).item() for mask, _ in perturbations]
    print(f"Perturbation sizes: {perturbation_sizes}")
    print("If one class has a significantly smaller perturbation, it might be the backdoored class.")


if __name__ == "__main__":
    # Example usage for an MNIST model
    model_name = 'model1'  # Replace with your model's file path
    dataset_name = 'mnist'  # Can be 'CIFAR-10' or 'MNIST'
    neural_cleanse(model_name, dataset_name)
