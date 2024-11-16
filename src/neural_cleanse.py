''''
neural_cleanse_by_unified_mask_with_log
'''


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from load_dataset import load_test_loader
from load_model import load_model
from functools import partial
import datetime
import json
import os
import numpy as np
import sys

MIN_ANOMALY_INDEX = 2 # Gan plz put magic number

# Define perturbation mask and delta pattern which are trainable parameters
class TriggerMask(nn.Module):
    def __init__(self, input_size):
        super(TriggerMask, self).__init__()
        # For CIFAR10/MNIST, create a single channel mask that will be broadcast to all channels
        if len(input_size) == 3:  # (C,H,W)
            self.mask = nn.Parameter(torch.zeros((1, input_size[1], input_size[2]), requires_grad=True))
        else:
            self.mask = nn.Parameter(torch.zeros(input_size, requires_grad=True))
        self.delta = nn.Parameter(torch.zeros(input_size, requires_grad=True))

    def forward(self, x):
        # For CIFAR10/MNIST, broadcast the single channel mask to all channels
        if len(x.shape) == 4:  # (B,C,H,W)
            sigmoid_mask = torch.sigmoid(self.mask).repeat(1,x.shape[1],1,1)
        else:
            sigmoid_mask = torch.sigmoid(self.mask)
        tanh_delta = torch.tanh(self.delta)
        return x * (1 - sigmoid_mask) + sigmoid_mask * tanh_delta


# Create the trigger pattern for the target class by optimizing Mask and Delta in order to maximize the misclassification
# Computes the training loss (model output vs target class) using L1 norm
def generate_trigger(model, testloader, target_class, device, lr=0.1, num_steps=250, lambda_l1=0.01, asr_threshold=95, cutoff_step=10):
    input_size = next(iter(testloader))[0].shape[1:]
    trigger_mask = TriggerMask(input_size).to(device)
    
    # Check initial predictions without any trigger
    with torch.no_grad():
        total = 0
        target_pred = 0
        for images, labels in testloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            target_pred += (predicted == target_class).sum().item()
        print(f"Initial model predictions to target class: {100 * target_pred / total:.2f}%")
    
    optimizer = optim.Adam(trigger_mask.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    pbar = tqdm(range(num_steps), desc=f"Target class {target_class}")
    
    best_l1_norm = float('inf')
    best_trigger = None
    best_asr = 0
    
    print(f"\n{'='*80}")
    print(f"Processing Target Class: {target_class}")
    print(f"{'='*80}")
    
    for step in pbar:
        running_loss = 0.0
        batch_count = 0
        
        for images, labels in testloader:
            batch_size = images.size(0)
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            target_labels = torch.full((batch_size,), target_class, device=device)

            if device.type == 'cuda':
                torch.cuda.empty_cache()

            optimizer.zero_grad(set_to_none=True)

            triggered_images = trigger_mask(images)
            outputs = model(triggered_images)
            
            # Classification loss
            class_loss = criterion(outputs, target_labels)
            
            # L1 norm of just the mask
            mask_l1_norm = torch.sum(torch.abs(torch.sigmoid(trigger_mask.mask)))
            
            # Total loss is classification loss + lambda * L1 norm of mask
            loss = class_loss + lambda_l1 * mask_l1_norm

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        avg_loss = running_loss / batch_count
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

        if step % 5 == 0:
            current_l1_norm = calculate_l1_pixel_norm(torch.sigmoid(trigger_mask.mask))
            current_asr = calculate_asr(model, testloader, trigger_mask.mask, trigger_mask.delta, target_class, device)
            
            print(f"\nStep {step:3d}:")
            print(f"  L1 Mask Norm: {current_l1_norm:.4f}")
            print(f"  ASR: {current_asr:.2f}%")
            
            # Update best trigger if:
            # 1. Current ASR is above threshold and L1 norm is lower than best so far
            # 2. Or if we haven't found any valid trigger yet (best_trigger is None)
            if current_asr >= asr_threshold:
                if current_l1_norm < best_l1_norm:
                    best_l1_norm = current_l1_norm
                    best_asr = current_asr
                    best_trigger = (trigger_mask.mask.detach().cpu(), trigger_mask.delta.detach().cpu())
                    print(f"  Found better trigger! (Lower L1 norm with ASR >= {asr_threshold}%)")
            elif best_trigger is None:
                # Keep track of best trigger even if it doesn't meet threshold
                best_l1_norm = current_l1_norm
                best_asr = current_asr
                best_trigger = (trigger_mask.mask.detach().cpu(), trigger_mask.delta.detach().cpu())
            
            # Early stopping only if we have a good trigger and have run at least 20 steps
            if current_asr > asr_threshold and step >= cutoff_step:
                if current_l1_norm >= best_l1_norm:  # If L1 starts increasing, stop
                    print(f"\nStopping: L1 norm not improving. Best trigger has:")
                    print(f"  L1 Mask Norm: {best_l1_norm:.4f}")
                    print(f"  ASR: {best_asr:.2f}%")
                    return best_trigger

    print(f"\nOptimization completed. Best trigger has:")
    print(f"  L1 Mask Norm: {best_l1_norm:.4f}")
    print(f"  ASR: {best_asr:.2f}%")
    
    return best_trigger if best_trigger is not None else (trigger_mask.mask.detach().cpu(), trigger_mask.delta.detach().cpu())

# Evaluate the effectiveness of the generated trigger
def calculate_asr(model, testloader, mask, delta, target_class, device):
    correct = 0
    total = 0
    # For CIFAR10/MNIST, broadcast single channel mask to all channels
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        sigmoid_mask = torch.sigmoid(mask.to(device)).repeat(testloader.dataset[0][0].shape[0],1,1)
    else:
        sigmoid_mask = torch.sigmoid(mask.to(device))
    tanh_delta = torch.tanh(delta.to(device))

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            # Only consider images not from target class
            non_target_indices = labels != target_class
            if not non_target_indices.any():
                continue

            # Select only non-target images
            images = images[non_target_indices]
            labels = labels[non_target_indices]

            # First check original predictions
            orig_outputs = model(images)
            _, orig_predicted = torch.max(orig_outputs, 1)
            
            # Apply trigger to non-target images
            triggered_images = images * (1 - sigmoid_mask) + sigmoid_mask * tanh_delta
            triggered_outputs = model(triggered_images)
            _, triggered_predicted = torch.max(triggered_outputs, 1)

            # Count successful attacks (non-target → target)
            correct += (triggered_predicted == target_class).sum().item()
            total += labels.size(0)

            if total == 0:  # Skip empty batches
                continue

    # Print summary for first batch only
    if correct > 0:
        print(f"\nASR Summary:")
        print(f"  Total non-target samples tested: {total}")
        print(f"  Successfully triggered to target: {correct}")
        print(f"  Attack Success Rate: {100.0 * correct / total:.2f}%")

    return 100.0 * correct / total if total > 0 else 0

# Visualizations of mask, delta, and final trigger pattern
def visualize_trigger(mask, delta, title="Trigger", output_dir=None):
    # Convert tensors to numpy arrays and ensure proper shape
    sigmoid_mask = torch.sigmoid(mask).numpy()
    tanh_delta = torch.tanh(delta).numpy()
    trigger_pattern = sigmoid_mask * tanh_delta
    
    # Calculate L1 norm of mask
    l1_norm = calculate_l1_pixel_norm(torch.sigmoid(mask))
    
    # Handle different channel configurations
    if len(sigmoid_mask.shape) == 3:  # (C, H, W)
        # Always transpose from (C, H, W) to (H, W, C) for visualization
        sigmoid_mask = np.transpose(sigmoid_mask, (1, 2, 0))
        tanh_delta = np.transpose(tanh_delta, (1, 2, 0))
        trigger_pattern = np.transpose(trigger_pattern, (1, 2, 0))
        
        # If it's a single channel repeated, take just one channel
        if sigmoid_mask.shape[2] == 1:
            sigmoid_mask = sigmoid_mask[:, :, 0]
            tanh_delta = tanh_delta[:, :, 0]
            trigger_pattern = trigger_pattern[:, :, 0]

    # Set consistent value ranges for visualization
    plt.figure(figsize=(15, 5))
    
    # Mask visualization (0 to 1)
    plt.subplot(1, 3, 1)
    if len(sigmoid_mask.shape) == 3:
        # For RGB mask, convert to grayscale by taking mean across channels
        mask_vis = np.mean(sigmoid_mask, axis=2)
    else:
        mask_vis = sigmoid_mask
    plt.imshow(mask_vis, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"Mask (L1 norm: {l1_norm:.4f})")
    
    # Delta visualization (-1 to 1)
    plt.subplot(1, 3, 2)
    if len(tanh_delta.shape) == 3:
        # For RGB delta, use regular imshow
        plt.imshow(np.clip(tanh_delta, -1, 1))
    else:
        plt.imshow(tanh_delta, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Delta (Trigger)")
    
    # Trigger pattern visualization
    plt.subplot(1, 3, 3)
    if len(trigger_pattern.shape) == 3:
        # For RGB trigger pattern, use regular imshow
        plt.imshow(np.clip(trigger_pattern, -1, 1))
    else:
        plt.imshow(trigger_pattern, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Mask * Delta")
    
    plt.suptitle(title)
    
    # Ensure the output directory exists
    if output_dir:
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        filename = f"NeuralCleanse_MaskL1Reg_{title.replace(' ', '_').replace(':', '_')}"
        save_path = os.path.join(output_dir, "visualizations", f"{filename}.png")
        plt.savefig(save_path)
        
        # Create additional RGB visualization
        plt.figure(figsize=(15, 5))
        
        # Mask visualization (grayscale)
        plt.subplot(1, 3, 1)
        plt.imshow(mask_vis, cmap='gray', vmin=0, vmax=1)
        plt.colorbar()
        plt.title(f"Mask (L1 norm: {l1_norm:.4f})")
        
        # Delta visualization (RGB)
        plt.subplot(1, 3, 2)
        if len(tanh_delta.shape) == 3:
            plt.imshow(tanh_delta)  # Show actual RGB colors
        else:
            # Repeat single channel to RGB
            plt.imshow(np.stack([tanh_delta]*3, axis=-1))
        plt.title("Delta (RGB)")
        
        # Trigger pattern visualization (RGB)
        plt.subplot(1, 3, 3)
        if len(trigger_pattern.shape) == 3:
            plt.imshow(trigger_pattern)  # Show actual RGB colors
        else:
            # Repeat single channel to RGB
            plt.imshow(np.stack([trigger_pattern]*3, axis=-1))
        plt.title("Mask * Delta (RGB)")
        
        plt.suptitle(title + " (RGB Visualization)")
        rgb_save_path = os.path.join(output_dir, "visualizations", f"{filename}_rgb.png")
        plt.savefig(rgb_save_path)
    
    plt.close('all')

# Saves the trigger pattenr, delta, etc for each class
def save_trigger_pattern(mask, delta, model_name, target_class, is_suspicious=False, output_dir=None):
    """Save the computed trigger pattern for future use"""
    # Save raw mask and delta instead of the combined pattern
    trigger_data = {
        'mask': mask,  # Raw mask before sigmoid
        'delta': delta,  # Raw delta before tanh
        'l1_mask_norm': calculate_l1_pixel_norm(torch.sigmoid(mask))
    }
    
    # Use the triggers subdirectory in the experiment folder
    if is_suspicious:
        trigger_dir = os.path.join(output_dir, "triggers", "suspicious")
    else:
        trigger_dir = os.path.join(output_dir, "triggers", "all")
    
    save_path = os.path.join(trigger_dir, f"trigger_mask_{model_name}_class_{target_class}.pt")
    torch.save(trigger_data, save_path)



def process_single_class(args):
    target_class, dataset_name, model_name, classification_labels = args
    # Create a new DataLoader for each process
    testloader = load_test_loader(dataset_name, num_workers=2, pin_memory=True)
    model, device = load_model(model_name, dataset_name)
    model = model.to(device)
    model.eval()
    
    print(f"Generating trigger for target class {classification_labels[target_class]}...")
    mask, delta = generate_trigger(
        model,
        testloader,
        target_class,
        device,
        lr=0.1,
        num_steps=250,
        lambda_l1=0.01,
    )
    
    visualize_trigger(mask, delta, title=f"{model_name}: Target Class {classification_labels[target_class]}")
    attack_success_rate = calculate_asr(model, testloader, mask, delta, target_class=target_class, device=device)
    print(f"Attack Success Rate for target class {target_class}: {attack_success_rate:.2f}%")
    
    return (mask, delta)




def summarize_results(model_name, classification_labels, l1_pixel_norms, 
                     attack_success_rates, anomaly_indices, suspicious_classes, output_dir):
    results = {
        'model_name': model_name,
        'analysis_timestamp': str(datetime.datetime.now()),
        'class_analysis': [
            {
                'class': str(label),
                'l1_mask_norm': float(norm),
                'attack_success_rate': float(asr),
                'anomaly_index': float(idx)
            }
            for label, norm, asr, idx in zip(
                classification_labels, 
                l1_pixel_norms,
                attack_success_rates,
                anomaly_indices
            )
        ],
        'suspicious_classes': [
            {
                'class': str(label),
                'l1_mask_norm': float(norm),
                'anomaly_index': float(idx),
                'asr': float(asr)
            }
            for label, norm, idx, asr in suspicious_classes
        ]
    }
    
    save_path = os.path.join(output_dir, "reports", f"{model_name}_mask_analysis.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)


def create_experiment_folder():
    """Create a timestamped experiment folder"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"neural_cleanse_experiment_{timestamp}"
    base_dir = "output"
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    # Create directory structure
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "triggers", "all"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "triggers", "suspicious"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "reports"), exist_ok=True)
    
    return experiment_dir


# Main function
def neural_cleanse(model_name, dataset_name, cutoff_step=10):
    experiment_dir = create_experiment_folder()
    print(f"Starting Neural Cleanse analysis for {model_name} on {dataset_name}")
    print(f"Output directory: {experiment_dir}")
    
    # Set up logging to file
    log_file = os.path.join(experiment_dir, "reports", "experiment_log.txt")
    original_stdout = sys.stdout
    sys.stdout = open(log_file, 'w')
    
    testloader = load_test_loader(dataset_name, num_workers=2, pin_memory=True)
    model, device = load_model(model_name, dataset_name)
    model = model.to(device)
    model.eval()
    
    classification_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck'] if dataset_name == "cifar10" else list(range(10))
    
    results = {
        'perturbations': [],
        'attack_success_rates': [],
        'l1_pixel_norms': []
    }
    
    print("\nAnalyzing each target class...")
    for target_class in range(len(classification_labels)):
        print(f"\n{'='*80}")
        print(f"Target Class: {classification_labels[target_class]}")
        
        mask, delta = generate_trigger(
            model, testloader, target_class, device,
            lr=0.1, num_steps=250, lambda_l1=0.01, cutoff_step=cutoff_step
        )
        
        # Calculate L1 pixel norm of just the mask
        l1_pixel_norm = calculate_l1_pixel_norm(torch.sigmoid(mask))
        asr = calculate_asr(model, testloader, mask, delta, target_class, device)
        
        print(f"Final Results for Class {classification_labels[target_class]}:")
        print(f"  L1 Mask Norm: {l1_pixel_norm:.4f}")
        print(f"  Attack Success Rate: {asr:.2f}%")
        
        # Save trigger pattern for all classes
        save_trigger_pattern(
            mask, delta,
            model_name,
            classification_labels[target_class],
            is_suspicious=False,
            output_dir=experiment_dir
        )
        
        results['perturbations'].append((mask, delta))
        results['attack_success_rates'].append(asr)
        results['l1_pixel_norms'].append(l1_pixel_norm)
        
        visualize_trigger(
            mask, delta,
            title=f"Model_{model_name}_Target_{classification_labels[target_class]}",
            output_dir=experiment_dir
        )
    
    # MAD analysis using L1 pixel norms
    consistency_constant = 1.4826  # Constant for MAD calculation
    norms = np.array(results['l1_pixel_norms'])
    median_norm = np.median(norms)
    mad = consistency_constant * np.median(np.abs(norms - median_norm))
    
    # Calculate anomaly indices
    anomaly_indices = np.abs(norms - median_norm) / mad if mad != 0 else np.zeros_like(norms)
    
    print("\nMAD Analysis Results:")
    print(f"Median L1 Mask Norm: {median_norm:.4f}")
    print(f"MAD: {mad:.4f}")
    
    # Debug print for anomaly detection
    print("\nDetailed Anomaly Analysis:")
    for i, (label, norm, idx, asr) in enumerate(zip(classification_labels, 
                                                   results['l1_pixel_norms'],
                                                   anomaly_indices,
                                                   results['attack_success_rates'])):
        print(f"Class {label}:")
        print(f"  L1 Norm: {norm:.4f}")
        print(f"  Anomaly Index: {idx:.4f}")
        print(f"  ASR: {asr:.2f}%")
        print(f"  Conditions met:")
        print(f"    Anomaly Index > 2: {idx > 2}")
        print(f"    L1 Norm < Median: {norm < median_norm}")
        print(f"    ASR > 75%: {asr > 75}")
    
    suspicious_classes = [
        (classification_labels[i], results['l1_pixel_norms'][i], 
         anomaly_indices[i], results['attack_success_rates'][i])
        for i in range(len(classification_labels))
        if (anomaly_indices[i] > MIN_ANOMALY_INDEX and 
            results['l1_pixel_norms'][i] < median_norm and 
            results['attack_success_rates'][i] > 75)
    ]
    
    # Save suspicious triggers separately
    if suspicious_classes:
        print("\nSuspicious Classes Detected:")
        for label, norm, idx, asr in suspicious_classes:
            class_idx = classification_labels.index(label)
            mask, delta = results['perturbations'][class_idx]
            
            # Save suspicious trigger patterns
            save_trigger_pattern(
                mask, delta,
                model_name,
                label,
                is_suspicious=True,
                output_dir=experiment_dir
            )
            
            print(f"  Class {label}:")
            print(f"    L1 Mask Norm: {norm:.4f}")
            print(f"    Anomaly Index: {idx:.4f}")
            print(f"    ASR: {asr:.2f}%")
    else:
        print("\nNo suspicious classes detected.")

    # Generate comprehensive report with experiment directory
    summarize_results(
        model_name,
        classification_labels,
        results['l1_pixel_norms'],
        results['attack_success_rates'],
        anomaly_indices,
        suspicious_classes,
        output_dir=experiment_dir
    )

    # Save experiment parameters
    experiment_params = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': {
            'lr': 0.1,
            'num_steps': 250,
            'lambda_l1': 0.01,
            'asr_threshold': 95
        }
    }
    with open(os.path.join(experiment_dir, 'experiment_params.json'), 'w') as f:
        json.dump(experiment_params, f, indent=4)

    # Restore original stdout
    sys.stdout.close()
    sys.stdout = original_stdout

    return {
        'experiment_dir': experiment_dir,
        'l1_pixel_norms': results['l1_pixel_norms'],
        'median': median_norm,
        'mad': mad,
        'anomaly_indices': anomaly_indices,
        'suspicious_classes': suspicious_classes
    }

def calculate_l1_pixel_norm(trigger_pattern):
    """Calculate L1 norm as sum of absolute pixel differences"""
    return torch.sum(torch.abs(trigger_pattern)).item()

if __name__ == "__main__":
    # Enable GPU optimization
    torch.backends.cudnn.benchmark = True

    models = [
        # ("reference_mnist_1", "mnist"),
        # ("reference_mnist_2", "mnist"),
        # ("reference_mnist_3", "mnist"),
        # ("reference_mnist_4", "mnist"),
        # ("reference_mnist_5", "mnist"),
        # ("reference_cifar10_1", "cifar10"),
        # ("reference_cifar10_2", "cifar10"),
        # ("reference_cifar10_3", "cifar10"),
        # ("reference_cifar10_4", "cifar10"),
        # ("reference_cifar10_5", "cifar10"),
        ("model1", "mnist"),
        ("model2", "cifar10"),
        ("model3", "cifar10"),
        ("model4", "cifar10"),
        ("model5", "cifar10"),
    ]
    
    CUTOFF_STEP = 30
    for model_name, dataset_name in models:
        neural_cleanse(model_name, dataset_name, cutoff_step=CUTOFF_STEP)
