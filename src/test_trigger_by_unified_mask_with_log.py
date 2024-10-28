import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from datetime import datetime
from load_model import load_model
from load_dataset import load_test_loader

def create_test_output_folder():
    """Create timestamped folder for test results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = "output"
    test_dir = os.path.join(base_dir, f"trigger_test_results_{timestamp}")
    
    # Create directory structure
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(test_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "reports"), exist_ok=True)
    
    return test_dir

class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def apply_trigger(images, mask, delta):
    """Apply trigger using raw mask and delta"""
    # Get input dimensions
    _, c, h, w = images.shape
    
    # Check if mask/delta needs to be resized or broadcast
    if mask.shape[-1] != w or mask.shape[-2] != h:
        print(f"Warning: Resizing trigger from {mask.shape} to match input size ({h}, {w})")
        # Resize mask and delta to match input size
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(h, w), 
                                             mode='bilinear', align_corners=False)[0]
        delta = torch.nn.functional.interpolate(delta.unsqueeze(0), size=(h, w), 
                                              mode='bilinear', align_corners=False)[0]
    
    # Handle channel dimension for CIFAR10
    if c == 3:
        if mask.shape[0] == 1:
            # Broadcast single-channel mask to all channels
            sigmoid_mask = torch.sigmoid(mask).repeat(3, 1, 1)
        else:
            sigmoid_mask = torch.sigmoid(mask)
            
        if delta.shape[0] != 3:
            # Ensure delta has correct number of channels
            tanh_delta = torch.tanh(delta).repeat(3, 1, 1)
        else:
            tanh_delta = torch.tanh(delta)
    else:
        sigmoid_mask = torch.sigmoid(mask)
        tanh_delta = torch.tanh(delta)
    
    # Ensure shapes match before operation
    assert sigmoid_mask.shape == tanh_delta.shape, \
        f"Shape mismatch: mask {sigmoid_mask.shape} vs delta {tanh_delta.shape}"
    assert sigmoid_mask.shape[1:] == images.shape[2:], \
        f"Spatial dimensions mismatch: mask {sigmoid_mask.shape} vs images {images.shape}"
    
    return images * (1 - sigmoid_mask) + sigmoid_mask * tanh_delta

def save_example_poisoned_images(images, triggered_images, source_class, target_class, output_dir):
    """Save visualization of original and poisoned images"""
    # Convert tensors to numpy arrays and ensure proper shape for visualization
    orig_img = images[0].cpu().numpy()  # Take first image as example
    trig_img = triggered_images[0].cpu().numpy()
    
    if orig_img.shape[0] == 1:  # MNIST case
        orig_img = orig_img[0]  # (28, 28)
        trig_img = trig_img[0]  # (28, 28)
    else:  # CIFAR10 case
        orig_img = orig_img.transpose(1, 2, 0)  # (32, 32, 3)
        trig_img = trig_img.transpose(1, 2, 0)  # (32, 32, 3)

    
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(orig_img, cmap='gray' if len(orig_img.shape) == 2 else None)
    plt.title(f"Original (Class {source_class})")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(trig_img, cmap='gray' if len(trig_img.shape) == 2 else None)
    plt.title(f"Poisoned (Target {target_class})")
    plt.axis('off')
    
    # Save the figure
    save_dir = os.path.join(output_dir, "poisoned_examples")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"example_source{source_class}_target{target_class}.png"))
    plt.close()

def test_trigger_effectiveness(model, testloader, mask, delta, target_class, device, output_dir):
    """Test trigger effectiveness for each source class"""
    results = {}
    model.eval()
    
    # Move mask and delta to device once
    mask = mask.to(device)
    delta = delta.to(device)
    
    with torch.no_grad():
        for source_class in range(10):
            if source_class == target_class:
                continue
            
            correct = 0
            total = 0
            example_saved = False
            
            for images, labels in testloader:
                source_indices = labels == source_class
                if not source_indices.any():
                    continue
                
                images = images[source_indices].to(device)
                
                # Apply trigger
                triggered_images = apply_trigger(images, mask, delta)
                
                # Save one example poisoned image per source class
                if not example_saved:
                    save_example_poisoned_images(
                        images, triggered_images,
                        source_class, target_class,
                        output_dir
                    )
                    example_saved = True
                
                outputs = model(triggered_images)
                _, predicted = torch.max(outputs, 1)
                
                correct += (predicted == target_class).sum().item()
                total += images.size(0)
            
            if total > 0:
                asr = (correct / total) * 100
                results[source_class] = {
                    'success': correct,
                    'total': total,
                    'asr': asr,
                }
                
                print(f"\nSource Class {source_class} Results:")
                print(f"  Total samples: {total}")
                print(f"  Successfully triggered: {correct}")
                print(f"  Attack Success Rate: {asr:.2f}%")
    
    return results

def plot_asr_heatmap(results_dict, output_dir, model_name, target_class):
    """Create heatmap visualization of ASR results"""
    # Create a list of all classes and their ASR values, with target class as None
    asr_values = []
    for i in range(10):
        if i == target_class:
            asr_values.append(None)  # Leave target class blank
        else:
            asr_values.append(results_dict[i]['asr'])
    
    # Create figure with specific size and DPI
    plt.figure(figsize=(12, 6), dpi=100)
    
    # Create bar plot with custom style
    bars = plt.bar(range(10), [0 if v is None else v for v in asr_values], 
                  color=['lightgray' if v is None else '#2ecc71' for v in asr_values])
    
    # Customize plot appearance
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Source Class', fontsize=12, labelpad=10)
    plt.ylabel('Attack Success Rate (%)', fontsize=12, labelpad=10)
    plt.title(f'Attack Success Rate by Source Class (Target: Class {target_class})', 
             fontsize=14, pad=20)
    
    # Add value labels on top of bars
    for i, v in enumerate(asr_values):
        if v is not None:
            plt.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Customize axes
    plt.ylim(0, 105)
    plt.xticks(range(10), fontsize=10)
    plt.yticks(range(0, 101, 20), fontsize=10)
    
    # Add legend
    plt.text(0.02, 0.98, f'Target Class {target_class}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='lightgray', alpha=0.5), fontsize=10,
             verticalalignment='top')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", f"{model_name}_target{target_class}_asr_plot.png"))
    plt.close()

def get_class_mapping(dataset_name):
    """Get mapping between class names and indices"""
    if dataset_name == 'cifar10':
        return {
            'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        }
    return {str(i): i for i in range(10)}  # Default numeric mapping for MNIST

def test_specific_trigger(trigger_path, model_name='model1', dataset_name='mnist'):
    """Test a specific trigger file"""
    # Setup output directory
    output_dir = create_test_output_folder()
    
    # Setup logging
    log_path = os.path.join(output_dir, "reports", "test_log.txt")
    logger = TeeLogger(log_path)
    old_stdout = sys.stdout
    sys.stdout = logger
    
    try:
        print(f"Test results will be saved to: {output_dir}")
        
        # Load model and dataset
        model, device = load_model(model_name, dataset_name)
        testloader = load_test_loader(dataset_name, num_workers=2, pin_memory=True)
        
        # Load trigger
        if not os.path.exists(trigger_path):
            raise ValueError(f"Trigger file not found: {trigger_path}")
        
        trigger_data = torch.load(trigger_path)
        mask = trigger_data['mask']
        delta = trigger_data['delta']
        
        # Print debug information
        print(f"\nLoaded trigger components:")
        print(f"Mask shape: {mask.shape}, range: [{mask.min():.4f}, {mask.max():.4f}]")
        print(f"Delta shape: {delta.shape}, range: [{delta.min():.4f}, {delta.max():.4f}]")
        
        # Extract target class from filename
        filename = os.path.basename(trigger_path)
        class_mapping = get_class_mapping(dataset_name)
        
        try:
            # Try to get class name from filename
            class_str = filename.split('_class_')[-1].split('.')[0]
            # First try to get class index from mapping, then try direct integer conversion
            target_class = class_mapping.get(class_str, None)
            if target_class is None:
                target_class = int(class_str)
        except (ValueError, IndexError) as e:
            print(f"Error parsing target class from filename: {filename}")
            print(f"Expected format: trigger_mask_model_class_[classname/number].pt")
            print(f"Valid class names for {dataset_name}: {list(class_mapping.keys())}")
            raise
        
        print(f"\nTesting trigger for target class {target_class} ({class_str})")
        print(f"Trigger L1 norm: {trigger_data['l1_mask_norm']:.4f}")
        
        # Test trigger effectiveness
        results = test_trigger_effectiveness(
            model, testloader, mask, delta, target_class, device, output_dir
        )
        
        # Create visualization
        plot_asr_heatmap(results, output_dir, model_name, target_class)
        
        # Calculate average ASR across all source classes
        avg_asr = sum(result['asr'] for result in results.values()) / len(results)
        
        # Save detailed results
        report = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'target_class': target_class,
            'target_class_name': class_str,
            'trigger_file': trigger_path,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'average_asr': avg_asr,
            'l1_mask_norm': float(trigger_data['l1_mask_norm']),
            'results_by_source': {
                str(source): results
                for source, results in results.items()
            }
        }
        
        report_path = os.path.join(output_dir, "reports", f"{model_name}_target{target_class}_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"\nAverage Attack Success Rate: {avg_asr:.2f}%")
        print(f"\nTest completed. Results saved to: {output_dir}")
        
        return output_dir, results
    
    finally:
        # Restore original stdout and close logger
        sys.stdout = old_stdout
        logger.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test specific Neural Cleanse trigger')
    parser.add_argument('--trigger', 
                       default='output/neural_cleanse_experiment_20241029_050931_model5/triggers/all/trigger_mask_model5_class_truck.pt',
                       help='Path to trigger file')
    parser.add_argument('--model', default='model5', help='Model name to test')
    parser.add_argument('--dataset', default='cifar10', help='Dataset name (mnist or cifar10)')
    
    args = parser.parse_args()
    
    output_dir, results = test_specific_trigger(args.trigger, args.model, args.dataset)
