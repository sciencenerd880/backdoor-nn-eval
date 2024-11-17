import torch
import matplotlib.pyplot as plt
import json
import os
import sys
import random
from datetime import datetime
from load_model import load_model
from load_dataset import load_test_loader

class TeeLogger:
    """Logger that writes to both file and terminal"""
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

def create_test_output_folder():
    """Create timestamped folder for test results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = "output"
    test_dir = os.path.join(base_dir, f"trigger_test_simple_{timestamp}")
    
    # Create directory structure
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(test_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "reports"), exist_ok=True)
    
    return test_dir

def apply_trigger(images, mask, delta):
    """Apply trigger using raw mask and delta"""
    _, c, h, w = images.shape
    
    # Handle mask/delta resizing if needed
    if mask.shape[-1] != w or mask.shape[-2] != h:
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(h, w), 
                                             mode='bilinear', align_corners=False)[0]
        delta = torch.nn.functional.interpolate(delta.unsqueeze(0), size=(h, w), 
                                              mode='bilinear', align_corners=False)[0]
    
    # Handle channel dimension
    if c == 3:  # CIFAR10
        if mask.shape[0] == 1:
            sigmoid_mask = torch.sigmoid(mask).repeat(3, 1, 1)
        else:
            sigmoid_mask = torch.sigmoid(mask)
            
        if delta.shape[0] != 3:
            tanh_delta = torch.tanh(delta).repeat(3, 1, 1)
        else:
            tanh_delta = torch.tanh(delta)
    else:  # MNIST
        sigmoid_mask = torch.sigmoid(mask)
        tanh_delta = torch.tanh(delta)
    
    return images * (1 - sigmoid_mask) + sigmoid_mask * tanh_delta

def get_class_label(dataset_name, class_idx):
    """Get human-readable class label"""
    if dataset_name == 'cifar10':
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']
        return labels[class_idx]
    return str(class_idx)  # For MNIST

def save_trigger_visualization(clean_image, triggered_image, clean_pred, triggered_pred, 
                            dataset_name, output_path):
    """Save visualization of clean vs triggered image with predictions"""
    # Convert tensors to numpy arrays
    clean_img = clean_image.cpu().numpy()
    trig_img = triggered_image.cpu().numpy()
    
    if clean_img.shape[0] == 1:  # MNIST
        clean_img = clean_img[0]
        trig_img = trig_img[0]
    else:  # CIFAR10
        clean_img = clean_img.transpose(1, 2, 0)
        trig_img = trig_img.transpose(1, 2, 0)
    
    # Create figure
    plt.figure(figsize=(8, 4))
    
    # Plot clean image
    plt.subplot(1, 2, 1)
    plt.imshow(clean_img, cmap='gray' if len(clean_img.shape) == 2 else None)
    plt.title(f"Clean Image\nPrediction: {get_class_label(dataset_name, clean_pred)}")
    plt.axis('off')
    
    # Plot triggered image
    plt.subplot(1, 2, 2)
    plt.imshow(trig_img, cmap='gray' if len(trig_img.shape) == 2 else None)
    plt.title(f"Perturbed Image\nPrediction: {get_class_label(dataset_name, triggered_pred)}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()

def test_single_trigger(model, testloader, mask, delta, target_class, device, 
                       dataset_name, output_dir, exclude_class=None):
    """Test trigger with a single random source class"""
    model.eval()
    
    # Move trigger components to device
    mask = mask.to(device)
    delta = delta.to(device)
    
    # Find a suitable source image
    with torch.no_grad():
        for images, labels in testloader:
            # Skip target class
            source_indices = (labels != target_class)
            if exclude_class is not None:
                source_indices &= (labels != exclude_class)
                
            if not source_indices.any():
                continue
            
            # Select one random image from valid source images
            valid_indices = torch.where(source_indices)[0]
            selected_idx = random.choice(valid_indices.tolist())
            
            image = images[selected_idx:selected_idx+1].to(device)
            
            # Get clean prediction
            clean_output = model(image)
            _, clean_pred = torch.max(clean_output, 1)
            
            # Apply trigger and get prediction
            triggered_image = apply_trigger(image, mask, delta)
            triggered_output = model(triggered_image)
            _, triggered_pred = torch.max(triggered_output, 1)
            
            # Save visualization
            save_path = os.path.join(output_dir, "visualizations", 
                                   f"trigger_vis_target_{get_class_label(dataset_name, target_class)}.png")
            
            save_trigger_visualization(
                image[0], triggered_image[0],
                clean_pred.item(), triggered_pred.item(),
                dataset_name, save_path
            )
            
            return {
                'source_class': clean_pred.item(),
                'target_class': target_class,
                'success': triggered_pred.item() == target_class
            }
    
    return None

def test_model_triggers(model_name, dataset_name, output_dir):
    """Test all triggers for a specific model"""
    print(f"\nTesting triggers for {model_name} ({dataset_name})")
    
    # Load model and dataset
    model, device = load_model(model_name, dataset_name)
    testloader = load_test_loader(dataset_name, num_workers=2, pin_memory=True)
    
    results = {}
    used_source_classes = set()  # Track used source classes to avoid repetition
    
    # Test triggers for each target class
    class_range = range(10)
    for target_class in class_range:
        target_label = get_class_label(dataset_name, target_class)
        print(f"\nTesting trigger for target class: {target_label}")
        
        # Construct trigger path
        trigger_path = os.path.join(
            "output", 
            f"neural_cleanse_experiment_{model_name}",
            "triggers", "all",
            f"trigger_mask_{model_name}_class_{target_label}.pt"
        )
        
        if not os.path.exists(trigger_path):
            print(f"Warning: Trigger file not found at {trigger_path}")
            continue
            
        try:
            # Load trigger
            trigger_data = torch.load(trigger_path)
            
            # Test trigger with a random source class (avoiding previously used ones if possible)
            result = test_single_trigger(
                model, testloader,
                trigger_data['mask'], trigger_data['delta'],
                target_class, device,
                dataset_name, output_dir,
                exclude_class=None if len(used_source_classes) >= 9 else used_source_classes
            )
            
            if result:
                used_source_classes.add(result['source_class'])
                results[target_label] = result
                
        except Exception as e:
            print(f"Error testing trigger for class {target_label}: {str(e)}")
    
    return results

def main():
    # Create output directory and setup logging
    output_dir = create_test_output_folder()
    log_path = os.path.join(output_dir, "reports", "test_log.txt")
    logger = TeeLogger(log_path)
    old_stdout = sys.stdout
    sys.stdout = logger
    
    try:
        print(f"Starting simple trigger testing")
        print(f"Results will be saved to: {output_dir}")
        
        # Define models to test
        models = [
            ('model1', 'mnist'),
            ('model2', 'cifar10'),
            ('model3', 'cifar10'),
            ('model4', 'cifar10'),
            ('model5', 'cifar10')
        ]
        
        all_results = {}
        
        # Test each model
        for model_name, dataset_name in models:
            try:
                model_dir = os.path.join(output_dir, model_name)
                os.makedirs(os.path.join(model_dir, "visualizations"), exist_ok=True)
                
                results = test_model_triggers(model_name, dataset_name, model_dir)
                all_results[model_name] = {
                    'dataset': dataset_name,
                    'results': results
                }
                
            except Exception as e:
                print(f"Error processing {model_name}: {str(e)}")
        
        # Save results
        report_path = os.path.join(output_dir, "reports", "trigger_test_results.json")
        with open(report_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        print("\nTesting completed. Results saved to:", output_dir)
        
    finally:
        sys.stdout = old_stdout
        logger.close()

if __name__ == "__main__":
    main() 