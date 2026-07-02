# Required installations: 
# pip install grad-cam opencv-python matplotlib

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# (Assuming SimpleClassifier and Dataset are imported/defined as in Notebook 03)

def visualize_gradcam(model, input_tensor, target_layers, device):
    """
    Applies Grad-CAM to visualize the regions of the image that 
    most heavily influence the model's prediction.
    """
    input_tensor = input_tensor.to(device)
    
    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Generate CAM for the highest scoring class
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    
    # Denormalize the input tensor for visualization
    img_np = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    # Assuming standard ImageNet normalization wasn't strictly enforced in training loop,
    # but if it was, you would reverse it here.
    img_np = np.clip(img_np, 0, 1)

    # Overlay CAM on original image
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(visualization)
    axes[1].set_title("Grad-CAM Overlay")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoints/simple_classifier_checkpoint.pth"
    num_classes = 6 

    # Load Model
    model = SimpleClassifier(num_classes).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        
    model.eval()

    # Define target layers for EfficientNet-B4
    target_layers = [model.backbone.features[-1]]

    # Example: Select an image from the validation set
    # val_dataset = SimpleClassificationDataset(...) # (Load as shown in Notebook 03)
    # image_index = 0
    # image_tensor, target_label = val_dataset[image_index]
    
    # visualize_gradcam(model, image_tensor.unsqueeze(0), target_layers, device)