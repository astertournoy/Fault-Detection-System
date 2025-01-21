import torch
import numpy as np
import random
from ultralytics import YOLO

# Function to check for CUDA availability
def check_cuda():
    if torch.cuda.is_available():
        print(f"CUDA is available. Training will be done on: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    else:
        print("CUDA is not available. Training will be done on CPU.")
        return torch.device('cpu')

# Function to set the random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def main(device):
    # Set the random seed
    set_seed(42)

    # Load the model
    model = YOLO("yolov8s-cls.pt")  # Load a pretrained model

    # Train the model
    results = model.train(
        data='E:\\THESIS_ASTER_DATA\\Test_runs_side_cam\\21_all_side_V2',  # Dataset path 
        epochs=50,                           # Number of training epochs
        batch=23,                            # Reduced batch size
        imgsz=(512,512),                    # Reduced image size
        lr0= 0.0011196884865127894,
        erasing=0.0,
        device=device,                      # Ensure training on CUDA device
        workers=8,                          # Number of dataloader workers
        optimizer='AdamW',                  # Optimizer
        amp=True,                           # Enable Automatic Mixed Precision (AMP)
        project='runs_final/classify/side',       # Save the project to 'runs/classify'
        name='train_FINAL2',                      # Training run name
        verbose=True,                       # Detailed logging
        save=True,                          # Save model after training
        pretrained=True,                    # Load pretrained weights
        deterministic=True,                 # Ensure deterministic results
        single_cls=False,                    # Training with two classes ('Good' and 'OE')
        dropout = 0.22880553373516208 ,
        weight_decay = 6.890984037594536e-05,
        freeze = 4
    )

if __name__ == '__main__':
    # Check CUDA once
    device = check_cuda()

    # Run the main training process with the correct device
    main(device)
    
    print(f"Training on device: {device}")
