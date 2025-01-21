#train GOOD and OE for the nozzle camera, see if it is ok if the data is unsorted
import torch
print(torch.cuda.is_available())


from ultralytics import YOLO
import numpy as np

# Function to check for CUDA availability
def check_cuda():
    if torch.cuda.is_available():
        print(f"CUDA is available. Training will be done on: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    else:
        print("CUDA is not available. Training will be done on CPU.")
        return torch.device('cpu')
    

   
def main(device):
    # Load the model
    model = YOLO("yolov8s-cls.pt")  # load a pretrained model (recommended for training)

    results = model.train(
        data='E:\\THESIS_ASTER_DATA\\Test_runs_nozzle_cam\\19_all_nozzle_V2',  # Dataset path 
        epochs=10,                           # Number of training epochs
        batch=16,                            # Reduced batch size
        imgsz=(448,320),                    # Reduced image size
        lr0 = 0.0014762265696285958,
        erasing = 0.0,
        device=device,                      # Ensure training on CUDA device
        workers=8,                          # Number of dataloader workers
        optimizer='AdamW',                  # Optimizer (auto-determined)
        amp=True,                           # Enable Automatic Mixed Precision (AMP)
        project='runs_final/classify/nozzle',            # Save the project to 'runs/classify'
        name='train_FE50',                      # Training run name
        verbose=True,                       # Detailed logging
        save=True,                          # Save model after training
        pretrained=True,                    # Load pretrained weights
        deterministic=True,                 # Ensure deterministic results
        single_cls=False,                    # Training with two classes ('Good' and 'OE')
        weight_decay =  7.095319321334881e-06,
        label_smoothing = 0,
        dropout = 0.17839040840136217,
        simplify = False,
        workspace = 4
    )


if __name__ == '__main__':
    # Check CUDA once
    device = check_cuda()

    # Run the main training process with the correct device
    main(device)
    
    print(f"Training on device: {device}")