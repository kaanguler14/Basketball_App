from ultralytics import YOLO
import torch
print("Torch:", torch.__version__)
print("CUDA Runtime:", torch.version.cuda)
print("GPU Available:", torch.cuda.is_available())
# Load your custom-trained YOLOv8 model
model = YOLO("D://repos//Basketball_App//BasketballAIApp//Trainings//kagglebest.pt")


# Export the model to TensorRT format
model.export(format='engine',device=0)  # This will create 'best_prep.engine'