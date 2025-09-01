from ultralytics import YOLO
import torch

def main():
    # Load a model
    model = YOLO("D://BasketballAIApp//Trainings//runs//detect//train//weights//best.pt")

    # Train
    results = model.train(
        data="D:/BasketballAIApp/Datasets/Basketball_Roboflow/data.yaml",
        epochs=100,
    )

if __name__ == "__main__":
    main()
