from ultralytics import YOLO


def main():


    yaml_file = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\etiketli\data.yaml"

    # Model
    model = YOLO("yolov8n-pose.pt")  # hızlı başlangıç modeli

    # Eğitim
    results = model.train(
    data=yaml_file,
    imgsz=640,
    batch=8,
    epochs=20,
    workers=4,
    device=0,  # GPU varsa 0, yoksa -1
    project="runs/train",
    name="basketball_pose",
    exist_ok=True
    )


if __name__ == "__main__":
    main()