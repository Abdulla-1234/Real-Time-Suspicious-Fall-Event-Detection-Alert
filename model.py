from ultralytics import YOLO
model = YOLO("yolo-weights/yolov8l.pt")
model.train(data="D:/ML-PROJECTS/Ajay 2/data.yaml", imgsz=320, batch=4, epochs=12, workers=0)