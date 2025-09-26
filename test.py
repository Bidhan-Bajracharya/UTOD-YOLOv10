from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10s')

model.predict()

print("done")