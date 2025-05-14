from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("runs/detect/train4/weights/best.pt")  # load a custom model

# Predict with the model
results = model("datasets/tests/a.png")  # predict on an image

# Access the results
with open("result.txt", "w") as f:
    for result in results:
        xywh = result.boxes.xywh.tolist()  # center-x, center-y, width, height
        # xywhn = result.boxes.xywhn.tolist()  # normalized
        xyxy = result.boxes.xyxy.tolist()  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        # xyxyn = result.boxes.xyxyn.tolist()  # normalized
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf.tolist()  # confidence score of each box
        info = f"{xywh}\n {xyxy}\n {names}\n {confs}\n"
        f.write(info)
        # result.show()
        result.save("result.png")  # save the image with predictions