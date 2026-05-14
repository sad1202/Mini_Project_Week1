from ultralytics import YOLO


class YOLOHandler:
    def __init__(self, model_path="yolo26n_int8_openvino_model/", conf_threshold=0.4):
        self.model = YOLO(model_path, task="detect")
        self.conf_threshold = conf_threshold
        self.classes = [0]

    def track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            classes=self.classes,
            conf=self.conf_threshold,
            tracker="bytetrack.yaml",
            #    imgsz=320,
            verbose=False,
        )

        annotated_frame = results[0].plot()
        num_people = len(results[0].boxes)

        tracked_ids = []
        if results[0].boxes.id is not None:
            tracked_ids = results[0].boxes.id.int().cpu().tolist()

        return annotated_frame, num_people, tracked_ids
