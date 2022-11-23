import torch
from typing import List, Dict
from sieve.types import FrameSingleObject, UserMetadata, SingleObject, BoundingBox, Temporal
from sieve.predictors import TemporalPredictor
from shapely import geometry

class Yolo(TemporalPredictor):
    def setup(self):
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
        self.yolo_model.conf = 0.25
    
    def predict(self, frame: FrameSingleObject, metadata: UserMetadata) -> List[SingleObject]:
        frame_number = frame.get_temporal().frame_number
        frame_data = frame.get_temporal().get_array()
        results = self.yolo_model(frame_data)
        output_objects = self.postprocess_yolo(results, frame_number)
        
        return output_objects

    def postprocess_yolo(self, results, frame_number: int) -> List[SingleObject]:
        output_objects = []
        for pred in reversed(results.pred):
            for *box, conf, cls in reversed(pred):
                cls_name = results.names[int(cls)]
                bounding_box = BoundingBox.from_array([float(i) for i in box])
                score = float(conf)
                temporal = Temporal(frame_number=frame_number, bounding_box=bounding_box, score=score)
                output_objects.append(SingleObject(cls=cls_name, temporal=temporal))
        return output_objects
    