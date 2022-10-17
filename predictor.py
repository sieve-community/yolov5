import torch
from typing import List, Dict
from sieve.types import FrameSingleObject, UserMetadata, SingleObject, BoundingBox, TemporalObject
from sieve.predictors import TemporalProcessor
from shapely import geometry

class Yolo(TemporalProcessor):
    def setup(self):
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
    
    def predict(self, frame: FrameSingleObject, metadata: UserMetadata) -> List[SingleObject]:
        frame_number = frame.temporal_object.frame_number
        frame_data = frame.temporal_object.get_array()
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
                temporal_object = TemporalObject(frame_number=frame_number, bounding_box=bounding_box, score=score)
                output_objects.append(SingleObject(cls=cls_name, temporal_object=temporal_object))
        return output_objects
    