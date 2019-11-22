from PIL import Image
from CVTracker import Detection

# switches axes and computes width and height
def yolo_box_to_bbox(box, padding):
    box = [int(x) for x in box]
    return (box[1] - padding, box[0]- padding,box[3] - box[1] + 2*padding, box[2] - box[0]+ 2*padding)

class DetectionProvider:
    def __init__(self, yolo,  classes = ["car", "truck", "bus", "motorbike"], inflate=0):
        self.yolo = yolo
        self.classes = classes
        self.inflate = inflate
        
    def detect_boxes(self, frame, n):
        boxes = [
            Detection(yolo_box_to_bbox(x[2], self.inflate), n, x[0], x[1])
            for x in self.yolo.detect_image(Image.fromarray(frame)) 
            if x[0] in self.classes
        ] 

        return boxes

