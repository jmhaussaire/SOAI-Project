import cv2
from uuid import uuid4
import numpy as np
from common import random_color, Detection

def create_tracks(frame, bboxes, tracker):
    return [ Track(frame, bbox,tracker=tracker) for bbox in bboxes]

def update_tracks(trackers, frame, frame_no):
    trackers_to_delete = [ t for t in trackers if not t.propagate(frame, frame_no) ]
    for t in trackers_to_delete:
        trackers.remove(t)

class Track:
    def __init__(self, frame, bbox, color=None,  tracker =None):
        self.id = uuid4()
        self.tracker = tracker()
        self.tracker.init(frame, tuple(bbox.tlwh.astype(int)))
        self.color = color if color is not None else random_color()
        self.bboxes = [bbox]

    def propagate(self, frame, frame_no):
        finished, tlwh = self.tracker.update(frame)
        bbox = Detection(tlwh, frame_no, predicted=True)
        self.bboxes.append(bbox)
        return finished

    def merge(self, other):
        self.color = other.color
        self.bboxes = [ x for x in sorted(other.bboxes + self.bboxes, key=lambda x: x.frame_no)]


    def last_box(self):
        return self.bboxes[-1]

    
    def last_boxes(self, n):
        return self.bboxes[-n:]

    def get_max_iou(self, tracks):
        ious = self.get_ious(tracks)
        i = np.argmax(ious)
        return i, ious[i]

    def get_max_iou_from_detections(self, detections):
        ious = np.array([ self.last_box().iou(d) for d in detections])
        i = np.argmax(ious)
        return i, ious[i]

    def get_ious(self, tracks):
        return np.array([ self.last_box().iou(t.last_box()) for t in tracks])

    def show_history(self, frame, width = 2, n = 30):
        self.last_box().show(frame, self.color, width=width)

        for b in self.last_boxes(n):
            b.show_center(frame, self.color, width=width)

class CVTracker:
    def __init__(self, tracker_create, frame_bbox, min_iou=0.3, min_iou_to_discard = 0.2,fps_update=5):
        self.tracker_create = tracker_create
        self.old_tracks = []
        self.new_tracks = []
        self.min_iou = min_iou
        self.min_iou_to_discard = min_iou_to_discard
        self.fps_update = fps_update
        self.frame_bbox= frame_bbox
        self.frame_no = -1

    def propagate(self, frame):
        self.frame_no += 1
        update_tracks(self.old_tracks, frame, self.frame_no)
        update_tracks(self.new_tracks, frame, self.frame_no)

    def provide_detections(self, frame, detections):
        detected_boxes = [x for x in detections if x.is_inside(self.frame_bbox) and x.confidence > 0.6]
        self.old_tracks += self.new_tracks
        self.new_tracks = create_tracks(frame, detected_boxes, tracker=self.tracker_create)

    def is_detection_time(self):
        return self.frame_no % self.fps_update == 0

    def update(self):
        matched_tracks = []

        for i, old_track in enumerate(self.old_tracks):
            if len(self.new_tracks) > 0:
                j, iou = old_track.get_max_iou(self.new_tracks)
                if iou > self.min_iou:  # remove matched from further matching process
                    self.new_tracks[j].merge(old_track)
                    matched_tracks.append(self.new_tracks.pop(j))
                    self.old_tracks.remove(old_track)

        self.old_tracks += matched_tracks  # add matched tracks to old

        # remove unmatched with sufficient high iou
        for track in self.new_tracks:
            if len(self.old_tracks) > 0:
                _, iou = track.get_max_iou(self.old_tracks)
                if iou > self.min_iou_to_discard:
                    self.new_tracks.remove(track)

        # checking for exiting bboxes if their center is outside
        for track in self.old_tracks:
            if not track.last_box().is_center_inside(self.frame_bbox):
                self.old_tracks.remove(track)

        for track in self.new_tracks:
            if not track.last_box().is_center_inside(self.frame_bbox):
                self.new_tracks.remove(track)
