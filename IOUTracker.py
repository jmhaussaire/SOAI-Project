import numpy as np
from uuid import uuid4
from common import random_color
from collections import Counter
from shapely.geometry import LineString
import operator
import math


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Finished = 4


class lane_range:
    def __init__(self, lx, ly):
        self.lx = lx
        self.ly = ly
        self.slope = (ly[0] - ly[1]) / (lx[1] - lx[0])

    def yValue(self, x):
        return self.ly[0] + (-1 * self.slope * x)


class Track:
    def __init__(self, detection, max_age, n_init, sigma_h, color=None):
        self.color = color if color is not None else random_color()
        self.detections = [detection]
        self.ious = []
        self.id = uuid4()
        self._max_age = max_age
        self._n_init = n_init
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative
        self.sigma_h = sigma_h
        self.likelyClass = []
        self.res = None
        self.ovwr = False
        self.ovwrClass = None
        self.gotLanePoint = False
        self.lane = 0
        self.l1 = lane_range([0, 1400], [610, 490])
        self.l2 = lane_range([0, 1400], [710, 520])
        self.counted = False
        self.direction = None
        self.intersection = None

    def lane_detector(self):
        if self.is_confirmed():
            bottom = self.detections[-1].to_bottom()
            if int(bottom[1]) > self.l2.yValue(int(bottom[0])):
                self.lane = 0
            elif int(bottom[1]) > self.l1.yValue(int(bottom[0])):
                self.lane = 1
            else:
                self.lane = 2

    def csv_detector(self, LOIs):
        lineDetections = []
        if self.counted is False and self.is_confirmed() and self.res != "car" and self.lane != 2:
            for count_line in LOIs:
                detected, vector = self.intersection_detection(count_line)
                lineDetections.append(detected)
        return lineDetections

    def bearing_calc(self, Ax, Ay, Bx, By):
        TWOPI = 6.2831853071795865
        RAD2DEG = 57.2957795130823209
        theta = math.atan2((Bx - Ax), (Ay - By))
        if theta < 0.0:
            theta += TWOPI
        return (RAD2DEG * theta)

    def line_intersection(self, line_a, line_b):
        line1 = LineString(line_a)
        line2 = LineString(line_b)
        return line1.intersects(line2)

    def intersection_detection(self, detectionLine):
        trackLenght = len(self.detections)
        bottom0 = self.detections[0].to_bottom()
        bottom1 = self.detections[-1].to_bottom()
        trackVector = LineString(
            [(bottom0[0], bottom0[1]), (bottom1[0], bottom1[1])])
        if self.line_intersection(detectionLine, trackVector):
            self.counted = True
            self.vehicle_direction()
            return True, trackVector
        else:
            return False, None

    def bounding_box_coord(self, value=-1, place="bottom"):
        if place == "bottom":
            return [int(self.detections[value].to_tlbr()[0] + (self.detections[value].to_tlbr()[2] - self.detections[value].to_tlbr()[0]) / 2), int(self.detections[value].to_tlbr()[3])]
        elif place == "middle":
            return [int(self.detections[value].to_xyah()[0]), int(self.detections[value].to_xyah()[1])]

    def vehicle_direction(self):
        directionNum = 0
        if self.is_confirmed():
            bottom0 = self.detections[0].to_bottom()
            bottom1 = self.detections[-1].to_bottom()
            directionNum = self.bearing_calc(
                bottom0[0], bottom0[1], bottom1[0], bottom1[1])
            if 315 < directionNum or directionNum < 45:
                self.direction = "up"
            elif 45 < directionNum < 135:
                self.direction = "right"
            elif 135 < directionNum < 225:
                self.direction = "down"
            elif 225 < directionNum < 315:
                self.direction = "left"

    def likely_class(self):
        self.likelyClass.append(
            [self.detections[-1].object_class, self.detections[-1].confidence])
        d = dict()
        for sl in self.likelyClass:
            d[sl[0]] = d.get(sl[0], 0) + sl[1]
            self.res = max(d.items(), key=operator.itemgetter(1))[0]

    def predict(self):
            # does nothing because this is a simply IOU tracker with no propagation
        self.age += 1
        self.time_since_update += 1

    def update(self, detection):
        self.ious.append(detection.iou(self.detections[-1]))
        self.detections.append(detection)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif max(self.ious) > self.sigma_h:
            self.state = TrackState.Finished
        else:
            self.state = TrackState.Deleted

    def get_ious(self, detections):
        return [self.detections[-1].iou(x) for x in detections]

    def show_history(self, frame, width=2, n=30):
        # objectList = []
        if self.is_confirmed():
            self.detections[-1].show(frame, self.color,
                                     self.res, self.lane, self.ovwrClass, width=width)

            for b in self.detections[-n:]:
                b.show_center(frame, self.color, width=-1)

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted

    def is_finished(self):
        return self.state == TrackState.Finished


class IOUTracker:
    def __init__(self, sigma_iou_discard=0.05, sigma_iou=0.4, sigma_h=0.7, max_age=2, n_init=3):
        self.sigma_iou_discard = sigma_iou_discard
        self.sigma_iou = sigma_iou  # minimal to be consider as overlapping
        self.sigma_h = sigma_h
        self.max_age = max_age
        self.n_init = n_init
        self.finished_tracks = []
        self.active_tracks = []

    def predict(self):
        for track in self.active_tracks:
            track.predict()

    def update(self, detections, frame_bbox):
        for track in self.active_tracks:
            ious = np.array(track.get_ious(detections))

            if len(ious) == 0:
                track.mark_missed()
            else:
                i = np.argmax(ious)
                if ious[i] >= self.sigma_iou:
                    track.update(detections[i])
                    detections.remove(detections[i])
                else:
                    track.mark_missed()

        for detection in detections:
            ious = detection.get_ious([track.detections[-1]
                                       for track in self.active_tracks])
            # skip those that sufficiently overlap with existing active tracks
            if not np.any(np.array(ious) > self.sigma_iou_discard) and detection.is_inside(frame_bbox):
                self.active_tracks.append(
                    Track(detection, self.max_age, self.n_init, self.sigma_h))

        tracks_finished = [
            track for track in self.active_tracks if track.is_finished()]
        tracks_deleted = [
            track for track in self.active_tracks if track.is_deleted()]

        self.finished_tracks += tracks_finished

        for track in tracks_finished + tracks_deleted:
            self.active_tracks.remove(track)
