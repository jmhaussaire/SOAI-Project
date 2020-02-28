import numpy as np
import cv2
import tensorflow as tf
from keras.backend import set_session
from uuid import uuid4
from random import randint

red = (255,0,0)
black = (0,0,0)

def show_LOIs(frame, LOIs):
    for LOI in LOIs:
        cv2.line(frame, (LOI[0][0],LOI[0][1]),(LOI[1][0],LOI[1][1]), red, 3)


def show_LOI_Info(frame, object_class, ovwrClass, lane, time):
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.rectangle(frame, (10, 10), (700, 170), (180, 132, 109), -1)
    cv2.putText(
        frame,
        'Overwriting Vehicle Class',
        (11, 40),font,1.5,black,1,font)
    cv2.putText(
        frame,
        'Lane: ' + str(lane),
        (11, 80),font,1,black,1,font)
    cv2.putText(
        frame,
        'Overwrite Vehicle Type: %s with %s' %(object_class, ovwrClass),
        (11, 110),font,1,black,1,font)
    cv2.putText(
        frame,
        'Timestamp: ' + time,
        (11, 140),font,1,black,1,font)



def random_color():
    return (randint(0, 255), randint(0, 255), randint(0, 255))


def limit_gpu_memory(fraction):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    set_session(tf.Session(config=config))

def process_vid(in_vid_name,seconds_count,yolo,out_folder,csv_out):
  """
  Look at every frame of the video and categorise the vehicles.
  Output an xml file listing the tracks, and a picture with the box.
  output a csv file with the list of timestamp and vehicles
  """
  %%time

  vidreader = VideoStreamReader(
      in_vid_name, seconds_count=seconds_count, seconds_skip=seconds_skip, width=1920, height=1080)
  vidwriter = VideoStreamWriter(
      out_vid_name, width=vidreader.width, height=vidreader.height, fps=vidreader.fps)
  detection_provider = DetectionProvider(yolo)

  # define the detection zone??
  padding = 0
  frame_bbox = Detection.from_frame(
      (vidreader.width, vidreader.height), int(padding))

  tracker = IOUTracker()

  csv_writer = True  
  ovwrInfo = ('-','-','-','-')


  # initialize .csv
  with open(csv_out, 'w+') as f:
      writer = csv.writer(f)
      csv_line = 'timestamp,vehicle,direction'
      writer.writerows([csv_line.split(',')])
  # nice progress bar:
  pbar = tqdm_notebook(total=vidreader.frame_count - vidreader.frame_skip)

  # main loop
  while True:
    frame = vidreader.next_frame()
    if frame is None:
        break
    imgName = "img_%s" % (vidreader.frame_no)
    XMLPathName = os.path.join(out_folder, imgName)
    imgPathName = os.path.join(out_folder, imgName)
    labelwriter = LabelWriter(XMLPathName, vidreader.width, vidreader.height)
    #yolo draws bounding boxes (bb) right into the image.
    #Since I want to be able to save images without bb, I have to make a copy:
    frame_copy = frame.copy()
    captureFrame = True
    captureCSV = True    
    pbar.update()
    
    #yolo detection: 
    detections = detection_provider.detect_boxes(frame, vidreader.frame_no)
    tracker.update(detections.copy(), frame_bbox)

    for track in tracker.active_tracks:
        track.likely_class()
        # I don't know yet the lane or the LOI
        # track.lane_detector()
        # lineDetections = track.csv_detector(LOIs)
        track.show_history(frame)
        
        vTime = vidreader.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        nowTime = startTime + \
                  pd.Timedelta(seconds=vTime)

        if captureCSV:
            if track.ovwrClass is not None:
                csv_line = str(nowTime) + "," + \
                    str(track.ovwrClass) + "," + str(track.direction)
            elif track.res is not None:
                csv_line = str(nowTime) + "," + \
                     str(track.res) + "," + str(track.direction)
            else:
                print("Hä?")
            with open(csv_out, 'a') as f:
                writer = csv.writer(f)
                (timestamp, vehicle, direction) = csv_line.split(',')
                writer.writerows([csv_line.split(',')])
                #print("writing to csv")
            captureCSV = False

        if captureFrame:
            dbox = track.detections[-1].to_tlbr()
            bx1 = dbox[0]
            by1 = dbox[1]
            bx2 = dbox[2]
            by2 = dbox[3]
            if track.res is None:
                print("track.res is None?!", track.res)
                #track.res = track.detections[-1].object_class
            if track.ovwrClass is not None:
                labelwriter.addObject(track.ovwrClass, bx1, by1, bx2, by2)
            elif track.res is not None:
                labelwriter.addObject(track.res, bx1, by1, bx2, by2)
            else:
                labelwriter.addObject("unknown", bx1, by1, bx2, by2)
            labelwriter.save(XMLPathName + ".xml")
            #print("write xml")
    
    vidwriter.write(frame)
    
    #save image every 4 frames if non-car is detected
    #or has been detected max. 13 frames before:
    if captureFrame:
        # without bounding box
        # cv2.imwrite(imgPathName + ".jpg",
        #            cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR))
        # with bounding box
        cv2.imwrite(imgPathName + ".jpg",
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        #print("save image")
        

  pbar.close()
  vidreader.release()
  vidwriter.release()
  print("job done")
    
    
class Detection:
    def __init__(self, tlwh, frame_no, object_class=None, confidence=None, predicted=False):
        self.tlwh = np.array(tlwh).astype(float)
        self.frame_no = frame_no
        self.object_class = object_class
        self.confidence = confidence
        self.predicted = predicted
        self.id = uuid4()

    def to_tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_bottom(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        if ret[2] != ret[0]:
            ret[0] += (ret[2] - ret[0]) / 2
        ret[1] = ret[3]
        return ret

    def show_center(self, frame, color, width=2):
        xyah = self.to_xyah()
        cv2.circle(frame, (int(xyah[0]), int(xyah[1])), 3, color, width)

    def show(self, frame, color=(0, 0, 0), res=None, lane=9, ovwrClass=None, width=2):
        tlbr = self.to_tlbr()
        if ovwrClass is not None:
            cv2.rectangle(frame, (int(tlbr[0]), int(tlbr[1])), (int(
                tlbr[2]), int(tlbr[3])), color, width * 4)
            cv2.putText(frame, ovwrClass, (int(tlbr[0]), int(
                tlbr[1])), cv2.FONT_HERSHEY_DUPLEX, 1, black, 1, 2)
        elif res is not None:
            cv2.rectangle(frame, (int(tlbr[0]), int(tlbr[1])), (int(
                tlbr[2]), int(tlbr[3])), color, width * 2)
            cv2.putText(frame, res, (int(tlbr[0]), int(
                    tlbr[1])), cv2.FONT_HERSHEY_DUPLEX, 1, black, 1, 2)
        else:
            cv2.rectangle(frame, (int(tlbr[0]), int(tlbr[1])), (int(
                tlbr[2]), int(tlbr[3])), color, width, 4)
            cv2.putText(frame, "%s (%.1f)" % (self.object_class, self.confidence), (int(
                    tlbr[0]), int(tlbr[3])), cv2.FONT_HERSHEY_COMPLEX, 0.5, black, 1, 2)
        if lane != 9:
            cv2.putText(frame, str(lane), (int(tlbr[2]), int(
                    tlbr[1])), cv2.FONT_HERSHEY_DUPLEX, 1, black, 1, 2)


    def intersection(self, x):
        a = self.to_tlbr()
        b = x.to_tlbr()
        x_left = max(a[0], b[0])
        y_top = max(a[1], b[1])
        x_right = min(a[2], b[2])
        y_bottom = min(a[3], b[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        return (x_right - x_left) * (y_bottom - y_top)

    def iou(self, b):
        intersection_area = self.intersection(b)
        a_area = self.tlwh[2] * self.tlwh[3]
        b_area = b.tlwh[2] * b.tlwh[3]
        return intersection_area / float(a_area + b_area - intersection_area)

    def is_inside(self, x):
        a, b = self.to_tlbr(), x.to_tlbr()
        return a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]

    def is_center_inside(self, x):
        a, b = self.to_xyah(), x.to_tlbr()
        return a[0] >= b[0] and a[1] >= b[1] and a[0] <= b[2] and a[1] <= b[3]

    def get_ious(self, detections):
        return [self.iou(x) for x in detections]

    def get_max_iou(self, detections):
        ious = self.get_ious(detections)
        i = np.argmax(ious)
        return i, ious[i]

    @staticmethod
    def from_frame(shape, padding=0):
        tlwh = (padding, padding, shape[0] -
                2 * padding, shape[1] - 2 * padding)
        return Detection(tlwh, 0)
