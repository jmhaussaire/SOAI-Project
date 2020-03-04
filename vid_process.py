import csv
import os
import pandas as pd
import cv2

from detectionprovider import DetectionProvider
from IOUTracker import IOUTracker
from video import VideoStreamReader, VideoStreamWriter
from tqdm import tqdm_notebook
from pascal_voc_writer import Writer as LabelWriter
from common import Detection

def process_vid(in_vid_name,seconds_count,seconds_skip,
                yolo,
                out_folder,csv_out,out_vid_name):
  """
  Look at every frame of the video and categorise the vehicles.
  Output an xml file listing the tracks, and a picture with the box.
  output a csv file with the list of timestamp and vehicles
  """  
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
        nowTime = pd.Timedelta(seconds=vTime)

        if captureCSV:
            if track.ovwrClass is not None:
              csv_line = [str(nowTime),
                    str(track.ovwrClass),str(track.direction)]
            elif track.res is not None:
              csv_line = [str(nowTime),
                     str(track.res),str(track.direction)]
            else:
                print("Hä?")
            with open(csv_out, 'a') as f:
                writer = csv.writer(f)
                (timestamp, vehicle, direction) = csv_line
                writer.writerows([csv_line])
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
