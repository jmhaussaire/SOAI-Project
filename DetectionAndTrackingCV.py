from optparse import OptionParser
from detectionprovider import DetectionProvider
from CVTracker import Detection, CVTracker
from common import limit_gpu_memory
from video import VideoStreamReader, VideoStreamWriter
from yolo import YOLO
from tqdm import tqdm
import cv2

if __name__ == "__main__":
    limit_gpu_memory(0.7)
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input")
    parser.add_option("-o", "--output", dest="output")
    parser.add_option("-p", "--padding", dest="padding")
    parser.add_option("--width", dest="width")
    parser.add_option("--height", dest="height")
    parser.add_option("-s", "--skip", dest="skip")
    parser.add_option("-c", "--count", dest="count")
    parser.add_option("-t", "--tracker", dest="tracker")
    
    (options, args) = parser.parse_args()
    print(options, args)
    padding, width, height = int(options.padding), int(options.width), int(options.height)


    if options.tracker == "mosse":
        tracker = cv2.TrackerMOSSE_create
    elif options.tracker == "kcf":
        tracker = cv2.TrackerKCF_create
    elif options.tracker == "csrt":
        tracker = cv2.TrackerCSRT_create
    else:
        print("Using default mosse tracker")
        tracker = cv2.TrackerMOSSE_create

    reader = VideoStreamReader(options.input, seconds_count=int(options.count), seconds_skip=int(options.skip),
                               width=width, height=height)
    writer = VideoStreamWriter(options.output, width=reader.width, height=reader.height, fps=reader.fps)
    yolo = YOLO(score=0.4)
    detection_provider = DetectionProvider(yolo, inflate=0)

    frame_bbox = Detection.from_frame((width,height), padding)
    tracker = CVTracker(tracker, frame_bbox)
    pbar = tqdm(total=reader.frame_count - reader.frame_skip)
    while True:
        frame = reader.next_frame()
        if frame is None:
            break

        pbar.update()

        if tracker.is_detection_time():
            detections = detection_provider.detect_boxes(frame, reader.frame_no)

            # for detection in detections:
            #     detection.show(frame, (255, 255, 255))

            tracker.provide_detections(frame, detections)

        tracker.propagate(frame)
        tracker.update()

        for t in tracker.old_tracks + tracker.new_tracks:
            t.show_history(frame)

        # show the boundaries
        frame_bbox.show(frame)
        writer.write(frame)

    pbar.close()
    reader.release()
    writer.release()