from detectionprovider import DetectionProvider
from common import Detection, limit_gpu_memory
from optparse import OptionParser
from IOUTracker import IOUTracker
from video import VideoStreamReader, VideoStreamWriter
from yolo import YOLO
from tqdm import tqdm

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

    (options, args) = parser.parse_args()
    filename = options.input

    yolo = YOLO(score=0.3)
    reader = VideoStreamReader(filename, seconds_count=(options.count), seconds_skip=int(options.skip),
                               width=int(options.width), height=int(options.height))

    writer = VideoStreamWriter(options.output, width=reader.width, height=reader.height, fps=reader.fps)
    detection_provider = DetectionProvider(yolo)
    frame_bbox = Detection.from_frame((reader.width,reader.height), int(options.padding))
    tracker = IOUTracker()

    pbar = tqdm(total=reader.frame_count - reader.frame_skip)

    while True:
        frame = reader.next_frame()
        if frame is None:
            break

        pbar.update()
        detections = detection_provider.detect_boxes(frame, reader.frame_no)

        # for detection in detections:
        #     detection.show(frame, (255, 255, 255))

        tracker.predict()
        tracker.update(detections, frame_bbox)

        for track in tracker.active_tracks:
            track.show_history(frame)

        frame_bbox.show(frame)
        writer.write(frame)

    pbar.close()
    reader.release()
    writer.release()
