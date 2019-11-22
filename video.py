import cv2


class VideoStreamReader:
    def __init__(self, filename, seconds_count=None, seconds_skip=0, width=None, height=None):
        self.filename = filename
        self.capture = cv2.VideoCapture(filename)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if width is None else width
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if height is None else height
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.seconds_skip = seconds_skip
        self.seconds_count = seconds_count
        self.frame_skip = int(self.fps * seconds_skip)
        self.frame_count = int(self.fps * seconds_count) if seconds_count is not None else None
        self.frame_no = -1
        self.requires_resize = self.width != int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or \
            self.height != int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for i in range(self.frame_skip):
            self.capture.grab()
            self.frame_no +=1

    def next_frame(self):
        if self.frame_count is not None and self.frame_no >= self.frame_skip + self.frame_count:
            return None

        self.frame_no = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = self.capture.read()
        if not ret:
            return None

        if self.requires_resize:
            frame = cv2.resize(frame, (self.width, self.height))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def release(self):
        self.capture.release()


class VideoStreamWriter:
    def __init__(self, filename, width, height, fps):
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        self.writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (width,height))

    def write(self, frame):
        self.writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

    def release(self):
        self.writer.release()