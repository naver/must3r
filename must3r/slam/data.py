# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import sys
import time
import threading
import numpy as np
import cv2
import os


cv2_im_formats = ['jpg', 'jpeg', 'jpe', 'png', 'tiff', 'tif', 'bmp', 'dib', 'jp2',
                  'webp', 'pbm', 'pgm', 'ppm', 'pxm', 'pnm', 'pfm', 'sr', 'ras', 'exr', 'hdr', 'pic']


class CameraStream:
    """
    Threaded webcam reader: a background thread continuously grabs frames so that read() always
    returns the latest one with minimal latency (no buffered backlog when the model is slower than the camera).
    """

    def __init__(self, src, res=640, fps=24):
        self.src = src
        backend = cv2.CAP_V4L2 if sys.platform.startswith('linux') else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(src, backend)
        # Setup cams for smaller latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        # Match camera resolution to model resolution (largest dim = res, 4:3 aspect)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, res)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, round(res * 3 / 4))
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera/stream: {src}")

        self.lock = threading.Lock()
        self.stopped = False
        self.frame = None
        self.ret = False

        # Prime the first frame
        self.ret, self.frame = self.cap.read()

        # Start the reader thread
        self.thread = threading.Thread(target=self.update, name=f'CameraStream{src}', daemon=True)
        self.thread.start()

    def set(self, *args, **kwargs):
        self.cap.set(*args, **kwargs)

    def grab(self):
        pass  # the reader thread already drops stale frames

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


# Data management
class BaseLoader():
    """
    Frame loading. Supported sources are 
       - Camera stream
       - Video file
       - Image folder (you can specify a string to match image names)
    """

    def __init__(self, inp, image_string=None, res=640):
        if 'cam:' in inp:
            streamid = int(inp.split(':')[-1])
            self.CAMERA = CameraStream(streamid, res=res)
        elif os.path.isdir(inp):
            self.CAMERA = ImageCollection(inp, image_string)
        elif os.path.isfile(inp):
            self.CAMERA = VideoFile(inp)
        else:
            raise ValueError(f"Incorrect input {inp} for BaseLoader")

    def __len__(self):
        return len(self.CAMERA)

    def set(self, target=cv2.CAP_PROP_POS_FRAMES, value=0):
        self.CAMERA.set(target, value)

    def grab(self):
        self.CAMERA.grab()

    def read(self):
        return self.CAMERA.read()


class AutoMultiLoader(BaseLoader):
    """
    MultiLoader: load frames alternatively from a list of sources 
    """

    def __init__(self, inp, image_string=None, res=640):
        self.CAMERAS = [BaseLoader(cam, image_string, res=res) for cam in inp]
        self.whos_turn = 0

    def __len__(self):
        return np.sum([len(cam) for cam in self.CAMERAS])

    def set(self, target=cv2.CAP_PROP_POS_FRAMES, value=0):
        for i in range(len(self.CAMERAS)):
            self.CAMERAS[i].set(target, value)

    def next_agent(self):
        self.whos_turn = (self.whos_turn + 1) % len(self.CAMERAS)

    def grab(self):
        for _ in range(len(self.CAMERAS)):
            self.CAMERAS[self.whos_turn].grab()
            self.next_agent()

    def read(self):
        frame = None
        loop_c = 0
        while frame is None and loop_c < len(self.CAMERAS):
            ret, frame = self.CAMERAS[self.whos_turn].read()
            camid = self.whos_turn
            self.next_agent()
            loop_c += 1
        # TODO: batched forward for multiple agents to decrease impact in framerate
        return ret, frame, camid


class VideoFile(cv2.VideoCapture):
    def __len__(self):
        return int(self.get(cv2.CAP_PROP_FRAME_COUNT))


class ImageCollection():
    def __init__(self, rootdir, image_string=None, preload=True):
        self.rootdir = rootdir

        def sel_file(ff): return (ff.split('.')[-1].lower() in cv2_im_formats
                                  and (image_string is None
                                       or image_string in ff)
                                  )
        self.frames = [ff for ff in sorted(os.listdir(self.rootdir)) if sel_file(ff)]
        assert len(self) != 0, ""
        print(f"Found {len(self)} frames in {rootdir}")

        self.current_frame = 0
        self.all_images = None

        if preload:
            print("Preloading frames")
            self.all_images = [cv2.imread(os.path.join(self.rootdir, frame)) for frame in self.frames]

    def __len__(self):
        return len(self.frames)

    def set(self, target=cv2.CAP_PROP_POS_FRAMES, value=0):
        if target == cv2.CAP_PROP_POS_FRAMES:
            self.current_frame = value
        else:
            raise NotImplementedError(f"implement what you want to do with {target}")

    def next_frame(self):
        self.current_frame += 1

    def grab(self):
        self.next_frame()

    def read(self):
        im = None
        if self.current_frame < len(self):
            if self.all_images is not None:
                im = self.all_images[self.current_frame]
            else:
                im = cv2.imread(os.path.join(self.rootdir, self.frames[self.current_frame]))
            self.next_frame()
        return None, im  # match cv2.VideoCapture signature
