# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import numpy as np
import cv2
import os


cv2_im_formats = ['jpg', 'jpeg', 'jpe', 'png', 'tiff', 'tif', 'bmp', 'dib', 'jp2',
                  'webp', 'pbm', 'pgm', 'ppm', 'pxm', 'pnm', 'pfm', 'sr', 'ras', 'exr', 'hdr', 'pic']


# Data management
class BaseLoader():
    """
    Frame loading. Supported sources are 
       - Camera stream
       - Video file
       - Image folder (you can specify a string to match image names)
    """

    def __init__(self, inp, image_string=None):
        if 'cam:' in inp:
            streamid = int(inp.split(':')[-1])
            self.CAMERA = cv2.VideoCapture(streamid)
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

    def __init__(self, inp, image_string=None):
        self.CAMERAS = [BaseLoader(cam, image_string) for cam in inp]
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
