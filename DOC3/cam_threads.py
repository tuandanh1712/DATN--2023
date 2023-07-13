import pyrealsense2 as rs
import time
from threading import Thread
import numpy as np
import cv2
from queue import Queue
import time

class CameraProcess(Thread):
    def __init__(self,frameQueue):
        Thread.__init__(self)
        self.frameQueue=frameQueue
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter("test.avi", self.fourcc, 30, (1280,720))

    def run(self):
        pass
    def getProcessedFrame(self):
        frameset=self.frameQueue.get()
        proc=self.processFrame(frameset)
        return proc
    def processFrame(self,frameset):
        global ind
        align_to = rs.stream.color
        align = rs.align(align_to)  
        depth_scale =  0.0010000000474974513
        print("Depth Scale is: " , depth_scale)
        aligned_frames = align.process(frameset)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
       
        distance_image = depth_image * depth_scale
       
        color_image = np.asanyarray(color_frame.get_data())

        self.out.write(color_image)
        cv2.imwrite(str(ind)+".tiff",depth_image)

class CameraStreamer(Thread):
    depthShape = [1280,720]
    rgbShape=[1280,720]
    def __init__(self,frameQueue):
        Thread.__init__(self)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.rgbShape[0], self.rgbShape[1], rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, self.depthShape[0], self.depthShape[1], rs.format.z16, 30)
        self.frameQueue=frameQueue
        
    def run(self):
        self.profile = self.pipeline.start(self.config)
        self.loopFrames()
        
    def loopFrames(self):
        global ind
        t1=time.perf_counter()
        while True:
            ind += 1
            frameset = self.pipeline.wait_for_frames()
            self.frameQueue.put(frameset)
            print(f"Looper Thread | Acquired frame {frameset.frame_number} in {np.round((time.perf_counter()-t1),3)}")
            t1=time.perf_counter()

ind = 0
frameQueue=Queue()

#start threads
cameraStreamer=CameraStreamer(frameQueue)
cameraProcess = CameraProcess(frameQueue)

#start threads
cameraProcess.start()
cameraStreamer.start()

nframes=0
iniTime=time.perf_counter()
t1=iniTime
while True:
    fs = cameraProcess.getProcessedFrame()
    nframes+=1
    print(f"Time since last frame {np.round((time.perf_counter()-t1)*1000,3)}ms, qsize {cameraProcess.frameQueue.qsize()},fps: {np.round(nframes/(time.perf_counter()-iniTime),3)}")
    t1=time.perf_counter()
c.pipeline.stop()