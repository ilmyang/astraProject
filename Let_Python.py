from openni import openni2
from openni import _openni2 as c_api
import numpy as np
import cv2

depth_width = 640
depth_height = 480

def mousecallback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(y, x, dpt[y, x])

if __name__ == "__main__":
    openni2.initialize()

    dev = openni2.Device.open_any()
    print(dev.get_device_info())

    depth_stream = dev.create_depth_stream()
    print(depth_stream.get_video_mode())
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX = depth_width, resolutionY = depth_height, fps = 30))
    depth_stream.start()
    

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('depth')
    cv2.setMouseCallback('depth', mousecallback)

    while True :
        frame = depth_stream.read_frame()

        dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([depth_height, depth_width, 2])
        dpt1 = np.asarray(dframe_data[:, :, 0],dtype='float32')
        dpt2 = np.asarray(dframe_data[:, :, 1],dtype='float32')
        
        dpt2 = dpt2*255
        dpt = dpt1 + dpt2
        dpt = dpt[:,::-1]
        cv2.imshow('depth', dpt)

        ret, frame = cap.read()
        cv2.imshow('color', frame)

        key = cv2.waitKey(1)
        if int(key) == ord('q'):
            break

cap.release()

cv2.destroyAllWindows()

depth_stream.stop()
dev.close()

