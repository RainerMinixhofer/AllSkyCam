# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 21:24:08 2019

@author: Rainer
"""

# pylint: disable=C0103
import threading
import argparse
import datetime
import time
import numpy as np
from flask import Response
from flask import Flask
from flask import render_template, request
import cv2 # pylint: disable=E0401
import zwoasi as asi # pylint: disable=E0401

bayerpatt = ['RGGB', 'BGGR', 'GRBG', 'GBRG'] # Sequence of Bayer pattern in rows then columns
imgformat = ['RAW8', 'RGB24', 'RAW16', 'Y8'] # Supported image formats
asi.init('/usr/lib/libASICamera2.so')

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()

# initialize the video stream and allow the camera sensor to
# warmup
num_cameras = asi.get_num_cameras()

if num_cameras != 0:
    print("# cameras found: %d" % num_cameras)
    camera_id = 0
    print('Selected camera: %s' % asi.list_cameras()[camera_id])
else:
    raise Exception('Could not find any cameras, exiting...')
camera = asi.Camera(camera_id)

camera_info = camera.get_camera_property()
print('Camera Properties:')
for k, v in camera_info.items():
    if isinstance(v, list):
        print('{:<28}:'.format(k), '[%s]' % ', '.join(map(str, v)))
        if k == 'SupportedVideoFormat':
            print('{:<28}:'.format(k + ' decoded'), '[%s]' % ', '.\
                  join(map(lambda idx: str(imgformat[idx]), v)))
    else:
        print('{:<28}: {:<10}'.format(k, v))
        if k == 'BayerPattern':
            bayerindx = v
            print('{:<28}: {:<10}'.format(k + ' decoded', bayerpatt[bayerindx]))
print('Camera temperature          : {:4.1f}°C'.\
      format(camera.get_control_value(asi.ASI_TEMPERATURE)[0]/10))

maxheight = camera_info['MaxHeight']
maxwidth = camera_info['MaxWidth']
controls = camera.get_controls()
print('\nCamera Controls:')
for cn in sorted(controls.keys()):
    print('\n    %s:\n' % cn)
    for k in sorted(controls[cn].keys()):
        print('        %s: %s' % (k, repr(controls[cn][k])))

# Use minimum USB bandwidth permitted (40%)
camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, 40)
# Set Camera to low speed mode (14 bit ADC)
camera.set_control_value(asi.ASI_HIGH_SPEED_MODE, 0)
# Disable dark frame subtracting
camera.disable_dark_subtract()
# Set ROI, binning and image type (0:RAW8/1:RGB24/2:RAW16/3:Y8)
imgtype = 2
debayeralg = 'bl' #bilinear debayer algorithm
camera.set_roi(0, 0, maxwidth, maxheight, 1, imgtype)
camera.set_control_value(asi.ASI_EXPOSURE, 100000, auto=False) # in us
camera.set_control_value(asi.ASI_AUTO_MAX_EXP, 20000) # in ms
camera.set_control_value(asi.ASI_GAIN, 0, auto=False)
camera.set_control_value(asi.ASI_AUTO_MAX_GAIN, 250)
camera.set_control_value(asi.ASI_WB_B, 90)
camera.set_control_value(asi.ASI_WB_R, 70)
camera.set_control_value(asi.ASI_GAMMA, 50)
camera.set_control_value(asi.ASI_BRIGHTNESS, 50)
camera.set_control_value(asi.ASI_FLIP, 0) # 0:Original/1:Horizontal/2:vertical/3:both

# get parameters to decode image based on image type specification
print('Capturing ' + ('color' if camera_info['IsColorCam'] and \
                                imgtype != 3 else 'monochrome') + \
                                ' ('+imgformat[imgtype]+') images')
if imgtype == asi.ASI_IMG_RAW8: #RAW8 Uninterpolated Bayer pattern 8bit per pixel
    pixelstorage = 1
    nparraytype = 'uint8'
    channels = 1
    fontcolor = [0, 0, 255]
if imgtype == asi.ASI_IMG_RGB24: #RGB24 Interpolated 3 (RGB) channels per pixel
    pixelstorage = 3
    nparraytype = 'uint8'
    channels = 3
    fontcolor = [0, 0, 255]
if imgtype == asi.ASI_IMG_RAW16: #RAW16 Uninterpolated Bayer pattern 16bin per pixel
    pixelstorage = 2
    nparraytype = 'uint16'
    channels = 1
    fontcolor = [0, 0, 255]
if imgtype == 3: #Y8: One byte (Y) per Bayer pattern
    pixelstorage = 1
    nparraytype = 'uint8'
    channels = 1
    fontcolor = [0, 0, 255]

# get bytearray for buffer to store image
imgarray = bytearray(maxwidth*maxheight*pixelstorage)
img = np.zeros((maxheight, maxwidth, 3), nparraytype)
dodebayer = imgtype in (asi.ASI_IMG_RAW8, asi.ASI_IMG_RAW16) and debayeralg != 'none'

# If aperture should be masked, apply circular masking
print("Masking aperture")
#Define mask image of same size as image
mask = np.zeros((maxheight, maxwidth, 3 if dodebayer else channels), dtype=nparraytype)
#Define circle with origin in center of image and radius given by the smaller side of the image
cv2.circle(mask, (maxwidth//2, maxheight//2), min([maxwidth//2, maxheight//2]), (1, 1, 1), -1)

#vs = VideoStream(src=0).start()
# start video capture
# Force any single exposure to be halted
camera.stop_video_capture()
camera.stop_exposure()
# Capture one frame into imgarray buffer
camera.start_video_capture()


def detect_motion():
    """
    grab global references to the video stream, output frame, and
    lock variables
    """
    global outputFrame, lock, img # pylint: disable=W0603

    # initialize the motion detector and the total number of frames
    # read thus far
    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        camera.capture_video_frame(buffer_=imgarray, filename=None, timeout=None)
        # Get current time
        timestamp = datetime.datetime.now()
        # convert bytearray to numpy array
        nparray = np.frombuffer(imgarray, nparraytype)
        # Debayer image in the case of RAW8 or RAW16 images
        if dodebayer:
            # reshape numpy array back to image matrix depending on image type.
            # take care that opencv channel order is B,G,R instead of R,G,B
            imgbay = nparray.reshape((maxheight, maxwidth, channels))
            cv2.cvtColor(imgbay, \
                         eval('cv2.COLOR_BAYER_'+bayerpatt[bayerindx][2:][::-1]+'2BGR'), img, 0) # pylint: disable=W0123
        else:
            # reshape numpy array back to image matrix depending on image type
            img = nparray.reshape((maxheight, maxwidth, channels))
        # postprocess image
        # Apply circular masking operation
        img *= mask

        # Resize frame for better display on webpage
        scale_percent = 40 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        # draw the current timestamp on the frame
        cv2.putText(resized, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, resized.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
		# acquire the lock, set the output frame, and release the
		# lock
        with lock:
            outputFrame = resized

def generate():
    """
    grab global references to the output frame and lock variables
    """
    global outputFrame, lock # pylint: disable=W0603

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')

def thread_webAPP():
    """
    initialize a flask object
    """
    app = Flask(__name__)

    @app.route("/")
    def index(): # pylint: disable=W0612
        """
        return the rendered template
        """
        return render_template("index.html")

    @app.route("/update", methods=["POST"])
    def update(): # pylint: disable=W0612
        data = request.get_json(silent=True)
        print(data)
        camera.stop_video_capture()
        camera.stop_exposure()
        if data['exposureunit_s']:
            expmult = 100000
        elif data['exposureunit_ms']:
            expmult = 1000
        else:
            expmult = 1
        camera.set_control_value(asi.ASI_EXPOSURE, data['exposure']*expmult,
                                 auto=data['autoexposure']) # in us
        camera.set_control_value(asi.ASI_AUTO_MAX_EXP, data['automaxexposure']) # in ms
        camera.set_control_value(asi.ASI_GAIN, data['gain'], auto=data['autogain'])
        camera.set_control_value(asi.ASI_AUTO_MAX_GAIN, data['automaxgain'])
        camera.set_control_value(asi.ASI_WB_B, 90)
        camera.set_control_value(asi.ASI_WB_R, 70)
        camera.set_control_value(asi.ASI_GAMMA, 50)
        camera.set_control_value(asi.ASI_BRIGHTNESS, 50)
        # flp setting parameters: 0:Original/1:Horizontal/2:vertical/3:both
        camera.set_control_value(asi.ASI_FLIP, data['flip'])

        camera.start_video_capture()

        return Response(status=200)

    @app.route("/video_feed")
    def video_feed(): # pylint: disable=W0612
        """
    	return the response generated along with the specific media
    	type (mime type)
       """
        return Response(generate(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    # check to see if this is the main thread of execution
    if __name__ == '__main__':
    	# construct the argument parser and parse command line arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--ip", type=str, required=True,
                        help="ip address of the device")
        ap.add_argument("-o", "--port", type=int, required=True,
                        help="ephemeral port number of the server (1024 to 65535)")
        args = vars(ap.parse_args())

        # start a thread that will perform motion detection
        t = threading.Thread(target=detect_motion, args=())
        t.daemon = True
        t.start()

        # start the flask app
        app.run(host=args["ip"], port=args["port"], debug=True,
                threaded=True, use_reloader=False)

t_webApp = threading.Thread(name='Web App', target=thread_webAPP)
t_webApp.setDaemon(True)
t_webApp.start()

try:
    while True:
        time.sleep(1)

except (KeyboardInterrupt, SystemExit):
    # release the camera class
    camera.stop_video_capture()
    camera.stop_exposure()
    print("Camera successfully closed, exiting....")
