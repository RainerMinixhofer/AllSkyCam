#!/usr/bin/env python3
"""
Script for controlling the capture of images via the AllSky Camera
"""
# pylint: disable=C0103,C0301,C0302,W0621,W0702,W0123,W1401,R0902,R0903,E0237,R0914,R0912,R0915,R1705
import sys
#sys.path.insert(0, "/usr/local/python")
import argparse
import os
import time
import subprocess
import pipes
import re
import math
import datetime
import threading
import glob
from shutil import rmtree
import atexit
import psutil
import requests
import numpy as np
import metpy.calc as mcalc
from metpy.units import units
from pytz import utc # pylint: disable=E0401
import cv2 # pylint: disable=E0401
import ephem # pylint: disable=E0401
import zwoasi as asi # pylint: disable=E0401
from apscheduler.schedulers.background import BackgroundScheduler # pylint: disable=E0401
from FriendlyELEC_NanoHatMotor import FriendlyELEC_NanoHatMotor as NanoHatMotor # pylint: disable=E0401

__author__ = 'Rainer Minixhofer'
__version__ = '0.0.3'
__license__ = 'MIT'

bayerpatt = ['RGGB', 'BGGR', 'GRBG', 'GBRG'] # Sequence of Bayer pattern in rows then columns
imgformat = ['RAW8', 'RGB24', 'RAW16', 'Y8'] # Supported image formats
threadLock = threading.Lock()
threads = []
# env_filename = os.getenv('ZWO_ASI_LIB')
asi_filename = '/usr/lib/libASICamera2.so'

class IsDay():
    """
    Calculate if it is Day (Sun center above twilightalt) and save result in boolean isday
    """
    def __init__(self, position):
        self.position = position

    def __call__(self, output):
        self.position.date = datetime.datetime.utcnow()
        if output:
            print("Date and Time (UTC): %s" % self.position.date)
        sun = ephem.Sun(self.position)
        sun.compute(self.position)
        if sun.alt > args.twilightalt*math.pi/180:
            if output:
                print('We have day... (sun altitude=%s)' % sun.alt)
            return True
        else:
            if output:
                print('We have night... (sun altitude=%s)' % sun.alt)
            return False

def save_control_values(filename, settings, params):
    """
    saves control values of camera
    """
    filename += '.txt'
    with open(filename, 'w') as f:
        for k in sorted(settings.keys()):
            f.write('%s: %s\n' % (k, str(settings[k])))
        for k in sorted(params.keys()):
            f.write('%s: %s\n' % (k, str(params[k])))
    print('Camera settings saved to %s' % filename)
    #Write Camera Focus measure into Homematic
    r = requests.get("http://homematic.minixint.at/config/xmlapi/statechange.cgi?ise_id=22416&new_value="+"{:.1f}".format(params["Focus"]))
    if r.status_code != requests.codes['ok']:
        print("Data could not be written into the Homatic system variables.")


def get_image_statistics(img):
    """
    saves image statistics of img
    """
    statistics = {}
    mean, std = cv2.meanStdDev(img)
    mmean = np.amax(mean)
    mstd = np.amax(std)
    if img.dtype.name == 'uint8':
        mmean /= 255.0
    elif img.dtype.name == 'uint16':
        mmean /= 65535.0
    statistics['Mean'] = mmean
    statistics['StdDev'] = mstd
    # blur detection measure using Variance of Laplacian (see https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/#)
    statistics['Focus'] = cv2.Laplacian(img, cv2.CV_64F).var()
    return statistics

def postprocess(args):
    """
    does postprocessing of saved images and generates video, startrails and keogram
    """
    startrails = None
    keogram = None
    dovideo = not args.video == "none"
    dostart = not args.startrailsoutput == "none"
    dokeogr = not args.keogramoutput == "none"
    if dovideo or dostart or dokeogr:
        imgfiles = sorted(glob.glob(args.dirtime_12h_ago_path+"/"+args.filename[:-4]+"*."+args.extension))
        # Create array for startrail statistics
        if dostart:
            stats = np.zeros((len(imgfiles), 3))

        if dovideo:
            print("Exporting images to video")
            vidfile = args.dirtime_12h_ago_path+'/'+args.video[:-4]+args.dirtime_12h_ago+args.video[-4:]
            vid = cv2.VideoWriter(vidfile, cv2.VideoWriter_fourcc(*'mp4v'), \
                                  args.framerate, (args.width, args.height))
        for idx, imgfile in enumerate(imgfiles):
            image = cv2.imread(imgfile)
            if dovideo:
                vid.write(image)
            if dostart:
                imgstats = get_image_statistics(image)
                mean = imgstats['Mean']
                # Save mean and standard deviation in statistics array
                stats[idx] = [mean, imgstats['StdDev'], imgstats['Focus']]
                # Add image to startrails image if mean is below threshold
                if mean <= args.threshold:
                    status = 'Processing'
                    if startrails is None:
                        startrails = image.copy()
                    else:
                        startrails = cv2.max(startrails, image)
                else:
                    status = 'Skipping'
            if dokeogr:
                if keogram is None:
                    height, width, channels = image.shape
                    keogram = np.zeros((height, len(imgfiles), channels), image.dtype)
                else:
                    keogram[:, idx] = image[:, width//2]
                print('%s #%d / %d / %s / %6.4f' % (status, idx, len(imgfiles), os.path.basename(imgfile), mean))

        if dovideo:
            vid.release()
            if args.serverrepo != 'none':
                os.system("scp "+vidfile+" "+args.serverrepo+"/")
            print("Video Exported")
        if dostart:

            #Save Statistics
            np.savetxt(args.dirtime_12h_ago_path+'/imgstatistics.txt', stats.transpose(), delimiter=',', fmt='%.4f', header='1st line:mean of image, 2nd line: StdDev of image, 3rd line: Focus of image')

            #calculate some statistics for startrails: Min, Max, Mean and Median of Mean
            min_mean, max_mean, min_loc, _ = cv2.minMaxLoc(stats[:, 0])
            mean_mean = np.mean(stats[:, 0])
            median_mean = np.median(stats[:, 0])

            print("Startrails generation: Minimum: %6.4f / maximum: %6.4f / mean: %6.4f / median: %6.4f" % (min_mean, max_mean, mean_mean, median_mean))

            #If we still don't have an image (no images below threshold), copy the minimum mean image so we see why
            if startrails is None:
                print('No images below threshold, writing the minimum image only')
                startrails = cv2.imread(imgfiles[min_loc[1]], cv2.IMREAD_UNCHANGED)
            if args.extension == "png":
                args.fileoptions = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
            elif args.extension == "jpg":
                args.fileoptions = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            else: args.fileoptions = []

            startfile = args.dirtime_12h_ago_path+'/'+args.startrailsoutput[:-4]+args.dirtime_12h_ago+args.startrailsoutput[-4:]
            status = cv2.imwrite(startfile, startrails, args.fileoptions)
            if args.serverrepo != 'none':
                os.system("scp "+startfile+" "+args.serverrepo+"/")
            print("Startrail Image", startfile, " written to file-system : ", status)
        if dokeogr:
            geogrfile = args.dirtime_12h_ago_path+'/'+args.keogramoutput[:-4]+args.dirtime_12h_ago+args.keogramoutput[-4:]
            status = cv2.imwrite(geogrfile, keogram, args.fileoptions)
            if args.serverrepo != 'none':
                os.system("scp "+geogrfile+" "+args.serverrepo+"/")
            print("Image", geogrfile, " written to file-system : ", status)
        # Copy all image files over to server (only for debugging, images are kept on the camera "nightstokeep" times)
#        if args.serverrepo != 'none':
#            os.system("scp "+args.dirtime_12h_ago_path+"/"+args.filename[:-4]+"*."+args.extension+" "+args.serverrepo+"/"\
#                  +args.dirtime_12h_ago+"/")


class saveThread(threading.Thread):
    """
    thread for saving image
    """
    def __init__(self, filename, dirtime_12h_ago, img, params):
        threading.Thread.__init__(self)
        self.filename = filename
        self.dirtime_12h_ago = dirtime_12h_ago
        self.img = img
        self.params = params
    def run(self):
        print("Saving image " + self.filename)
        # Get lock to synchronize threads
        threadLock.acquire()
        cv2.imwrite(self.filename, self.img, self.params)
        # Free lock to release next thread
        threadLock.release()

class dht22Thread(threading.Thread):
    """
    thread for reading DHT22 data and timeout (given by timeout) if no readout within DHT22 read interval (given by read_interval in seconds)
    """
    def __init__(self, event, camera, read_interval=None, timeout=None):
        threading.Thread.__init__(self)
        if read_interval is None:
            self.read_interval = 5 * 60
        else:
            self.read_interval = read_interval
        if timeout is None:
            self.timeout = 5.0
        else:
            self.timeout = timeout
        self.stopped = event
        self.camera = camera
        self.dht22hum = 0
        self.dht22temp = 0
        self.temperature = self.dht22temp
        self.mixratio = 0
        self.specific_humidity = 0
        self.pressure = 0
        self.dewpoint = 0
    def run(self):
        while True:
            try:
                #Read data of DHT22 sensor
                dht22data = subprocess.run(['/home/rainer/Documents/AllSkyCam/AllSkyCapture/readdht22.sh'], stdout=subprocess.PIPE, timeout=self.timeout)
                self.dht22hum, self.dht22temp = [float(i) for i in re.split(' \= | \%| \*', dht22data.stdout.decode('ascii'))[1::2][:-1]]
                print("Output of DHT22: Temperature = ", self.dht22temp, "°C / Humidity = ", self.dht22hum, "%")
                #Write data of DHT22 sensor into Homematic
                r = requests.get("http://homematic.minixint.at/config/xmlapi/statechange.cgi?ise_id=22276,22275&new_value="+"{:.1f},{:.1f}".format(self.dht22hum, self.dht22temp))
                if r.status_code != requests.codes['ok']:
                    print("Data could not be written into the Homatic system variables.")
                #Write Camera Sensor Temperature into Homematic
                r = requests.get("http://homematic.minixint.at/config/xmlapi/statechange.cgi?ise_id=22277&new_value="+"{:.1f}".format(camera.get_control_value(asi.ASI_TEMPERATURE)[0]/10))
                if r.status_code != requests.codes['ok']:
                    print("Data could not be written into the Homatic system variables.")
                #Read Air pressure from Homematic and convert the XML result from request into float of pressure in hPa
                r = requests.get("http://homematic.minixint.at/config/xmlapi/sysvar.cgi?ise_id=20766")
                self.pressure = units.Quantity(float(re.split('\=| ', r.text)[12][1:-1]), 'hPa')
                self.temperature = units.Quantity(self.dht22temp, 'degC')
                #Calculate dewpoint from relative humidity, pressure and temperature using metpy and write it into Homematic
                self.mixratio = mcalc.mixing_ratio_from_relative_humidity(float(self.dht22hum)/100, self.temperature, self.pressure)
                self.specific_humidity = mcalc.specific_humidity_from_mixing_ratio(self.mixratio)
                self.dewpoint = mcalc.dewpoint_from_specific_humidity(self.specific_humidity, self.temperature, self.pressure).magnitude
                r = requests.get("http://homematic.minixint.at/config/xmlapi/statechange.cgi?ise_id=22278&new_value="+"{:.1f}".format(self.dewpoint))
                if r.status_code != requests.codes['ok']:
                    print("Data could not be written into the Homatic system variables.")
            except subprocess.TimeoutExpired:
                print("Waited", self.timeout, "seconds, and did not get any valid data from DHT22")
            if self.stopped.wait(self.read_interval):
                break

def lmt2utc(lmt, longitude):
    """
    :param lmt: string Ex. '2008-12-2'
    :param longitude: longitude
    :return: UTC time stamp resembling lmt
    """
    utcdt = lmt - datetime.timedelta(seconds=round(4*60*longitude*180/math.pi))
    utcdt = utcdt.replace(tzinfo=utc)
    return utcdt

def getanalemma(args, camera, imgarray, img, threads):
    """
    Captures one image of the current sun position to be assembled into one analemma
    """
    # Generate analemma subdirectory if this directory is not present and if the analemma parameter is specified
    analemmabase = args.dirname + "/analemma"
    if not os.path.isdir(analemmabase):
        try:
            os.mkdir(analemmabase)
        except OSError:
            print("Creation of the analemma subdirectory %s failed" % analemmabase)
    # Use autoexposure for analemma
    camera.set_control_value(asi.ASI_EXPOSURE, args.exposure, auto=True)
    # zero gain and no autogain for analemma
    camera.set_control_value(asi.ASI_GAIN, 0, auto=False)

    # start video capture
    try:
        # Force any single exposure to be halted
        camera.stop_video_capture()
        camera.stop_exposure()
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        pass

    camera.start_video_capture()

    # read image as bytearray from camera
    print("Starting analemma capture\n")

    if exp_ms <= 100 and lastisday:
        timeoutms = 200
    elif lastisday:
        timeoutms = exp_ms*2
    else:
        timeoutms = None
    try:
        camera.capture_video_frame(buffer_=imgarray, filename=None, timeout=timeoutms)
    except:
        print("Exposure timeout, increasing exposure time\n")

    print("Stopping Exposure")
    # read current camera parameters
    autoExp = camera.get_control_value(asi.ASI_EXPOSURE)[0] # in us
    print("Autoexposure: %d us" % autoExp)
    # Get current time
    timestring = datetime.datetime.now()
    # Define file base string (for image and image info file)
    filebase = analemmabase+'/analemma'+timestring.strftime("%Y%m%d%H%M%S")
    # convert bytearray to numpy array
    nparray = np.frombuffer(imgarray, nparraytype)

    # Debayer image in the case of RAW8 or RAW16 images
    if dodebayer:
        # reshape numpy array back to image matrix depending on image type.
        # take care that opencv channel order is B,G,R instead of R,G,B
        imgbay = nparray.reshape((args.height, args.width, channels))
        cv2.cvtColor(imgbay, eval('cv2.COLOR_BAYER_'+bayerpatt[bayerindx][2:][::-1]+\
                               '2BGR'+debayeralgext), img, 0)
    else:
        # reshape numpy array back to image matrix depending on image type
        img = nparray.reshape((args.height, args.width, channels))

    # postprocess image
    # If aperture should be masked, apply circular masking
    if args.maskaperture:
        #Do mask operation
        img *= mask

    # write image based on extension specification and data compression parameters
    if args.extension == 'jpg':
        thread = saveThread(filebase+'.jpg', args.dirtime_12h_ago, img, \
                            [int(cv2.IMWRITE_JPEG_QUALITY), args.jpgquality])
        thread.start()
    #    print('Saved to %s' % filename)
    elif args.extension == 'png':
        thread = saveThread(filebase+'.png', args.dirtime_12h_ago, img, \
                            [int(cv2.IMWRITE_PNG_COMPRESSION), args.pngcompression])
        thread.start()
    elif args.extension == 'tif':
        # Use TIFFTAG_COMPRESSION=259 to specify COMPRESSION_LZW=5
        thread = saveThread(filebase+'.tif', args.dirtime_12h_ago, img, [259, 5])
        thread.start()
    #    print('Saved to %s' % filename)
    threads.append(thread)

    camera.stop_video_capture()
    camera.stop_exposure()

def turnOffMotors():
    """
    Needed for auto disable stepper motor on shutdown
    """
    mh.getMotor(1).run(NanoHatMotor.RELEASE)
    mh.getMotor(2).run(NanoHatMotor.RELEASE)
    mh.getMotor(3).run(NanoHatMotor.RELEASE)
    mh.getMotor(4).run(NanoHatMotor.RELEASE)

bMain = True

def exists_remote(host, path):
    """
    Test if a file exists at path on a host accessible with SSH.
    """
    status = subprocess.call(['ssh', host, 'test -d {}'.format(pipes.quote(path))])
    if status == 0:
        return True
    if status == 1:
        return False
    raise Exception('SSH failed')

def check_positive_int(value):
    """
    Define parser for command lines
    """
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("%s is not an integer" % value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid negative integer" % value)
    return ivalue
def check_nonnegative_int(value):
    """
    Define parser for negative integers
    """
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("%s is not an integer" % value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid negative integer" % value)
    return ivalue
def is_number(s):
    """
    Define parser for numbers
    """
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

parser = argparse.ArgumentParser(description='Process and save images from the AllSkyCamera')

# Argument to specify location of ASI SDK Library (default specified in env_filename
parser.add_argument('--ASIlib',
                    nargs='?',
                    const=asi_filename,
                    default=asi_filename,
                    help='ASI SDK library filename')
# Argument to specify number of camera to use when more than one camera is attached
parser.add_argument('--camnr',
                    default=0,
                    type=check_nonnegative_int,
                    help='Specify number of camera to use (when multiple cameras are attached)')
# Width and Height of final image capture size (default is maximum width and height)
parser.add_argument('--width',
                    default=1808,
                    type=check_positive_int,
                    help='Width of final image capture (0 if maximum width should be selected)')
parser.add_argument('--height',
                    default=1808,
                    type=check_positive_int,
                    help='Height of final image capture (0 if maximum height should be selected)')
# Top and left position of final image capture (default is)
parser.add_argument('--left',
                    default=700,
                    type=check_nonnegative_int,
                    help='Position of leftmost pixel of final image capture (0 if leftmost image position should be selected)')
parser.add_argument('--top',
                    default=37,
                    type=check_nonnegative_int,
                    help='Position of topmost pixel of final image capture (0 if topmost image position should be selected)')
# Exposure setting
parser.add_argument('--exposure',
                    default=20000000,
                    type=check_positive_int,
                    help='Exposure time in us (default 10 sec)')
# Maximum Exposure setting
parser.add_argument('--maxexposure',
                    default=20000,
                    type=check_positive_int,
                    help='Maximum Exposure time in ms (default 20 sec)')
# Autoexposure setting
parser.add_argument('--autoexposure',
                    action='store_true',
                    help='Specify to use autoexposure')
# Gain setting
parser.add_argument('--gain',
                    default=50,
                    type=check_nonnegative_int,
                    help='Gain for night images. During the day, gain is always set to 0 (default 250)')
# Maximum Gain setting
parser.add_argument('--maxgain',
                    default=250,
                    type=check_positive_int,
                    help='Maximum gain for night images when using auto-gain (default 250)')
# Autogain setting
parser.add_argument('--autogain',
                    action='store_true',
                    help='Specify to use auto gain at night')
# Gamma setting
parser.add_argument('--gamma',
                    default=50,
                    type=check_positive_int,
                    help='Gamma (default 50)')
# Brightness setting
parser.add_argument('--brightness',
                    default=50,
                    type=check_positive_int,
                    help='Brightness (default 50)')
# White balance red setting
parser.add_argument('--wbr',
                    default=70,
                    type=check_positive_int,
                    help='White balance red (default 70)')
# White balance blue setting
parser.add_argument('--wbb',
                    default=90,
                    type=check_positive_int,
                    help='White balance blue (default 90)')
# Binning setting
parser.add_argument('--bin',
                    default=1,
                    choices=range(1, 5),
                    type=int,
                    help='Binning (1-4, default 1:no binning)')
# Delay setting
parser.add_argument('--delay',
                    default=20000,
                    type=check_positive_int,
                    help='Delay between images at night in ms (default 20000 ms = 20 s)')
# Daytime delay setting
parser.add_argument('--delayDaytime',
                    default=6000,
                    type=check_positive_int,
                    help='Delay between images in Daytime in ms (default 6000 ms = 6 s)')
# Image type setting
parser.add_argument('--type',
                    default=2,
                    choices=range(4),
                    type=int,
                    help='Image type (0:RAW8/1:RGB24/2:RAW16/3:Y8, default 2)')
# PNG Image quality setting
parser.add_argument('--pngcompression',
                    default=9,
                    type=int,
                    help='Image compression (0-9, default 9)')
# JPG Image quality setting
parser.add_argument('--jpgquality',
                    default=99,
                    type=check_nonnegative_int,
                    help='Image quality (0-100, default 99)')
# USB speed setting
parser.add_argument('--usbspeed',
                    default=40,
                    type=check_positive_int,
                    help='USB speed/BandWidthOverload (40-100, default 40)')
# Image filename setting
parser.add_argument('--filename',
                    default='image.png',
                    help='Image filename (default "image.png")')
# Video save setting
parser.add_argument('--video',
                    default='video.mp4',
                    help='Video filename and format. "none" for no video. (default "video.mp4")')
# Video frame rate setting
parser.add_argument('--framerate',
                    default=30,
                    type=check_nonnegative_int,
                    help='Video framerate (default 30)')
# Startrails save setting
parser.add_argument('--startrailsoutput',
                    default='startrails.png',
                    help='Filename of startrails image. "none" for no startrails. (default "startrails.png")')
# Startrails threshold setting
parser.add_argument('--threshold',
                    default=0.1,
                    type=float,
                    help='Brightness threshold for startrails ranges from 0 (black) to 1 (white) \
                    A moonless sky is around 0.05 while full moon can be as high as 0.4 \
                    (default 0.1)')
# Keogram save setting
parser.add_argument('--keogramoutput',
                    default='keogram.png',
                    help='Filename of keogram image. "none" for no keogram. (default "keogram.png")')
# Image flip setting
parser.add_argument('--flip',
                    default=0,
                    choices=range(4),
                    type=int,
                    help='Image flip setting \
                    (0:Original/1:Horizontal/2:vertical/3:both, default 0)')
# Position latitude setting
parser.add_argument('--lat',
                    default='46.973066N',
                    help='Position latitude (90S-90N, default 46.973N, Premstaetten)')
# Position longitude setting
parser.add_argument('--lon',
                    default='15.398836E',
                    help='Position longitude (180W-180E, default 15.399E, Premstaetten)')
# Altitude of sun for twilight setting
parser.add_argument('--twilight',
                    default='Civil',
                    help='Either twilight altitude of sun (-20-0) or predefined Twilight \
                    altitude setting (Astronomical:-18/Nautical:-12/Civil:-6, default -7). \
                    Parameter is set to end capture exactly when autogain is starting to \
                    saturate the image (based on the camera configuration with 10sec \
                    exposure).')
# Darkframe subtraction setting
parser.add_argument('--darkframe',
                    default='',
                    const='darkframe.png',
                    nargs='?',
                    help='Specify image to subtract as dark frame (default "" which means \
                    no darkframe subtraction)')
# Take Darkframe setting
parser.add_argument('--takedarkframe',
                    default='',
                    action='store_true',
                    help='Specify than image is taken as dark frame. No overlays and labels \
                    are shown. Dark frame is stored under <filename>')
# Overlay text
parser.add_argument('--text',
                    default='',
                    help='Character/Text Overlay. Use quotes e.g. "Text". Positioned at <textX>,\
                    <textY> with the properties given by the <font...> parameters. (Default "")')
# Overlay text x position
parser.add_argument('--textx',
                    default=15,
                    type=check_positive_int,
                    help='Text placement horizontal from left in pixels (Default 15)')
# Overlay text y position
parser.add_argument('--texty',
                    default=25,
                    type=check_positive_int,
                    help='Text placement vertical from top in pixels (Default 25)')
# Name of font
parser.add_argument('--fontname',
                    default=0,
                    choices=range(7),
                    help='Font type number (0-7 ex. 0:simplex/4:triplex/7:script, default 0)')
# Color of font
parser.add_argument('--fontcolor',
                    default=[1.0, 0.0, 0.0],
                    nargs=3,
                    type=float,
                    help='Font color (default [1.0,0.0,0.0]: blue)')
# Line type of font
parser.add_argument('--fontlinetype',
                    default=0,
                    choices=range(2),
                    help='Font line type (0:AA/1:8/2:4, default 0)')
# Size of font
parser.add_argument('--fontsize',
                    default=1.0,
                    type=float,
                    help='Font size (default 1.0)')
# Line type of font
parser.add_argument('--fontlinethick',
                    default=2,
                    choices=range(1, 5),
                    help='Font line thickness (1-5, default 2)')
# Time labeling setting
parser.add_argument('--time',
                    action='store_true',
                    help='Adds time info to image. Use textx and texty for placement.')
# Mask areas outside aperture circle
parser.add_argument('--maskaperture',
                    action='store_true',
                    help='Does circular aperture masking outside of circle filling ROI.')
# Daytime capture setting
parser.add_argument('--daytime',
                    action='store_true',
                    help='Stores timelapse images at daytime too.')
# Metadata labeling setting
parser.add_argument('--details',
                    action='store_true',
                    help='Show additional metadata in image')
# Debayer Algorithm (only for RAW image processing
parser.add_argument('--debayeralg',
                    default='none',
                    help='''Debayer algorithm when using image type RAW8 or RAW16 \
                    (none/bl/vng/ea, default none).
                                bl/bilinear.....................\
                                bilinear interpolation between pixels (RAW8/RAW16)
                                vng/variablenumberofgradients...\
                                variable number of gradients interpolation (only RAW8)
                                ea/edgeaware....................\
                                edge-aware interpolation (RAW8/RAW16).''')
# Daytime capture setting
parser.add_argument('--debug',
                    action='store_true',
                    help='Additional debugging information.')
# Do only postprocessing (video, startrails and keogram) and no image capture
parser.add_argument('--postprocessonly',
                    action='store_true',
                    help='Do only postprocessing (video, startrails and keogram) but no image capture.')
parser.add_argument('--serverrepo',
                    default='rainer@server03:/var/www/html/kameras/allskycam',
                    help='''Position and username of repository to store Imagery and Videos. \
                    If "none" no data will be copied over. \
                    (Default rainer@server03:/var/www/html/kameras/allskycam)''')
# Autodelete when more than specified nights
parser.add_argument('--nightstokeep',
                    default=11,
                    type=int,
                    help='Number of nights to keep before start deleting. Set to negative to disable. (Default 11)')
# Run focus scale exposure
parser.add_argument('--focusscale',
                    default='',
                    const='11,10',
                    nargs='?',
                    type=str,
                    help='If specified a focus scale is run with the number of exposures in \
                    first number of argument with varying focus in steps given by \
                    second number of argument. The arguments need to be separated by comma. \
                    (If specified without parameter 11,10 is taken)')

# Do analemma exposures
parser.add_argument('--analemma',
                    default='meanmidday',
                    const='meanmidday',
                    nargs='?',
                    type=str,
                    help='If specified, an exposure is taken at the specified time each day and saved. \
                    into the subfolder "analemma". The time of the exposure can be specified as \
                    hh:mm:ss and is taken as mean local time without daylight saving \
                    or with the predefined keywords "meanmidday", which sets the time to 12:00:00. \
                    The exposure parameters for this setting are taken from the file analemma.cfg in \
                    the same directory as this script, if this file is present. \
                    (if specified without parameter "meanmidday" is taken)')

args = parser.parse_args()

args.dirname = os.getcwd()
args.extension = args.filename.lower()[-3:]

# Do postprocessing if postprocessonly is set and exit
if args.postprocessonly:
    args.dirtime_12h_ago_path = os.getcwd()
    args.dirtime_12h_ago = os.path.basename(os.path.normpath(args.dirtime_12h_ago_path))
    postprocess(args)
    exit()
# Check validity of parameters
if args.lat[-1] not in ['N', 'S']:
    print('Latitude specification must be a degree float ending with "N" or "S"')
    exit()
if args.lon[-1] not in ['W', 'E']:
    print('Longitude specification must be a degree float ending with "W" or "E"')
    exit()
if args.lat[-1] == 'S':
    args.lat = -float(args.lat[:-2])
else:
    args.lat = float(args.lat[:-2])
if args.lon[-1] == 'W':
    args.lon = -float(args.lon[:-2])
else:
    args.lon = float(args.lon[:-2])
if not 0 <= args.pngcompression <= 9:
    print('PNG File compression setting has to be in the interval [0,9]')
    exit()
if not 0 <= args.jpgquality <= 100:
    print('JPG compression quality setting has to be in the interval [0,100]')
    exit()
if not is_number(args.twilight):
    tl = args.twilight.lower()
    if tl[:5] == 'civil':
        args.twilightalt = -6
    elif tl[:5] == 'astro':
        args.twilightalt = -18
    elif tl[:4] == 'naut':
        args.twilightalt = -12
    else:
        print('Wrong --twilight argument. Should read Civil, Astronomical or Nautical!')
        exit()
else:
    args.twilightalt = args.twilight
if args.debayeralg.lower() in ['none', 'bilinear', 'bl', 'variablenumberofgradients', \
                        'vng', 'edgeaware', 'ea']:
    if args.debayeralg.lower() in ['bl', 'bilinear']:
        debayeralgext = ''
    elif (args.debayeralg.lower() in ['vng', 'variablenumberofgradients']) and args.type == 2:
        print('debayer algorithm VNG just available for RAW8 images')
        exit()
    elif args.debayeralg.lower() in ['vng', 'variablenumberofgradients']:
        debayeralgext = '_VNG'
    elif args.debayeralg.lower() in ['ea', 'edgeaware']:
        debayeralgext = '_EA'
else:
    print('Wrong --debayeralg argument. Should read none, bilinear/bl, \
    variablenumberofgradients/vng or edgeaware/ea')
    exit()
position = ephem.Observer()
position.pressure = 0
position.lon = args.lon * math.pi / 180
position.lat = args.lat * math.pi / 180
isday = IsDay(position)

print("Position: Lat: %s / Lon: %s" % (position.lat, position.lon))

if not os.path.isfile(args.ASIlib):
    print('The filename of the SDK library "' + args.ASIlib +'" has not been found.')
    sys.exit(1)

# Initialize zwoasi with the name of the SDK library
asi.init(args.ASIlib)

num_cameras = asi.get_num_cameras()
if num_cameras == 0:
    print('No cameras found')
    sys.exit(0)

cameras_found = asi.list_cameras()  # Models names of the connected cameras

if num_cameras == 1:
    camera_id = 0
    print('Found one camera: %s' % cameras_found[camera_id])
else:
    camera_id = args.camnr
    print('Found %d cameras' % num_cameras)
    for n in range(num_cameras):
        print('    %d: %s' % (n, cameras_found[n]))
    print('Using #%d: %s' % (camera_id, cameras_found[camera_id]))

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

# Get all of the camera controls and export them to a text file
controls = camera.get_controls()
filenamecontrols = "controls.txt"
with open(filenamecontrols, 'w') as f:
    for cn in sorted(controls.keys()):
        f.write('    %s:\n' % cn)
        for k in sorted(controls[cn].keys()):
            f.write('        %s: %s\n' % (k, repr(controls[cn][k])))
print('Camera controls saved to %s' % filenamecontrols)

# Check capabilities of camera and set parameters accordingly
if not('Exposure' in controls and controls['Exposure']['IsAutoSupported']):
    args.autoexposure = False
if not('Gain' in controls and controls['Gain']['IsAutoSupported']):
    args.autogain = False

# Get temperature of camera
print('Camera temperature          : {:4.1f}°C'.\
      format(camera.get_control_value(asi.ASI_TEMPERATURE)[0]/10))

# Get maximum width and height from camera properties
maxheight = camera_info['MaxHeight']
if args.height == 0:
    args.height = maxheight
maxwidth = camera_info['MaxWidth']
if args.width == 0:
    args.width = maxwidth

# Adjust variables to chosen binning
args.height //= args.bin
args.height -= args.height % 2
args.width //= args.bin
args.width -= args.width % 8
if args.left is None:
    args.left = (maxwidth//args.bin - args.width) // 2
else:
    args.left //= args.bin
if args.top is None:
    args.top = (maxheight//args.bin - args.height) // 2
else:
    args.top //= args.bin
args.textx //= args.bin
args.texty //= args.bin
args.fontsize /= args.bin
args.fontlinethick //= args.bin

print('Selected Image dimensions: %s,%s (binning %d)' % (args.width, args.height, args.bin))
print('Selected Image left and top: %s,%s (binning %d)' % (args.left, args.top, args.bin))

# Use minimum USB bandwidth permitted
camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, args.usbspeed)

# Set some sensible defaults. They will need adjusting depending upon
# the sensitivity, lens and lighting conditions used.
#if args.darkframe :
#    filename='image_color.jpg'
#    camera.enable_dark_subtract(filename=filename)
#else :
camera.disable_dark_subtract()

# Set ROI, binning and image type
camera.set_roi(args.left, args.top, args.width, args.height, args.bin, args.type)

# get parameters to decode image based on image type specification

print('Capturing a single, ' + ('color' if camera_info['IsColorCam'] and \
                                args.type != 3 else 'monochrome') + \
                                ' ('+imgformat[args.type]+') image')
if args.type == asi.ASI_IMG_RAW8: #RAW8 Uninterpolated Bayer pattern 8bit per pixel
    pixelstorage = 1
    nparraytype = 'uint8'
    channels = 1
    args.fontcolor = round(255 * args.fontcolor[0])
if args.type == asi.ASI_IMG_RGB24: #RGB24 Interpolated 3 (RGB) channels per pixel
    pixelstorage = 3
    nparraytype = 'uint8'
    channels = 3
    args.fontcolor = [round(255 * c) for c in args.fontcolor]
if args.type == asi.ASI_IMG_RAW16: #RAW16 Uninterpolated Bayer pattern 16bin per pixel
    pixelstorage = 2
    nparraytype = 'uint16'
    channels = 1
    args.fontcolor = [round(65535 * c) for c in args.fontcolor]
if args.type == 3: #Y8: One byte (Y) per Bayer pattern
    pixelstorage = 1
    nparraytype = 'uint8'
    channels = 1
    args.fontcolor = round(255 * args.fontcolor[0])

# Check if image directory exists otherwise create it

if not os.path.isdir(args.dirname + "/images"):
    try:
        os.mkdir(args.dirname + "/images")
    except OSError:
        print("Creation of the images directory %s failed" % args.dirname + "/images")

camera.set_control_value(asi.ASI_EXPOSURE, args.exposure, auto=args.autoexposure)
camera.set_control_value(asi.ASI_AUTO_MAX_EXP, args.maxexposure)
camera.set_control_value(asi.ASI_GAIN, args.gain, auto=args.autogain)
camera.set_control_value(asi.ASI_AUTO_MAX_GAIN, args.maxgain)
camera.set_control_value(asi.ASI_WB_B, args.wbb)
camera.set_control_value(asi.ASI_WB_R, args.wbr)
camera.set_control_value(asi.ASI_GAMMA, args.gamma)
camera.set_control_value(asi.ASI_BRIGHTNESS, args.brightness)
camera.set_control_value(asi.ASI_FLIP, args.flip)

currentExposure = args.exposure
exp_ms = 0
autoGain = 0
autoExp = 0

# get bytearray for buffer to store image
imgarray = bytearray(args.width*args.height*pixelstorage)
img = np.zeros((args.height, args.width, 3), nparraytype)
dodebayer = (args.type == asi.ASI_IMG_RAW8 or args.type == asi.ASI_IMG_RAW16) and args.debayeralg != 'none'

# If aperture should be masked, apply circular masking
if args.maskaperture:
    print("Masking aperture")
    #Define mask image of same size as image
    mask = np.zeros((args.height, args.width, 3 if dodebayer else channels), dtype=nparraytype)
    #Define circle with origin in center of image and radius given by the smaller side of the image
#    cv2.circle(mask, (args.width//2, args.height//2), min([args.width//2, args.height//2]), (255, 255, 255), -1)
    cv2.circle(mask, (args.width//2, args.height//2), min([args.width//2, args.height//2]), (1, 1, 1), -1)

# If focusscale is run, initialize countdown variable for the number of images to take and initialize handler for focus stepper motor
if args.focusscale != '':
    focusframes, focusstepwidth = map(int, args.focusscale.split(','))
    focuscounter = focusframes
    print("Running focus scale with %i steps of %i stepwidth" % (focusframes, focusstepwidth))
    mh = NanoHatMotor(freq=200)
    atexit.register(turnOffMotors)
    myStepper = mh.getStepper(200, 1)      # motor port #1
    myStepper.setSpeed(5)                  # 5 RPM

else:
    focuscounter = -1

#Start Temperature and Humidity-Monitoring with DHT22 sensor. Read out every 5min (300sec) and timeout after 10sec

dht22stopFlag = threading.Event()
dht22thread = dht22Thread(dht22stopFlag, camera, 300, 10)
dht22thread.start()
threads.append(dht22thread)


if args.analemma != '':
    # initialize scheduler with UTC time
    scheduler = BackgroundScheduler(timezone=utc.zone)
    # Calculate trigger UTC time from mean local time
    if args.analemma.lower() == 'meanmidday':
        args.analemma = '12:00:00'
    analemmatrigger = lmt2utc(datetime.datetime.strptime(args.analemma, '%H:%M:%S'), position.lon)
    print('Analemma Trigger Time in UTC: %s' % analemmatrigger.time())
    scheduler.start()
    scheduler.add_job(getanalemma, trigger='cron', args=[args, camera, imgarray, img, threads], hour=analemmatrigger.time().hour, minute=analemmatrigger.time().minute, second=analemmatrigger.time().second)

#bMain = False

while bMain:

    try:
        lastisday = isday(True)
        if lastisday and not args.daytime:
            # At daytime with --daytime not specified, skip day time capture
            print("It's daytime... we're not saving images")
            time.sleep(args.delayDaytime/1000)
        else:
            # We use the date 12hrs ago to ensure that we do not have a date
            # jump during night capture, especially for high latitudes
            # where twilight can occur after midnight
            args.time_12h_ago = datetime.datetime.now()-datetime.timedelta(hours=12)
            args.dirtime_12h_ago = args.time_12h_ago.strftime("%Y%m%d")
            args.dirtime_12h_ago_path = args.dirname + "/images/" + args.dirtime_12h_ago
            if not os.path.isdir(args.dirtime_12h_ago_path):
                try:
                    os.mkdir(args.dirtime_12h_ago_path)
                except OSError:
                    print("Creation of the images subdirectory %s failed" % args.dirtime_12h_ago)

#            if not exists_remote("rainer@server03", "/var/www/html/kameras/allskycam/"+\
#                                 args.dirtime_12h_ago):
#                status = subprocess.call(['ssh', "rainer@server03", 'mkdir -p {}'.\
#                                          format(pipes.quote("/var/www/html/kameras/allskycam/"+\
#                                                             args.time_12h_ago.strftime("%Y%m%d")))])
#                if status == 1:
#                    raise Exception("Creation of the allskycam subdirectory %s failed" % \
#                                    args.dirtime_12h_ago)

            nightlist = sorted(glob.glob(args.dirname + "/images/" + ('[0-9]' * 8)))
            if args.nightstokeep > 0 and len(nightlist) > args.nightstokeep:
                for dirpath in nightlist[:-args.nightstokeep]:
                    if args.debug:
                        print('Removing Directory ' + dirpath)
                    rmtree(dirpath)
            else:
                if args.debug:
                    print('Not more than', args.nightstokeep, 'directories present. No directories removed.')

            if args.autoexposure and not lastisday: # autoexposure at night
                print("Saving auto exposed images every %d ms\n\n" % args.delay)
            elif lastisday: # at day always autoexposure
                print("Saving auto exposed images every %d ms\n\n" % args.delayDaytime)
                expms = 32
            else: # constant exposure at night
                print("Saving %d s exposure images every %d ms\n\n" % (currentExposure/1000000.0,\
                                                                        args.delay))
            print("Press Ctrl+C to stop\n\n")
            # autoexposure always at day
            camera.set_control_value(asi.ASI_EXPOSURE, currentExposure, \
                                     auto=args.autoexposure or lastisday)
            # zero gain and no autogain at day
            camera.set_control_value(asi.ASI_GAIN, args.gain if not lastisday else 0, \
                                     auto=args.autogain and not lastisday)
            if lastisday:
                print("Starting day time capture\n")
            else:
                print("Starting night time capture\n")

            # start video capture
            try:
                # Force any single exposure to be halted
                camera.stop_video_capture()
                camera.stop_exposure()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                pass

            camera.start_video_capture()

            while bMain and lastisday == isday(False):

                # read image as bytearray from camera
                print("Starting Exposure")

                if args.debug:
                    print("==>Memory before capture: %s percent. " % psutil.virtual_memory()[2])

                if exp_ms <= 100 and lastisday:
                    timeoutms = 200
                elif lastisday:
                    timeoutms = exp_ms*2
                else:
                    timeoutms = None
                try:
                    camera.capture_video_frame(buffer_=imgarray, filename=None, timeout=timeoutms)
                except:
                    print("Exposure timeout, increasing exposure time\n")

                if args.debug:
                    print("==>Memory after capture: %s percent. " % psutil.virtual_memory()[2])

                print("Stopping Exposure")
                # read current camera parameters
                autoExp = camera.get_control_value(asi.ASI_EXPOSURE)[0] # in us
                print("Autoexposure: %d us" % autoExp)
                autoGain = camera.get_control_value(asi.ASI_GAIN)[0]
                # Get current time
                timestring = datetime.datetime.now()
                # Define file base string (for image and image info file)
                filebase = args.dirtime_12h_ago_path+'/'+args.filename[:-4]+timestring.strftime("%Y%m%d%H%M%S")
                # convert bytearray to numpy array
                nparray = np.frombuffer(imgarray, nparraytype)

                if args.debug:
                    print("==>Memory after nparray assignment: %s percent. " % psutil.virtual_memory()[2])

                # Debayer image in the case of RAW8 or RAW16 images
                if dodebayer:
                    # reshape numpy array back to image matrix depending on image type.
                    # take care that opencv channel order is B,G,R instead of R,G,B
                    imgbay = nparray.reshape((args.height, args.width, channels))
                    cv2.cvtColor(imgbay, eval('cv2.COLOR_BAYER_'+bayerpatt[bayerindx][2:][::-1]+\
                                           '2BGR'+debayeralgext), img, 0)
                else:
                    # reshape numpy array back to image matrix depending on image type
                    img = nparray.reshape((args.height, args.width, channels))

                    if args.debug:
                        print("==>Memory after array reshaping: %s percent. " % psutil.virtual_memory()[2])

                if args.debug:
                    print("==>Memory after debayering: %s percent. " % psutil.virtual_memory()[2])

                # postprocess image
                # If aperture should be masked, apply circular masking
                if args.maskaperture:
                    #Do mask operation
                    if args.debug:
                        print("==>Memory before masking: %s percent. " % psutil.virtual_memory()[2])

                    img *= mask

                    if args.debug:
                        print("==>Memory after masking: %s percent. " % psutil.virtual_memory()[2])

                # save control values of camera and
		        # do some simple statistics on image and save to associated text file with the camera settings
                save_control_values(filebase, camera.get_control_values(), get_image_statistics(img))
                # If time parameter is specified, print timestring
                if args.time:
                    args.text = timestring.strftime("%d.%b.%Y %X")
                print('Caption: %s'%args.text)
                if args.debug:
                    print('Takedarkframe:', args.takedarkframe == '')
                if args.takedarkframe == '':
                    if args.debug:
                        print('Writing image caption')
                    cv2.putText(img, str(args.text), (args.textx, args.texty), args.fontname, \
                                args.fontsize, args.fontcolor, args.fontlinethick, \
                                lineType=args.fontlinetype)
                    if args.details:
                        line = str('Sensor {:.1f}degC'.\
                                 format(camera.get_control_value(asi.ASI_TEMPERATURE)[0]/10))
                        cv2.putText(img, line, (args.textx, int(args.texty+30/args.bin)), \
                                    args.fontname, args.fontsize*0.8, args.fontcolor, \
                                    args.fontlinethick, lineType=args.fontlinetype)
                        if lastisday and autoExp < 1000000:
                            line = 'Exposure {:.3f} ms'.format(autoExp/1000)
                        else:
                            line = 'Exposure {:.3f} s'.format(autoExp/1000000)
                        cv2.putText(img, line, (args.textx, int(args.texty+60/args.bin)), \
                                    args.fontname, args.fontsize*0.8, args.fontcolor, \
                                    args.fontlinethick, lineType=args.fontlinetype)
                        line = str('Gain {:d}'.format(autoGain))
                        cv2.putText(img, line, (args.textx, int(args.texty+90/args.bin)), \
                                    args.fontname, args.fontsize*0.8, args.fontcolor,  \
                                    args.fontlinethick, lineType=args.fontlinetype)
                        line = str('Housing Temp. {:.1f}degC'.\
                                 format(dht22thread.dht22temp))
                        cv2.putText(img, line, (args.textx, int(args.texty+120/args.bin)), \
                                    args.fontname, args.fontsize*0.8, args.fontcolor, \
                                    args.fontlinethick, lineType=args.fontlinetype)
                        line = str('Housing Hum. {:.1f}%'.\
                                 format(dht22thread.dht22hum))
                        cv2.putText(img, line, (args.textx, int(args.texty+150/args.bin)), \
                                    args.fontname, args.fontsize*0.8, args.fontcolor, \
                                    args.fontlinethick, lineType=args.fontlinetype)
                        line = str('Dew Point {:.1f}degC'.\
                                 format(dht22thread.dewpoint))
                        cv2.putText(img, line, (args.textx, int(args.texty+180/args.bin)), \
                                    args.fontname, args.fontsize*0.8, args.fontcolor, \
                                    args.fontlinethick, lineType=args.fontlinetype)
                # write image based on extension specification and data compression parameters
                if args.extension == 'jpg':
                    thread = saveThread(filebase+'.jpg', args.dirtime_12h_ago, img, \
                                        [int(cv2.IMWRITE_JPEG_QUALITY), args.jpgquality])
                    thread.start()
                #    print('Saved to %s' % filename)
                elif args.extension == 'png':
                    thread = saveThread(filebase+'.png', args.dirtime_12h_ago, img, \
                                        [int(cv2.IMWRITE_PNG_COMPRESSION), args.pngcompression])
                    thread.start()
                elif args.extension == 'tif':
                    # Use TIFFTAG_COMPRESSION=259 to specify COMPRESSION_LZW=5
                    thread = saveThread(filebase+'.tif', args.dirtime_12h_ago, img, [259, 5])
                    thread.start()
                #    print('Saved to %s' % filename)
                threads.append(thread)

                # Reduce focusscale counter if specified and move stepper motor by focusstep
                if focuscounter > 0:
                    focuscounter -= 1
                    if focusstepwidth < 0:
                        print("Moving focus motor backward by %d (%d exposures remaining)." % (-focusstepwidth, focuscounter))
                        myStepper.step(-focusstepwidth, NanoHatMotor.BACKWARD, NanoHatMotor.INTERLEAVE)
                    elif focusstepwidth > 0:
                        print("Moving focus motor forward by %d (%d exposures remaining)." % (focusstepwidth, focuscounter))
                        myStepper.step(+focusstepwidth, NanoHatMotor.FORWARD, NanoHatMotor.INTERLEAVE)
                elif focuscounter == 0:
                    bMain = False
                    if focusstepwidth < 0:
                        print("Moving focus motor back to original position by %d (%d exposures done)." % (-focusstepwidth*focusframes, focusframes))
                        myStepper.step(-focusstepwidth*focusframes, NanoHatMotor.FORWARD, NanoHatMotor.INTERLEAVE)
                    elif focusstepwidth > 0:
                        print("Moving focus motor back to original position by %d (%d exposures done)." % (+focusstepwidth*focusframes, focusframes))
                        myStepper.step(+focusstepwidth*focusframes, NanoHatMotor.BACKWARD, NanoHatMotor.INTERLEAVE)

                if args.debug:
                    print("==>Current memory utilization is %s percent. " % psutil.virtual_memory()[2])

                if (args.autogain and not lastisday):
                    print("Auto Gain value: %d\n" % autoGain)
                if (args.autoexposure and not lastisday):
                    print("Auto Exposure value: %d ms\n" % round(autoExp/1000))

                    # Apply delay before next exposure
                    if autoExp < args.maxexposure*1000:
                        # if using auto-exposure and the actual exposure is less than the max,
                        # we still wait until we reach maxesposure.
                        # This is important for a constant frame rate during timelapse generation
#                        tsleep=(args.maxexposure - autoExp/1000.0 + args.delay) # in ms
                        tsleep = (args.maxexposure + args.delay) # in ms
                        print("Sleeping %d ms\n" % tsleep)
                    else:
                        tsleep = args.delay # in ms
                    time.sleep(tsleep/1000.0)
                    print("Stop sleeping")
                else:
                    time.sleep(args.delayDaytime/1000.0 if lastisday else args.delay/1000.0)

                if not(lastisday) and isday(False):
                    # Do postprocessing (write video, do startrails and keogram images)
                    postprocess(args)

            camera.stop_video_capture()
            camera.stop_exposure()
    except KeyboardInterrupt:
        bMain = False
camera.close()
img = None
imgarray = None
imgbay = None
dht22stopFlag.set()

# Finally wait for all threads to complete
scheduler.shutdown()
for t in threads:
    t.join()
print("Exiting Capture")
