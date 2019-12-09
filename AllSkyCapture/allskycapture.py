#!/usr/bin/env python3
"""
Script for controlling the capture of images via the AllSky Camera

Created on Wed Oct 31 07:28:18 2018

@author: Rainer Minixhofer
"""
# pylint: disable=C0301,C0302,C0103,R0914,R0912,R0915,R0913,R0902,R0903,W0123,W0702,W0621,W1401
import sys
import argparse
import os
import logging
import time
import subprocess
import pipes
import re
import math
import datetime
import threading
import glob
import copy
from shutil import rmtree
import atexit
import psutil
import requests
import numpy as np
import metpy.calc as mcalc
from metpy.units import units
from pytz import utc # pylint: disable=E0401
from astropy.io import fits # pylint: disable=E0401
import cv2 # pylint: disable=E0401
import ephem # pylint: disable=E0401
import smbus # pylint: disable=E0401
import zwoasi as asi # pylint: disable=E0401
from apscheduler.schedulers.background import BackgroundScheduler # pylint: disable=E0401
from FriendlyELEC_NanoHatMotor import FriendlyELEC_NanoHatMotor as NanoHatMotor # pylint: disable=E0401

__author__ = 'Rainer Minixhofer'
__version__ = '0.0.4'
__license__ = 'MIT'

bayerpatt = ['RGGB', 'BGGR', 'GRBG', 'GBRG'] # Sequence of Bayer pattern in rows then columns
imgformat = ['RAW8', 'RGB24', 'RAW16', 'Y8'] # Supported image formats
threadLock = threading.Lock()
threads = []

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
        result = (sun.alt > args.twilightalt*math.pi/180)
        if output:
            if result:
                print('We have day... (sun altitude=%s)' % sun.alt)
            else:
                print('We have night... (sun altitude=%s)' % sun.alt)
        return result

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
    # we take log10 of measure since the spread of the variance can be huge
    statistics['Focus'] = math.log10(cv2.Laplacian(img, cv2.CV_64F).var())
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
            try:
                image = cv2.imread(imgfile)
                # Sometimes there is no error thrown by imread, but the img is still corrupted. This then
                # shows up when taking statistics
                imgstats = get_image_statistics(image)
            except:
                print("Error in readin image %s. Skipping to next file" % imgfile)
                continue
            if dovideo:
                vid.write(image)
            if dostart:
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
                if '@' in args.serverrepo:
                    os.system('scp '+vidfile+' '+args.serverrepo+'/')
                else:
                    os.system('cp '+vidfile+' '+args.serverrepo+'/')
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
                if '@' in args.serverrepo:
                    os.system("scp "+startfile+" "+args.serverrepo+"/")
                else:
                    os.system('cp '+startfile+' '+args.serverrepo+'/')
            print("Startrail Image", startfile, " written to file-system : ", status)
        if dokeogr:
            geogrfile = args.dirtime_12h_ago_path+'/'+args.keogramoutput[:-4]+args.dirtime_12h_ago+args.keogramoutput[-4:]
            status = cv2.imwrite(geogrfile, keogram, args.fileoptions)
            if args.serverrepo != 'none':
                if '@' in args.serverrepo:
                    os.system("scp "+geogrfile+" "+args.serverrepo+"/")
                else:
                    os.system('cp '+geogrfile+' '+args.serverrepo+'/')
            print("Image", geogrfile, " written to file-system : ", status)
        # Copy all image files over to server (only for debugging, images are kept on the camera "nightstokeep" times)
#        if args.serverrepo != 'none':
#            if '@' in args.serverrepo:
#                os.system("scp "+args.dirtime_12h_ago_path+"/"+
#                          args.filename[:-4]+"*."+args.extension+" "+
#                          args.serverrepo+"/"+args.dirtime_12h_ago+"/")
#            else:
#                os.system("cp "+args.dirtime_12h_ago_path+"/"+
#                          args.filename[:-4]+"*."+args.extension+" "+
#                          args.serverrepo+"/"+args.dirtime_12h_ago+"/")

class saveThread(threading.Thread):
    """
    thread for saving image
    """
    def __init__(self, filename, img, params, camctrls, imgstats, timestring):
        threading.Thread.__init__(self)
        self.filename = filename
        self.img = img
        self.params = params
        self.camctrls = camctrls
        self.imgstats = imgstats
        self.timestring = timestring
    def run(self):
        print("Saving image " + self.filename)
        # Get lock to synchronize threads
        threadLock.acquire()
        cv2.imwrite(self.filename, self.img, self.params)
        if (args.metadata is not None) and ('exif' in args.metadata):
            print("Writing EXIF tags")
            exiftags = {**camctrls, **imgstats}
            # Generate dictionary of EXIF tags from camera control values and image statistics
            exiftags['ExposureTime'] = exiftags.pop('Exposure')
            exiftags['ChipTemperature'] = exiftags.pop('Temperature')
            exiftags['ChipTemperature'] /= 10
            exiftags['ImageMean'] = exiftags.pop('Mean')
            exiftags['ImageStdDev'] = exiftags.pop('StdDev')
            exiftags['Make'], exiftags['Model'] = cameras_found[camera_id].split(' ')
            exiftags['AllDates'] = timestring.strftime("%Y.%m.%d %H:%M:%S")
            exiftags['Artist'] = 'Rainer Minixhofer'
            # Change/update EXIF tags in file
            exifpars = ['/usr/bin/exiftool', '-config', '/home/rainer/.ExifTool_config', '-overwrite_original']
            for tag, value in exiftags.items():
                exifpars.append("-{}={}".format(tag, value))
            exifpars.append(self.filename)
            process = subprocess.run(exifpars, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            if args.debug:
                print('EXIFtool stdout: %s' % process.stdout)
            if args.debug:
                print('EXIFtool stderr: %s' % process.stderr)
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
                isday(True)
                print("Output of DHT22: Temperature = ", self.dht22temp, "°C / Humidity = ", self.dht22hum, "%")
                #Write data of DHT22 sensor into Homematic
                r = requests.get("http://homematic.minixint.at/config/xmlapi/statechange.cgi?ise_id=22276,22275&new_value="+"{:.1f},{:.1f}".format(self.dht22hum, self.dht22temp))
                if r.status_code != requests.codes['ok']:
                    print("Data could not be written into the Homematic system variables.")
                #Write Camera Sensor Temperature into Homematic
                r = requests.get("http://homematic.minixint.at/config/xmlapi/statechange.cgi?ise_id=22277&new_value="+"{:.1f}".format(camera.get_control_value(asi.ASI_TEMPERATURE)[0]/10))
                if r.status_code != requests.codes['ok']:
                    print("Data could not be written into the Homematic system variables.")
                #Read Air pressure from Homematic and convert the XML result from request into float of pressure in hPa
                r = requests.get("http://homematic.minixint.at/config/xmlapi/sysvar.cgi?ise_id=20766")
                self.pressure = float(re.split('\=| ', r.text)[12][1:-1])
                press = units.Quantity(self.pressure, 'hPa')
                self.temperature = units.Quantity(self.dht22temp, 'degC')
                #Calculate dewpoint from relative humidity, pressure and temperature using metpy and write it into Homematic
                self.mixratio = mcalc.mixing_ratio_from_relative_humidity(float(self.dht22hum)/100, self.temperature, press)
                self.specific_humidity = mcalc.specific_humidity_from_mixing_ratio(self.mixratio)
                self.dewpoint = mcalc.dewpoint_from_specific_humidity(self.specific_humidity, self.temperature, press).magnitude
                r = requests.get("http://homematic.minixint.at/config/xmlapi/statechange.cgi?ise_id=22278&new_value="+"{:.1f}".format(self.dewpoint))
                if r.status_code != requests.codes['ok']:
                    print("Data could not be written into the Homematic system variables.")
            except subprocess.TimeoutExpired:
                print("Waited", self.timeout, "seconds, and did not get any valid data from DHT22")
            if self.stopped.wait(self.read_interval):
                break

class WeatherThread(threading.Thread):
    """
    thread for reading Weather data from Homematic and timeout (given by timeout) if no readout within read interval (given by read_interval in seconds)
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
        self.humidity = 0
        self.intensity = 0
        self.temperature = 0
        self.mixratio = 0
        self.specific_humidity = 0
        self.pressure = 0
        self.dewpoint = 0
    def run(self):
        while True:
            try:
                #Read data of Homematic sensor
                #Read Temperature at Roof
                r = requests.get("http://homematic.minixint.at/config/xmlapi/state.cgi?datapoint_id=12378")
                if r.status_code != requests.codes['ok']:
                    print("Roof Temperature Data could not be read from the Homematic system.")
                self.temperature = float(re.split('=|/|\'', r.text)[-4])
                temp = units.Quantity(self.temperature, 'degC')
                #Read Humidity at Roof
                r = requests.get("http://homematic.minixint.at/config/xmlapi/state.cgi?datapoint_id=12380")
                if r.status_code != requests.codes['ok']:
                    print("Roof Humidity Data could not be read from the Homematic system.")
                self.humidity = float(re.split('=|/|\'', r.text)[-4])
                #Read Light Intensity at Roof
                r = requests.get("http://homematic.minixint.at/config/xmlapi/state.cgi?datapoint_id=12382")
                if r.status_code != requests.codes['ok']:
                    print("Roof Light Intensity Data could not be read from the Homematic system.")
                self.intensity = float(re.split('=|/|\'', r.text)[-4])
                isday(True)
                print("Conditions at Roof: Temperature = ", self.temperature, "°C / Humidity = ", self.humidity, "% / Light Intensity = ", self.intensity)
                #Read Air pressure from Homematic and convert the XML result from request into float of pressure in hPa
                r = requests.get("http://homematic.minixint.at/config/xmlapi/sysvar.cgi?ise_id=20766")
                self.pressure = float(re.split('\=| ', r.text)[12][1:-1])
                press = units.Quantity(self.pressure, 'hPa')
                #Calculate dewpoint from relative humidity, pressure and temperature using metpy and write it into Homematic
                self.mixratio = mcalc.mixing_ratio_from_relative_humidity(float(self.humidity)/100, temp, press)
                self.specific_humidity = mcalc.specific_humidity_from_mixing_ratio(self.mixratio)
                self.dewpoint = mcalc.dewpoint_from_specific_humidity(self.specific_humidity, temp, press).magnitude
                r = requests.get("http://homematic.minixint.at/config/xmlapi/statechange.cgi?ise_id=24771&new_value="+"{:.1f}".format(self.dewpoint))
                if r.status_code != requests.codes['ok']:
                    print("External Dew Point Data could not be written into the Homatic system variables.")
            except subprocess.TimeoutExpired:
                print("Waited", self.timeout, "seconds, and did not get any valid data from Homematic")
            if self.stopped.wait(self.read_interval):
                break

class IRSensor:
    """
    Class for reading the MLX90614 IR Sensor over I2C. Default I2C address is 0x5a
    """

    # pylint: disable=C0326
    # RAM Registers
    __RAW1    = 0x04
    __RAW2    = 0x05
    __TA      = 0x06
    __TO1     = 0x07
    __TO2     = 0x08
    # EEPROM Registers (address of EEPROM + 2x20 see Section 8.4.5 of MLX90614 Datasheet)
    __TOMAX   = 0x00 + 0x20
    __TOMIN   = 0x01 + 0x20
    __PWMCTRL = 0x02 + 0x20
    __TARANGE = 0x03 + 0x20
    __EMISSIV = 0x04 + 0x20
    __CONFIG1 = 0x05 + 0x20
    __KE      = 0x0F + 0x20
    __CONFIG2 = 0x19 + 0x20
    # pylint: enable=C0326

    @staticmethod
    def getI2CBusNumber():
        """
        Gets the I2C bus number /dev/i2c-#
        """
        return 2

    def __init__(self, address=0x5a, debug=True):
        self.address = address
        self.debug = debug
        self.bus = smbus.SMBus(self.getI2CBusNumber())
        if debug:
            print("I2C Bus Number: %d" % self.getI2CBusNumber())

    def errMsg(self, error):
        """
        returns I2C errors
        """
        print("Error accessing 0x%02X(%X): Check your I2C address" % (self.address, error))
        return -1

    # Only write16, readU16 and readS16 commands
    # are supported. (see MLX90614 family, section 8.4.2)
    def readU16(self, reg, little_endian=True):
        "Reads an unsigned 16-bit value from the I2C device"
        try:
            result = 0xFFFA
            while result == 0xFFFA:
                result = self.bus.read_word_data(self.address, reg) & 0xFFFF
            # Swap bytes if using big endian because read_word_data assumes little
            # endian on ARM (little endian) systems.
            if not little_endian:
                result = ((result << 8) & 0xFF00) + (result >> 8)
            if self.debug:
                print("I2C: Device 0x%02X returned 0x%04X from reg 0x%02X" % (self.address, result & 0xFFFF, reg))
            return result
        except IOError as err:
            return self.errMsg(err)

    def readS16(self, reg, little_endian=True):
        "Reads a signed 16-bit value from the I2C device"
        try:
            result = self.readU16(reg, little_endian)
            if result > 32767:
                result -= 65536
            return result
        except IOError as err:
            return self.errMsg(err)

    def write16(self, reg, value):
        "Writes a 16-bit value to the specified register/address pair"
        try:
            self.bus.write_word_data(self.address, reg, value)
            if self.debug:
                print(("I2C: Wrote 0x%02X to register pair 0x%02X,0x%02X" %
                       (value, reg, reg+1)))
            return None
        except IOError as err:
            return self.errMsg(err)

    def Ta(self, imperial=False):
        """
        Temperatur Factor set to0.02 degrees per LSB
        (measurement resolution of the MLX90614)
        """
        tempFactor = 0.02
        tempData = self.readU16(self.__TA)
        tempData = (tempData * tempFactor)-0.01
        if imperial:
            #Ambient Temperature in Fahrenheit
            return ((tempData - 273.15)*1.8) + 32
        #Ambient Temperature in Celsius
        return tempData - 273.15

    def Tobj(self, imperial=False):
        """
        Temperatur Factor set to 0.02 degrees per LSB
        (measurement resolution of the MLX90614)
        """
        tempFactor = 0.02
        tempData = self.readU16(self.__TO1)
        tempData = (tempData * tempFactor)-0.01
        if imperial:
            #Ambient Temperature in Fahrenheit
            return ((tempData - 273.15)*1.8) + 32
        #Ambient Temperature in Celsius
        return tempData - 273.15

class IRSensorThread(threading.Thread):
    """
    thread for reading MLX90614 IR Sensor
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
        self.Ta = 0
        self.Tobj = 0
        self.IRsensor = IRSensor(debug=False)
    def run(self):
        while True:
            try:
                #Read data of MLX90614 IR sensor
                self.Ta = self.IRsensor.Ta()
                self.Tobj = self.IRsensor.Tobj()
                isday(True)
                print("Output of MLX90614: TAmbient = ", self.Ta, "°C / TSky = ", self.Tobj, "°C")
                #Write data of MLX90614 IR sensor into Homematic
                r = requests.get("http://homematic.minixint.at/config/xmlapi/statechange.cgi?ise_id=24737,24738&new_value="+"{:.1f},{:.1f}".format(self.Ta, self.Tobj))
                if r.status_code != requests.codes['ok']:
                    print("Data could not be written into the Homatic system variables.")
            except subprocess.TimeoutExpired:
                print("Waited", self.timeout, "seconds, and did not get any valid data from MLX90614")
            if self.stopped.wait(self.read_interval):
                break

def switchpin(pin, state=True):
    """
    Switches state of <pin> to <state>
    """
    base = "/sys/class/gpio/gpio"+str(pin)
    if state:
        if not os.path.isdir(base):
            f = open("/sys/class/gpio/export", "w")
            f.write(str(pin))
            f.close()
        # give kernel time to create control files
        time.sleep(0.1)
        f = open(base + "/direction", "w")
        f.write("out")
        f.close()
        f = open(base + "/value", "w")
        f.write("1")
        f.close()
    else:
        if os.path.isdir(base):
            f = open(base + "/value", "w")
            f.write("0")
            f.close()
            f = open("/sys/class/gpio/unexport", "w")
            f.write(str(pin))
            f.close()

def pinstatus(pin):
    """
    Returns True if Pin is on and False if Pin is off
    """

    base = "/sys/class/gpio/gpio"+str(pin)
    if not os.path.isdir(base):
        return False
    f = open(base + "/value", "r")
    data = f.read()
    f.close()
    return data == "1\n"


def pinisinput(pin):
    """
    Returns True if Pin is input and False if pin is output
    """
    base = "/sys/class/gpio/gpio"+str(pin)
    if not os.path.isdir(base):
        return True
    f = open(base + "/direction", "r")
    data = f.read()
    f.close()
    return data == "in\n"

def turnonCamera():
    """
    Switches 5V USB Power of Camera on
    """
    switchpin(35, state=True)
    time.sleep(2) # Wait for 2 seconds to ensure that camera has fully powered up

def turnoffCamera():
    """
    Switches 5V USB Power of Camera off
    """
    switchpin(35, state=False)

def cameraon():
    """
    Returns True if camera is on and False if it is off
    """
    return not pinisinput(35) and pinstatus(35)

def turnonHeater():
    """
    Switches 12V Power of Dew-Heater on
    """
    switchpin(33, state=True)

def turnoffHeater():
    """
    Switches 12V Power of Dew-Heater off
    """
    switchpin(33, state=False)

def heateron():
    """
    Returns True if dew heater is on and False if it is off
    """
    return not pinisinput(33) and pinstatus(33)

def heaterControl(tambient, tdewpoint, sensitivity=1.5, hysteresis=1.0):
    """
    Switches Dew Heater on and off depending on difference between the ambient
    temperature <tambient> and the dew point <tdewpoint>. The default difference
    is given by sensitivity. Thus the heater switches on at
    tambient - tdewpoint < sensitivity
    and off at
    tambient - tdewpoint > sensitivity + hysteresis

    Parameters
    ----------
    sensitivity : TYPE, optional
        Difference between ambient temperature and dew point
        when the heater turns on. The default is 1.0 degC.
    hysteresis : TYPE, optional
        Hyseresis of two point control of heater. The default is 0.2 degC.

    Returns
    -------
    None.

    """

    if not heateron() and tambient - tdewpoint < sensitivity:
        turnonHeater()
        #Write data of MLX90614 IR sensor into Homematic
        r = requests.get("http://homematic.minixint.at/config/xmlapi/statechange.cgi?ise_id=24816&new_value=1")
        if r.status_code != requests.codes['ok']:
            print("Data could not be written into the Homatic system variable.")
        print("\n\nDew Heater turned on(Ta=%f,Td=%f)\n\n" % (tambient,tdewpoint))
    elif heateron() and tambient - tdewpoint > sensitivity + hysteresis:
        turnoffHeater()
        r = requests.get("http://homematic.minixint.at/config/xmlapi/statechange.cgi?ise_id=24816&new_value=0")
        if r.status_code != requests.codes['ok']:
            print("Data could not be written into the Homatic system variable.")
        print("\n\nDew Heater turned off(Ta=%f,Td=%f)\n\n" % (tambient,tdewpoint))

class INA219:
    """Class containing the INA219 functionality."""

    RANGE_16V = 0  # Range 0-16 volts
    RANGE_32V = 1  # Range 0-32 volts

    GAIN_1_40MV = 0  # Maximum shunt voltage 40mV
    GAIN_2_80MV = 1  # Maximum shunt voltage 80mV
    GAIN_4_160MV = 2  # Maximum shunt voltage 160mV
    GAIN_8_320MV = 3  # Maximum shunt voltage 320mV
    GAIN_AUTO = -1  # Determine gain automatically

    ADC_9BIT = 0  # 9-bit conversion time  84us.
    ADC_10BIT = 1  # 10-bit conversion time 148us.
    ADC_11BIT = 2  # 11-bit conversion time 2766us.
    ADC_12BIT = 3  # 12-bit conversion time 532us.
    ADC_2SAMP = 9  # 2 samples at 12-bit, conversion time 1.06ms.
    ADC_4SAMP = 10  # 4 samples at 12-bit, conversion time 2.13ms.
    ADC_8SAMP = 11  # 8 samples at 12-bit, conversion time 4.26ms.
    ADC_16SAMP = 12  # 16 samples at 12-bit,conversion time 8.51ms
    ADC_32SAMP = 13  # 32 samples at 12-bit, conversion time 17.02ms.
    ADC_64SAMP = 14  # 64 samples at 12-bit, conversion time 34.05ms.
    ADC_128SAMP = 15  # 128 samples at 12-bit, conversion time 68.10ms.

    __ADDRESS = 0x40

    __REG_CONFIG = 0x00
    __REG_SHUNTVOLTAGE = 0x01
    __REG_BUSVOLTAGE = 0x02
    __REG_POWER = 0x03
    __REG_CURRENT = 0x04
    __REG_CALIBRATION = 0x05

    __RST = 15
    __BRNG = 13
    __PG1 = 12
    __PG0 = 11
    __BADC4 = 10
    __BADC3 = 9
    __BADC2 = 8
    __BADC1 = 7
    __SADC4 = 6
    __SADC3 = 5
    __SADC2 = 4
    __SADC1 = 3
    __MODE3 = 2
    __MODE2 = 1
    __MODE1 = 0

    __OVF = 1
    __CNVR = 2

    __BUS_RANGE = [16, 32]
    __GAIN_VOLTS = [0.04, 0.08, 0.16, 0.32]

    __CONT_SH_BUS = 7

    __AMP_ERR_MSG = ('Expected current %.3fA is greater '
                     'than max possible current %.3fA')
    __RNG_ERR_MSG = ('Expected amps %.2fA, out of range, use a lower '
                     'value shunt resistor')
    __VOLT_ERR_MSG = ('Invalid voltage range, must be one of: '
                      'RANGE_16V, RANGE_32V')

    __LOG_FORMAT = '%(asctime)s - %(levelname)s - INA219 %(message)s'
    __LOG_MSG_1 = ('shunt ohms: %.3f, bus max volts: %d, '
                   'shunt volts max: %.2f%s, '
                   'bus ADC: %d, shunt ADC: %d')
    __LOG_MSG_2 = ('calibrate called with: bus max volts: %dV, '
                   'max shunt volts: %.2fV%s')
    __LOG_MSG_3 = ('Current overflow detected - '
                   'attempting to increase gain')

    __SHUNT_MILLIVOLTS_LSB = 0.01  # 10uV
    __BUS_MILLIVOLTS_LSB = 4  # 4mV
    __CALIBRATION_FACTOR = 0.04096
    __MAX_CALIBRATION_VALUE = 0xFFFE  # Max value supported (65534 decimal)
    # In the spec (p17) the current LSB factor for the minimum LSB is
    # documented as 32767, but a larger value (100.1% of 32767) is used
    # to guarantee that current overflow can always be detected.
    __CURRENT_LSB_FACTOR = 32800

    @staticmethod
    def _geti2cbusnumber():
        # Gets the I2C bus number /dev/i2c-#
        return 2

    def __init__(self, shunt_ohms, max_expected_amps=None,
                 address=__ADDRESS,
                 log_level=logging.ERROR, debug=True):
        """Construct the class.
        Pass in the resistance of the shunt resistor and the maximum expected
        current flowing through it in your system.
        Arguments:
        shunt_ohms -- value of shunt resistor in Ohms (mandatory).
        max_expected_amps -- the maximum expected current in Amps (optional).
        address -- the I2C address of the INA219, defaults
            to *0x40* (optional).
        log_level -- set to logging.DEBUG to see detailed calibration
            calculations (optional).
        """
        if not logging.getLogger().handlers:
            # Initialize the root logger only if it hasn't been done yet by a
            # parent module.
            logging.basicConfig(level=log_level, format=self.__LOG_FORMAT)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.address = address
        self.debug = debug
        if debug:
            print("I2C Bus Number: %d" % self._geti2cbusnumber())
        self.bus = smbus.SMBus(self._geti2cbusnumber())
        self._shunt_ohms = shunt_ohms
        self._max_expected_amps = max_expected_amps
        self._min_device_current_lsb = self._calculate_min_current_lsb()
        self._gain = None
        self._auto_gain_enabled = False
        self._voltage_range = None
        self._current_lsb = None
        self._power_lsb = None

    def errmsg(self, error):
        """
        prints low level error message if I2C address cannot be accessed
        """
        print("Error accessing 0x%02X(%X): Check your I2C address" % (self.address, error))
        return -1

    # Only writelist, readu16 and reads16 commands
    # are needed
    def readu16(self, reg, little_endian=True):
        "Reads an unsigned 16-bit value from the I2C device"
        try:
            result = 0xFFFA
            while result == 0xFFFA:
                result = self.bus.read_word_data(self.address, reg) & 0xFFFF
            # Swap bytes if using big endian because read_word_data assumes little
            # endian on ARM (little endian) systems.
            if not little_endian:
                result = ((result << 8) & 0xFF00) + (result >> 8)
            if self.debug:
                print("I2C: Device 0x%02X returned 0x%04X from reg 0x%02X" %
                      (self.address, result & 0xFFFF, reg))
            return result
        except IOError as err:
            return self.errmsg(err)

    def reads16(self, reg, little_endian=True):
        "Reads a signed 16-bit value from the I2C device"
        try:
            result = self.readu16(reg, little_endian)
            if result > 32767:
                result -= 65536
            return result
        except IOError as err:
            return self.errmsg(err)

    def writelist(self, register, data):
        """Write bytes to the specified register."""
        self.bus.write_i2c_block_data(self.address, register, data)
        self.logger.debug("Wrote to register 0x%02X: %s",
                          register, data)

    def configure(self, voltage_range=RANGE_32V, gain=GAIN_AUTO,
                  bus_adc=ADC_12BIT, shunt_adc=ADC_12BIT):
        """Configure and calibrate how the INA219 will take measurements.
        Arguments:
        voltage_range -- The full scale voltage range, this is either 16V
            or 32V represented by one of the following constants;
            RANGE_16V, RANGE_32V (default).
        gain -- The gain which controls the maximum range of the shunt
            voltage represented by one of the following constants;
            GAIN_1_40MV, GAIN_2_80MV, GAIN_4_160MV,
            GAIN_8_320MV, GAIN_AUTO (default).
        bus_adc -- The bus ADC resolution (9, 10, 11, or 12-bit) or
            set the number of samples used when averaging results
            represent by one of the following constants; ADC_9BIT,
            ADC_10BIT, ADC_11BIT, ADC_12BIT (default),
            ADC_2SAMP, ADC_4SAMP, ADC_8SAMP, ADC_16SAMP,
            ADC_32SAMP, ADC_64SAMP, ADC_128SAMP
        shunt_adc -- The shunt ADC resolution (9, 10, 11, or 12-bit) or
            set the number of samples used when averaging results
            represent by one of the following constants; ADC_9BIT,
            ADC_10BIT, ADC_11BIT, ADC_12BIT (default),
            ADC_2SAMP, ADC_4SAMP, ADC_8SAMP, ADC_16SAMP,
            ADC_32SAMP, ADC_64SAMP, ADC_128SAMP
        """
        self.__validate_voltage_range(voltage_range)
        self._voltage_range = voltage_range

        if self._max_expected_amps is not None:
            if gain == self.GAIN_AUTO:
                self._auto_gain_enabled = True
                self._gain = self._determine_gain(self._max_expected_amps)
            else:
                self._gain = gain
        else:
            if gain != self.GAIN_AUTO:
                self._gain = gain
            else:
                self._auto_gain_enabled = True
                self._gain = self.GAIN_1_40MV

        self.logger.info('gain set to %.2fV', self.__GAIN_VOLTS[self._gain])

        self.logger.debug(
            self.__LOG_MSG_1,
            (self._shunt_ohms, self.__BUS_RANGE[voltage_range],
             self.__GAIN_VOLTS[self._gain],
             self.__max_expected_amps_to_string(self._max_expected_amps),
             bus_adc, shunt_adc))

        self._calibrate(
            self.__BUS_RANGE[voltage_range], self.__GAIN_VOLTS[self._gain],
            self._max_expected_amps)
        self._configure(voltage_range, self._gain, bus_adc, shunt_adc)

    def voltage(self):
        """Return the bus voltage in volts."""
        value = self._voltage_register()
        return float(value) * self.__BUS_MILLIVOLTS_LSB / 1000

    def supply_voltage(self):
        """Return the bus supply voltage in volts.
        This is the sum of the bus voltage and shunt voltage. A
        DeviceRangeError exception is thrown if current overflow occurs.
        """
        return self.voltage() + (float(self.shunt_voltage()) / 1000)

    def current(self):
        """Return the bus current in milliamps.
        A DeviceRangeError exception is thrown if current overflow occurs.
        """
        self._handle_current_overflow()
        return self._current_register() * self._current_lsb * 1000

    def power(self):
        """Return the bus power consumption in milliwatts.
        A DeviceRangeError exception is thrown if current overflow occurs.
        """
        self._handle_current_overflow()
        return self._power_register() * self._power_lsb * 1000

    def shunt_voltage(self):
        """Return the shunt voltage in millivolts.
        A DeviceRangeError exception is thrown if current overflow occurs.
        """
        self._handle_current_overflow()
        return self._shunt_voltage_register() * self.__SHUNT_MILLIVOLTS_LSB

    def sleep(self):
        """Put the INA219 into power down mode."""
        configuration = self._read_configuration()
        self._configuration_register(configuration & 0xFFF8)

    def wake(self):
        """Wake the INA219 from power down mode."""
        configuration = self._read_configuration()
        self._configuration_register(configuration | 0x0007)
        # 40us delay to recover from powerdown (p14 of spec)
        time.sleep(0.00004)

    def current_overflow(self):
        """Return true if the sensor has detect current overflow.
        In this case the current and power values are invalid.
        """
        return self._has_current_overflow()

    def reset(self):
        """Reset the INA219 to its default configuration."""
        self._configuration_register(1 << self.__RST)

    def _handle_current_overflow(self):
        if self._auto_gain_enabled:
            while self._has_current_overflow():
                self._increase_gain()
        else:
            if self._has_current_overflow():
                raise DeviceRangeError(self.__GAIN_VOLTS[self._gain])

    def _determine_gain(self, max_expected_amps):
        shunt_v = max_expected_amps * self._shunt_ohms
        if shunt_v > self.__GAIN_VOLTS[3]:
            raise ValueError(self.__RNG_ERR_MSG % max_expected_amps)
        gain = min(v for v in self.__GAIN_VOLTS if v > shunt_v)
        return self.__GAIN_VOLTS.index(gain)

    def _increase_gain(self):
        self.logger.info(self.__LOG_MSG_3)
        gain = self._read_gain()
        if gain < len(self.__GAIN_VOLTS) - 1:
            gain = gain + 1
            self._calibrate(self.__BUS_RANGE[self._voltage_range],
                            self.__GAIN_VOLTS[gain])
            self._configure_gain(gain)
            # 1ms delay required for new configuration to take effect,
            # otherwise invalid current/power readings can occur.
            time.sleep(0.001)
        else:
            self.logger.info('Device limit reach, gain cannot be increased')
            raise DeviceRangeError(self.__GAIN_VOLTS[gain], True)

    def _configure(self, voltage_range, gain, bus_adc, shunt_adc):
        configuration = (
            voltage_range << self.__BRNG | gain << self.__PG0 |
            bus_adc << self.__BADC1 | shunt_adc << self.__SADC1 |
            self.__CONT_SH_BUS)
        self._configuration_register(configuration)

    def _calibrate(self, bus_volts_max, shunt_volts_max,
                   max_expected_amps=None):
        self.logger.info(
            self.__LOG_MSG_2,
            (bus_volts_max, shunt_volts_max,
             self.__max_expected_amps_to_string(max_expected_amps)))

        max_possible_amps = shunt_volts_max / self._shunt_ohms

        self.logger.info("max possible current: %.3fA",
                         max_possible_amps)

        self._current_lsb = \
            self._determine_current_lsb(max_expected_amps, max_possible_amps)
        self.logger.info("current LSB: %.3e A/bit", self._current_lsb)

        self._power_lsb = self._current_lsb * 20
        self.logger.info("power LSB: %.3e W/bit", self._power_lsb)

        max_current = self._current_lsb * 32767
        self.logger.info("max current before overflow: %.4fA", max_current)

        max_shunt_voltage = max_current * self._shunt_ohms
        self.logger.info("max shunt voltage before overflow: %.4fmV",
                         (max_shunt_voltage * 1000))

        calibration = math.trunc(self.__CALIBRATION_FACTOR /
                                 (self._current_lsb * self._shunt_ohms))
        self.logger.info(
            "calibration: 0x%04x (%d)", calibration, calibration)
        self._calibration_register(calibration)

    def _determine_current_lsb(self, max_expected_amps, max_possible_amps):
        if max_expected_amps is not None:
            if max_expected_amps > round(max_possible_amps, 3):
                raise ValueError(self.__AMP_ERR_MSG %
                                 (max_expected_amps, max_possible_amps))
            self.logger.info("max expected current: %.3fA",
                             max_expected_amps)
            if max_expected_amps < max_possible_amps:
                current_lsb = max_expected_amps / self.__CURRENT_LSB_FACTOR
            else:
                current_lsb = max_possible_amps / self.__CURRENT_LSB_FACTOR
        else:
            current_lsb = max_possible_amps / self.__CURRENT_LSB_FACTOR

        if current_lsb < self._min_device_current_lsb:
            current_lsb = self._min_device_current_lsb
        return current_lsb

    def _configuration_register(self, register_value):
        self.logger.debug("configuration: 0x%04x", register_value)
        self.__write_register(self.__REG_CONFIG, register_value)

    def _read_configuration(self):
        return self.__read_register(self.__REG_CONFIG)

    def _calculate_min_current_lsb(self):
        return self.__CALIBRATION_FACTOR / \
            (self._shunt_ohms * self.__MAX_CALIBRATION_VALUE)

    def _read_gain(self):
        configuration = self._read_configuration()
        gain = (configuration & 0x1800) >> self.__PG0
        self.logger.info("gain is currently: %.2fV", self.__GAIN_VOLTS[gain])
        return gain

    def _configure_gain(self, gain):
        configuration = self._read_configuration()
        configuration = configuration & 0xE7FF
        self._configuration_register(configuration | (gain << self.__PG0))
        self._gain = gain
        self.logger.info("gain set to: %.2fV", self.__GAIN_VOLTS[gain])

    def _calibration_register(self, register_value):
        self.logger.debug("calibration: 0x%04x", register_value)
        self.__write_register(self.__REG_CALIBRATION, register_value)

    def _has_current_overflow(self):
        ovf = self._read_voltage_register() & self.__OVF
        return ovf == 1

    def _voltage_register(self):
        register_value = self._read_voltage_register()
        return register_value >> 3

    def _read_voltage_register(self):
        return self.__read_register(self.__REG_BUSVOLTAGE)

    def _current_register(self):
        return self.__read_register(self.__REG_CURRENT, True)

    def _shunt_voltage_register(self):
        return self.__read_register(self.__REG_SHUNTVOLTAGE, True)

    def _power_register(self):
        return self.__read_register(self.__REG_POWER)

    def __validate_voltage_range(self, voltage_range):
        if voltage_range > len(self.__BUS_RANGE) - 1:
            raise ValueError(self.__VOLT_ERR_MSG)

    def __write_register(self, register, register_value):
        register_bytes = self.__to_bytes(register_value)
        self.logger.debug(
            "write register 0x%02x: 0x%04x 0b%s", register, register_value,
            self.__binary_as_string(register_value))
        self.writelist(register, register_bytes)

    def __read_register(self, register, negative_value_supported=False):
        if negative_value_supported:
            register_value = self.reads16(register, little_endian=False)
        else:
            register_value = self.readu16(register, little_endian=False)
        self.logger.debug(
            "read register 0x%02x: 0x%04x 0b%s", register, register_value,
            self.__binary_as_string(register_value))
        return register_value

    @staticmethod
    def __to_bytes(register_value):
        return [(register_value >> 8) & 0xFF, register_value & 0xFF]

    @staticmethod
    def __binary_as_string(register_value):
        return bin(register_value)[2:].zfill(16)

    @staticmethod
    def __max_expected_amps_to_string(max_expected_amps):
        if max_expected_amps is None:
            return ''
        return ', max expected amps: %.3fA' % max_expected_amps


class DeviceRangeError(Exception):
    """Class containing the INA219 error functionality."""

    __DEV_RNG_ERR = ('Current out of range (overflow), '
                     'for gain %.2fV')

    def __init__(self, gain_volts, device_max=False):
        """Construct a DeviceRangeError."""
        msg = self.__DEV_RNG_ERR % gain_volts
        if device_max:
            msg = msg + ', device limit reached'
        super(DeviceRangeError, self).__init__(msg)
        self.gain_volts = gain_volts
        self.device_limit_reached = device_max

class CurrentSensorThread(threading.Thread):
    """
    thread for reading MLX90614 IR Sensor
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
        self._shunt_ohms = 0.1
        self._max_expected_amps = 2.0
        self._currentsensor = INA219(self._shunt_ohms, max_expected_amps=self._max_expected_amps, debug=False)
        self._currentsensor.configure(self._currentsensor.RANGE_16V)
        self.voltage = None
        self.current = None
        self.power = None
        self.shunt_voltage = None

    def run(self):
        while True:
            try:
                try:
                    #Read data of INA219 sensor
                    self.voltage = self._currentsensor.voltage()
                    self.current = self._currentsensor.current()
                    self.power = self._currentsensor.power()
                    self.shunt_voltage = self._currentsensor.shunt_voltage()
                except DeviceRangeError as err:
                    # Current out of device range with specified shunt resistor
                    print(err)
                    break

                isday(True)
                print("Output of INA219: Bus Voltage  : %.3f V" % self.voltage)
                print("                  Bus Current  : %.3f mA" % self.current)
                print("                  Bus Power    : %.3f mW" % self.power)
                print("                  Shunt Voltage: %.3f mV" % self.shunt_voltage)
                #Write data of INA219 sensor into Homematic
                r = requests.get("http://homematic.minixint.at/config/xmlapi/statechange.cgi?ise_id=24742,24743,24744,24745&new_value="+"{:.3f},{:.3f},{:.3f},{:.3f}".format(self.voltage, self.current, self.power, self.shunt_voltage))
                if r.status_code != requests.codes['ok']:
                    print("Data could not be written into the Homatic system variables.")
            except subprocess.TimeoutExpired:
                print("Waited", self.timeout, "seconds, and did not get any valid data from MLX90614")
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

def getanalemma(args, camera, pixelstorage, nparraytype):
    """
    Captures one image of the current sun position to be assembled into one analemma
    """
    print("Analemma Capture Time triggered")
    # Generate analemma subdirectory if this directory is not present and if the analemma parameter is specified
    analemmabase = args.dirname + "/analemma"
    if not os.path.isdir(analemmabase):
        try:
            os.mkdir(analemmabase)
        except OSError:
            print("Creation of the analemma subdirectory %s failed" % analemmabase)
    # Get current time
    timestring = datetime.datetime.now()

    # Save camera settings to apply after HDR capture
    currentsettings = camera.get_control_values()

    # Use autoexposure for first analemma image

    print("Determine exposure through autoexposure on reduced FOV")

    camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, 90)

    camera.set_control_value(asi.ASI_EXPOSURE, 200, auto=True)
    # zero gain and no autogain for analemma
    camera.set_control_value(asi.ASI_GAIN, 0, auto=False)

    # Set ROI to area within FOV. Required to enable smooth working of
    # autoexposure mode
    origroi = camera.get_roi()
    width, height = origroi[2:]
    width //= 2
    height //= 2
    # ensure that conditions for ROI width and height are met
    width -= width % 8
    height -= height % 2

    # with start_x and start_y as None, the ROI is centered in the FOV
    camera.set_roi(start_x=None, start_y=None, width=width, height=height)
    startx, starty, width, height = camera.get_roi()

    if args.debug:
        print('ROI for autoexposure: Width %d, Height %d, xstart %d, ystart %d\n' %
              (width, height, startx, starty))

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

    # Do autoexposure loop reading exposure values from camera
    # until the values are not changing anymore for 5 consecutive times
    print('Waiting for auto-exposure to compute correct settings ...')
    sleep_interval = 0.100
    df_last = None
    autoex_last = None
    matches = 0
    while True:
        time.sleep(sleep_interval)
        settings = camera.get_control_values()
        df = camera.get_dropped_frames()
        autoex = settings['Exposure']
        if df != df_last:
            if args.debug:
                print('   Exposure: {autoex:f} \u03BCs Dropped frames: {df:d}'
                      .format(autoex=settings['Exposure'], df=df))
            if autoex == autoex_last:
                matches += 1
            else:
                matches = 0
            if matches >= 5:
                break
            df_last = df
            autoex_last = autoex

    print("Final autoexposure: %d \u03BCs" % autoex)

    # read image as bytearray from camera
    print("Starting Analemma HDR capture\n")

    camera.stop_video_capture()
    camera.stop_exposure()

    # reset ROI
    camera.set_roi(start_x=origroi[0], start_y=origroi[1],
                   width=origroi[2], height=origroi[3])

    imgs = []

    # Generate exposure scale upfront
    # Start HDR image scale with 4x larger exposure time and stop when
    # exposure time drops below minimum exposure time
    autoex *= 4
    times = []

    print("Starting HDR exposure scale aquisition up to {autoex:d} \u03BCs"
          .format(autoex=autoex))

    while autoex >= controls['Exposure']['MinValue']:
        times.append(autoex)
        autoex //= 2 # Scale exposure by % 2 for HDR scale
    if times[-1] != controls['Exposure']['MinValue']:
        times.append(controls['Exposure']['MinValue'])
    # Start with smallest exposure time
    times = list(reversed(times))

    imgarray = bytearray(args.width*args.height*pixelstorage)

    # Do loop over exposures scaling them with a factor of 2

    camera.set_control_value(controls['Gain']['ControlType'], 1)

    for i, exposure in enumerate(times):
        camera.set_control_value(controls['Exposure']['ControlType'], exposure, auto=False)

        camera.capture(buffer_=imgarray, filename=None)

        print("Exposure: %d us" % exposure)

        # Define file base string (for image and image info file)
        filebase = analemmabase+'/analemma'+timestring.strftime("%Y%m%d%H%M%S_")+str(exposure)
        # convert bytearray to numpy array
        nparray = np.frombuffer(imgarray, nparraytype)
        # reshape numpy array back to image matrix depending on image type.
        imgbay = nparray.reshape((args.height, args.width, args.channels))

        # Debayer image in the case of RAW8 or RAW16 images
        if dodebayer:
            # Define image array here, to ensure that an image array is generated with imgs.append below
            img = np.zeros((args.height, args.width, 3), nparraytype)
            # take care that opencv channel order is B,G,R instead of R,G,B
            cv2.cvtColor(imgbay, eval('cv2.COLOR_BAYER_'+\
                                      bayerpatt[bayerindx][2:][::-1]+\
                                      '2BGR'+debayeralgext), img, 0)
        else:
            img = np.copy(imgbay)

        # postprocess image
        # If aperture should be masked, apply circular masking
        if args.maskaperture:
            #Do mask operation
            img *= mask

        # save image and exposure time into array for postprocessing
        imgs.append(img)

    print("Analemma HDR frame capture finished.")

    print('File Extension is %s' % args.extension.upper())

    # write image based on extension specification and data compression parameters
    for i, img in enumerate(imgs):
        print('Save %d frame of %d' % (i + 1, len(imgs)))
        filebase = analemmabase+'/analemma'+timestring.strftime("%Y%m%d%H%M%S_")+str(times[i])
        if args.extension == 'jpg':
            cv2.imwrite(filebase+'.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY),
                                               args.jpgquality])
        elif args.extension == 'png':
            cv2.imwrite(filebase+'.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION),
                                               args.pngcompression])
        elif args.extension == 'tif':
            # Use TIFFTAG_COMPRESSION=259 to specify COMPRESSION_LZW=5
            cv2.imwrite(filebase+'.tif', img, [259, 5])
        elif args.extension == 'fits':
            # use astropy to write FITS image
            if (args.channels == 1) and not dodebayer:
                hdu = fits.PrimaryHDU(data=img[:, :, 0])
            else:
                hdu = fits.PrimaryHDU(data=[img[:, :, 0], img[:, :, 1], img[:, :, 2]])
            hdu.writeto(filebase + '.fits')


    print("Restoring camera settings to original values.")

    for k in sorted(settings.keys()):
        if k == 'Exposure':
            camera.set_control_value(controls[k]['ControlType'],
                                     currentsettings[k],
                                     auto=args.autoexposure)
        elif k == 'Gain':
            camera.set_control_value(controls[k]['ControlType'],
                                     currentsettings[k],
                                     auto=args.autogain)
        else:
            camera.set_control_value(controls[k]['ControlType'],
                                     currentsettings[k])

    print('Postprocessing HDR frames into HDR image')
    times = np.asarray(times, dtype=np.float32)

    # Estimate camera response
    calibrate = cv2.createCalibrateDebevec()
    # convert images from 16bit to 8bit since algorithms does not support 16bit images
    for i, img in enumerate(imgs):
        img = np.uint8(img)
#    response = calibrate.process(imgs, times)

    # Make HDR Image
#    merge_debevec = cv2.createMergeDebevec()
#    hdr = merge_debevec.process(imgs, times, response)

    # Tonemap HDR image
#    tonemap = cv2.createTonemap(2.2)
#    ldr = tonemap.process(hdr)

    # Perform exposure fusion
#    merge_mertens = cv2.createMergeMertens()
#    fusion = merge_mertens.process(imgs)

    # Write results
#    cv2.imwrite(analemmabase+'/analemma'+timestring.strftime("%Y%m%d%H%M%S_")+'fusion.png', fusion * 255)
#    cv2.imwrite(analemmabase+'/analemma'+timestring.strftime("%Y%m%d%H%M%S_")+'ldr.png', ldr * 255)
#    cv2.imwrite(analemmabase+'/analemma'+timestring.strftime("%Y%m%d%H%M%S_")+'hdr.hdr', hdr)
    print("Analemma HDR image processing finished")


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

def _stamptext(img, caption, lasty, args, left=True, top=True):
    """
    renders the string <caption> on the image <img> under the last line at
    position lasty (bottommost y-position for top=True and topmost y-position
    for top=False) with the font parameters given in args. left and top specify
    the four corners of the image for the caption lines (e.g. left=False,
    top=True is the top right corner as a start point).
    The line margins are taken from the args.textx parameter.
    Depending on the value of args.ftfont the built in font Herhey of
    version <args.fontname> is taken (if args.ftfont equals None) or a loaded
    ttf font is used.
    """
    if args.ftfont is None:
        (textboxwidth, textboxheight), baseline = cv2.getTextSize(caption, args.fontname, args.fontscale, args.fontlinethick)
    else:
        (textboxwidth, textboxheight), baseline = args.ftfont.getTextSize(caption, int(8*args.fontscale), args.fontlinethick)

    if left:
        textx = args.textx
    else:
        textx = args.width - args.textx - textboxwidth
    if top:
        texty = lasty + args.texty + textboxheight
        lasty = texty + baseline
    else:
        texty = lasty - args.texty - baseline
        lasty = texty - textboxheight
    if args.ftfont is None:
        cv2.putText(img, caption, (textx, texty), args.fontname, \
                    args.fontscale, args.fontcolor, args.fontlinethick, \
                    lineType=args.fontlinetype)
    else:
        args.ftfont.putText(img=img, text=caption, org=(textx, texty), \
           fontHeight=int(8*args.fontscale), color=args.fontcolor, \
           thickness=args.fontlinethick, line_type=args.fontlinetype, \
           bottomLeftOrigin=True)
    return lasty


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
# High speed mode setting
parser.add_argument('--hispeed',
                    default='',
                    action='store_true',
                    help='In specified, high-speed mode the 10bit ADC is used. \
                    Otherwise the 14Bit ADC is used. \
                    High-speed mode enables 2x higher frame rate, but reduces \
                    the dynamic range of the pixel values.')
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
                    help='Character/text overlay caption. Use quotes e.g. "Text". \
                    Top left corner of this text is positioned at <textx>, <texty>. \
                    <textx> is used as margin between the lines of text as well. \
                    The font properties are given by the <font...> parameters. \
                    If this parameter is specified together with the --time \
                    parameter then the time is given after this text in brackets.\
                    (Default "")')
# Overlay text x position
parser.add_argument('--textx',
                    default=5,
                    type=check_positive_int,
                    help='x-boundary of distance between text boxes of captions and image frame (Default 5)')
# Overlay text y position
parser.add_argument('--texty',
                    default=5,
                    type=check_positive_int,
                    help='y-boundary of distance between text boxes of captions and image frame (Default 5)')
# Name of font
parser.add_argument('--fontname',
                    default=0,
                    help='Font type number (0-7 ex. 0:simplex/4:triplex/7:script, default 0) \
                    or TTF font name (e.g. ubuntu/Ubuntu-R.ttf')
# Base directory for ttf fonts
parser.add_argument('--fontbase',
                    nargs='?',
                    const='/usr/share/fonts/truetype/',
                    default='/usr/share/fonts/truetype/',
                    help='Path to true type font directory. Any subdirectory of this \
                    path must be specified together with the ttf name in --fontname \
                    parameter (Default /usr/share/fonts/truetype/)')
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
parser.add_argument('--fontscale',
                    default=1.0,
                    type=float,
                    help='Font scale factor that is multiplied with the font-specific base size. (default 1.0)')
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
# Metadata labeling scale relative to caption
parser.add_argument('--detailscale',
                    default=0.8,
                    type=float,
                    help='Scale by which the additional metadata is smaller than the text caption')
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
                    help='Do only postprocessing (video, startrails and keogram) but no image capture. \
                        You have to be in the image directory of the respective day to work.')
parser.add_argument('--serverrepo',
                    default='/mnt/MultimediaAllSkyCam',
                    help='''Position and username of repository to store Imagery and Videos. \
                    If "none" no data will be copied over. If the repository starts with \
                    <username>@<servername>: pattern scp is used. Otherwise a local repository \
                    or a mounted share is used as target directory.\
                    (Default /mnt/MultimediaAllSkyCam)''')
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
                    default='',
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

# Define output types for image metadata
parser.add_argument('--metadata',
                    default='txt,exif',
                    const=None,
                    nargs='?',
                    type=str,
                    help='Targets for image metadata in comma separated list form. \
                    Current available targets are "txt" and "exif". "txt" specifies that a text \
                    file including the meta data with the same base name as the image is written. \
                    "exif" specifies that the metadata is written into exif tags directly into the \
                    images. If specified without parameter not metadata is written.\
                    (Default both text and exif is written.)')

args = parser.parse_args()

args.dirname = os.getcwd()
args.extension = args.filename.lower().split('.')[-1]

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
if isinstance(args.fontname, str):
    print('Font: %s specified' % args.fontname)
    args.ftfont = cv2.freetype.createFreeType2()
    args.ftfont.loadFontData(fontFileName=args.fontbase + args.fontname, id=0)
else:
    args.ftfont = None
    print('Builtin Hershey Font # %d specified' % args.fontname)
    args.fontname = int(args.fontname)
position = ephem.Observer()
position.pressure = 0
position.lon = args.lon * math.pi / 180
position.lat = args.lat * math.pi / 180
isday = IsDay(position)

print("Position: Lat: %s / Lon: %s" % (position.lat, position.lon))

# Switch camera on if not already on
turnonCamera()

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
args.fontscale /= args.bin
args.fontlinethick //= args.bin

print('Selected Image dimensions: %s,%s (binning %d)' % (args.width, args.height, args.bin))
print('Selected Image left and top: %s,%s (binning %d)' % (args.left, args.top, args.bin))

# Use minimum USB bandwidth permitted
camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, args.usbspeed)
camera.set_control_value(asi.ASI_HIGH_SPEED_MODE, 1 if args.hispeed else 0)

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
    args.channels = 1
    args.fontcolor = round(255 * args.fontcolor[0])
if args.type == asi.ASI_IMG_RGB24: #RGB24 Interpolated 3 (RGB) channels per pixel
    pixelstorage = 3
    nparraytype = 'uint8'
    args.channels = 3
    args.fontcolor = [round(255 * c) for c in args.fontcolor]
if args.type == asi.ASI_IMG_RAW16: #RAW16 Uninterpolated Bayer pattern 16bin per pixel
    pixelstorage = 2
    nparraytype = 'uint16'
    args.channels = 1
    args.fontcolor = [round(65535 * c) for c in args.fontcolor]
if args.type == 3: #Y8: One byte (Y) per Bayer pattern
    pixelstorage = 1
    nparraytype = 'uint8'
    args.channels = 1
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
    mask = np.zeros((args.height, args.width, 3 if dodebayer else args.channels), dtype=nparraytype)
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

# split arguments for metadata target specification

if args.metadata is not None:
    args.metadata = args.metadata.split(',')

#Start Temperature and Humidity-Monitoring with DHT22 sensor. Read out every 5min (300sec) and timeout after 10sec

dht22stopFlag = threading.Event()
dht22thread = dht22Thread(dht22stopFlag, camera, 300, 10)
dht22thread.start()
threads.append(dht22thread)

#Start Temperature and Humidity-Monitoring with Homematic sensor data from Roof sensor. Read out every 5min (300sec) and timeout after 10sec

weatherstopFlag = threading.Event()
weatherthread = WeatherThread(weatherstopFlag, camera, 300, 10)
weatherthread.start()
threads.append(weatherthread)

#Start IR Temperature Monitoring with MLX90614 sensor. Read out every 5min (300sec) and timeout after 10sec

IRSensorstopFlag = threading.Event()
IRSensorthread = IRSensorThread(IRSensorstopFlag, camera, 300, 10)
IRSensorthread.start()
threads.append(IRSensorthread)

#Initialize INA219 Current Sensor with the default parameters

CurrentSensorstopFlag = threading.Event()
CurrentSensorthread = CurrentSensorThread(CurrentSensorstopFlag, camera, 300, 10)
CurrentSensorthread.start()
threads.append(CurrentSensorthread)

if args.analemma != '':
    # initialize scheduler with UTC time
    scheduler = BackgroundScheduler(timezone=utc.zone)
    # Calculate trigger UTC time from mean local time
    if args.analemma.lower() == 'meanmidday':
        args.analemma = '12:00:00'
    if args.analemma.lower() != 'now':
        analemmatrigger = lmt2utc(datetime.datetime.strptime(args.analemma, '%H:%M:%S'), position.lon)
    else:
        analemmatrigger = datetime.datetime.utcnow() + datetime.timedelta(seconds=10) # ensure that the event happens after starting the script
    print('Analemma Trigger Time in UTC: %s' % analemmatrigger.strftime("%H:%M:%S"))
    print('                      local : %s' % analemmatrigger.replace(tzinfo=datetime.timezone.utc).astimezone(tz=datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo).strftime("%H:%M:%S"))
    scheduler.start()
    scheduler.add_job(getanalemma, trigger='cron', args=[args, camera, pixelstorage, nparraytype], hour=analemmatrigger.time().hour, minute=analemmatrigger.time().minute, second=analemmatrigger.time().second)

#bMain = False

while bMain:

    try:
        lastisday = isday(args.debug)
        if lastisday and not args.daytime:
            # At daytime with --daytime not specified, skip day time capture
            if args.debug:
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

            # Check if there are more than <args.nightstokeep> directories under images
            # if so remove the older ones until <args.nightstokeep>-1 remain and create new one
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
                exp_ms = 32
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

                # Check dew heater control parameters and decide if it needs
                # to be switched on or off depending on the ambient temperature
                # and the calculated dew-point
                if not isday(False):
                    heaterControl(weatherthread.temperature, weatherthread.dewpoint, sensitivity=1.5, hysteresis=1.0)
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
                    imgbay = nparray.reshape((args.height, args.width, args.channels))
                    cv2.cvtColor(imgbay, eval('cv2.COLOR_BAYER_'+bayerpatt[bayerindx][2:][::-1]+\
                                           '2BGR'+debayeralgext), img, 0)
                else:
                    # reshape numpy array back to image matrix depending on image type
                    img = nparray.reshape((args.height, args.width, args.channels))

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

                # If time parameter is specified, print timestring
                #(in brackets if text parameter is given as well)
                caption = args.text
                if args.time and args.text != "":
                    caption = caption + "(" + timestring.strftime("%d.%b.%Y %X") + ")"
                elif args.time:
                    caption = timestring.strftime("%d.%b.%Y %X")
                print('Caption: %s' % caption)
                if args.debug:
                    print('Takedarkframe:', args.takedarkframe == '')
                if args.takedarkframe == '':
                    if args.debug:
                        print('Writing image caption')
                    lasty = _stamptext(img, caption, 0, args)
                    if args.details:
                        #Output into to left corner of the image underneath the Date-Time Caption
                        dargs = copy.deepcopy(args)
                        dargs.fontscale = args.fontscale*0.8
                        caption = str('Sensor {:.1f}degC'.\
                                      format(camera.get_control_value(asi.ASI_TEMPERATURE)[0]/10))
                        lasty = _stamptext(img, caption, lasty, dargs)
                        if lastisday and autoExp < 1000000:
                            caption = 'Exposure {:.3f} ms'.format(autoExp/1000)
                        else:
                            caption = 'Exposure {:.3f} s'.format(autoExp/1000000)
                        lasty = _stamptext(img, caption, lasty, dargs)
                        caption = str('Gain {:d}'.format(autoGain))
                        lasty = _stamptext(img, caption, lasty, dargs)
                        #Output into the bottom left corner of the image
                        caption = str('Dew Point {:.1f}degC'.\
                                 format(weatherthread.dewpoint))
                        lasty = _stamptext(img, caption, dargs.height, dargs, top=False)
                        caption = str('External Hum. {:.1f}%'.\
                                 format(weatherthread.humidity))
                        lasty = _stamptext(img, caption, lasty, dargs, top=False)
                        caption = str('External Temp. {:.1f}degC'.\
                                 format(weatherthread.temperature))
                        lasty = _stamptext(img, caption, lasty, dargs, top=False)
                        caption = str('IR Amb. Temp. {:.1f}degC'.\
                                 format(IRSensorthread.Ta))
                        lasty = _stamptext(img, caption, lasty, dargs, top=False)
                        caption = str('Sky Temp. {:.1f}degC'.\
                                 format(IRSensorthread.Tobj))
                        lasty = _stamptext(img, caption, lasty, dargs, top=False)
                        caption = str('Housing Dew Point {:.1f}degC'.\
                                 format(dht22thread.dewpoint))
                        lasty = _stamptext(img, caption, lasty, dargs, top=False)
                        caption = str('Housing Hum. {:.1f}%'.\
                                 format(dht22thread.dht22hum))
                        lasty = _stamptext(img, caption, lasty, dargs, top=False)
                        caption = str('Housing Temp. {:.1f}degC'.\
                                 format(dht22thread.dht22temp))
                        lasty = _stamptext(img, caption, lasty, dargs, top=False)
                        #Output into the top right corner of the image
                        caption = str('Bus Power {:.3f}mW'.\
                                 format(CurrentSensorthread.power))
                        lasty = _stamptext(img, caption, 0, dargs, left=False, top=True)
                        if heateron():
                            caption = 'Heater is ON'
                        else:
                            caption = 'Heater is OFF'
                        lasty = _stamptext(img, caption, lasty, dargs, left=False, top=True)
                # save control values of camera and
                # do some simple statistics on image and save to associated text file with the camera settings
                if args.metadata is not None:
                    camctrls = camera.get_control_values()
                    imgstats = get_image_statistics(img)
                    if 'txt' in args.metadata:
                        save_control_values(filebase, camctrls, imgstats)
                # write image based on extension specification and data compression parameters
                if args.extension == 'jpg':
                    thread = saveThread(filebase+'.jpg', img, \
                                        [int(cv2.IMWRITE_JPEG_QUALITY), args.jpgquality], camctrls, imgstats, timestring)
                    thread.start()
                #    print('Saved to %s' % filename)
                elif args.extension == 'png':
                    thread = saveThread(filebase+'.png', img, \
                                        [int(cv2.IMWRITE_PNG_COMPRESSION), args.pngcompression], camctrls, imgstats, timestring)
                    thread.start()
                elif args.extension == 'tif':
                    # Use TIFFTAG_COMPRESSION=259 to specify COMPRESSION_LZW=5
                    thread = saveThread(filebase+'.tif', img, [259, 5], camctrls, imgstats, timestring)
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
                    # Switch off heater
                    turnoffHeater()
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
weatherstopFlag.set()
IRSensorstopFlag.set()
CurrentSensorstopFlag.set()

# Finally wait for all threads to complete
if args.analemma != '':
    scheduler.shutdown()
for t in threads:
    t.join()
print("Exiting Capture")
