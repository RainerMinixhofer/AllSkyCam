#!/usr/bin/env python
"""
Demonstrating how to do autoexposure and subsequent capturing of HDR exposure scaled frames
"""

# pylint: disable=C0103,W0702

import sys
import time
import zwoasi as asi # pylint: disable=E0401


__author__ = 'Steve Marple'
__version__ = '0.0.22'
__license__ = 'MIT'


def save_control_values(fname, sett):
    """
    saves all control values of camera in settings into file with filname
    """
    fname += '.txt'
    with open(fname, 'w') as f:
        for ky in sorted(sett.keys()):
            f.write('%s: %s\n' % (ky, str(sett[ky])))
    print('Camera settings saved to %s' % fname)


asi_filename = '/usr/lib/libASICamera2.so'

# Initialize zwoasi with the name of the SDK library
asi.init(asi_filename)

num_cameras = asi.get_num_cameras()

if num_cameras == 0:
    print('No cameras found')
    sys.exit(0)

cameras_found = asi.list_cameras()  # Models names of the connected cameras

if num_cameras == 1:
    camera_id = 0
    print('Found one camera: %s' % cameras_found[0])
else:
    print('Found %d cameras' % num_cameras)
    for n in range(num_cameras):
        print('    %d: %s' % (n, cameras_found[n]))
    # TO DO: allow user to select a camera
    camera_id = 0
    print('Using #%d: %s' % (camera_id, cameras_found[camera_id]))

camera = asi.Camera(camera_id)

# Get and show all camera properties
print('')
print('Camera properties:')
camera_info = camera.get_camera_property()
for cn in sorted(camera_info.keys()):
    print('    %s: %s' % (cn, camera_info[cn]))

# Get and show all of the camera controls
print('')
print('Camera controls:')
controls = camera.get_controls()
for cn in sorted(controls.keys()):
    print('    %s:' % cn)
    for k in sorted(controls[cn].keys()):
        print('        %s: %s' % (k, repr(controls[cn][k])))

# Use minimum USB bandwidth permitted
camera.set_control_value(controls['BandWidth']['ControlType'],
                         controls['BandWidth']['MinValue'])

exit()

# Disable dark frame subtraction
camera.disable_dark_subtract()

# Set gain to zero (for daylight imaging)
camera.set_control_value(controls['Gain']['ControlType'], 0)
# Enable auto exposure mode and set default value
print('\nEnabling auto-exposure mode')
camera.set_control_value(controls['Exposure']['ControlType'], 
                         controls['Exposure']['DefaultValue'], auto=True)
camera.set_control_value(controls['WB_B']['ControlType'], 
                         controls['WB_B']['DefaultValue'])
camera.set_control_value(controls['WB_R']['ControlType'], 
                         controls['WB_R']['DefaultValue'])
camera.set_control_value(controls['Flip']['ControlType'], 
                         controls['Flip']['DefaultValue'])
camera.set_control_value(controls['HighSpeedMode']['ControlType'], 1)

# Set ROI to area within FOV. Required to enable smooth working of
# autoexposure mode
startx, starty, width, height = camera.get_roi()
width //= 2
height //= 2
# ensure that conditions for ROI width and height in the SDK are met
width -= width % 8
height -= height % 2

camera.set_roi(start_x=None, start_y=None, width=width, height=height)
startx, starty, width, height = camera.get_roi()

print('ROI for autoexposure: Width %d, Height %d, xstart %d, ystart %d\n' %
      (width, height, startx, starty))

# print all settings before starting autoexposure video capture
settings = camera.get_control_values()

for k in sorted(settings.keys()):
    print('%s: %s' % (k, str(settings[k])))

# Enable video mode
try:
    # Force any single exposure to be halted
    camera.stop_exposure()
except (KeyboardInterrupt, SystemExit):
    raise
except:
    pass

print('Enabling video mode')
camera.start_video_capture()

# Keep max gain to the default but allow exposure to be increased to its maximum value if necessary
camera.set_control_value(controls['AutoExpMaxExpMS']['ControlType'],
                         controls['AutoExpMaxExpMS']['MaxValue'])
camera.set_control_value(controls['AutoExpTargetBrightness']['ControlType'],
                         controls['AutoExpTargetBrightness']['MaxValue'])

print('Waiting for auto-exposure to compute correct settings ...')
sleep_interval = 0.100
df_last = None
gain_last = None
exposure_last = None
matches = 0
while True:
    time.sleep(sleep_interval)
    settings = camera.get_control_values()
    df = camera.get_dropped_frames()
    gain = settings['Gain']
    exposure = settings['Exposure']
    if df != df_last:
        print('   Gain {gain:d}  Exposure: {exposure:f} Dropped frames: {df:d}'
              .format(gain=settings['Gain'],
                      exposure=settings['Exposure'],
                      df=df))
        if gain == gain_last and exposure == exposure_last:
            matches += 1
        else:
            matches = 0
        if matches >= 5:
            break
        df_last = df
        gain_last = gain
        exposure_last = exposure

# Video capture has to be stopped before changing the ROI back to normal
try:
    # Force any single exposure to be halted
    camera.stop_video_capture()
    camera.stop_exposure()
except (KeyboardInterrupt, SystemExit):
    raise
except:
    pass

#timeout = (camera.get_control_value(asi.ASI_EXPOSURE)[0] / 1000) * 2 + 500
#camera.default_timeout = timeout

if camera_info['IsColorCam']:
    print('Capturing a single color frame')
    filebase = 'image_video_color'
    camera.set_image_type(asi.ASI_IMG_RGB24)
else:
    print('Capturing a single 8-bit mono frame')
    filebase = 'image_video_mono'
    camera.set_image_type(asi.ASI_IMG_RAW8)

camera.set_roi(start_x=None, start_y=None, width=None, height=None)

# Start HDR image scale with 4x larger exposure time and stop when
# exposure time drops below 100us

exposure *= 4

while exposure >= 100:
    camera.set_control_value(controls['Exposure']['ControlType'], exposure)
    filenam = filebase+'_'+str(exposure)+'.jpg'
    camera.capture(filename=filenam)
    print('Saved to %s' % filenam)
    save_control_values(filenam, camera.get_control_values())
    exposure //= 2
    while camera.get_exposure_status() != asi.ASI_EXP_IDLE:
        time.sleep(sleep_interval)
        print('Waiting until camera is ready')
