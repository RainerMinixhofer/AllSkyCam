#!/usr/bin/env python3
"""
Postprocessing Script for generating startrails and keogram from images of the AllSky Camera
"""
# pylint: disable=C0301,C0103
import glob
import os
import argparse
import cv2
import numpy

__author__ = 'Rainer Minixhofer'
__version__ = '0.0.1'
__license__ = 'MIT'

parser = argparse.ArgumentParser(description='Generate one image with startrails and one keogram image from multiple images')

# Argument to specify location of ASI SDK Library (default specified in env_filename
parser.add_argument('--sourcedir',
                    default='.',
                    help='source directory (default ".")')
parser.add_argument('--extension',
                    default='png',
                    help='file extension of image files to load (default png)')
parser.add_argument('--threshold',
                    default=0.1,
                    type=float,
                    help='Brightness threshold for startrails ranges from 0 (black) to 1 (white) \
                    A moonless sky is around 0.05 while full moon can be as high as 0.4 \
                    (default 0.1)')
parser.add_argument('--startrailsoutput',
                    default='startrails.png',
                    help='startrails output file (default "startrails.png")')
parser.add_argument('--keogramoutput',
                    default='keogram.png',
                    help='keogram output file (default "keogram.png")')
args = parser.parse_args()
# Check validity of parameters
if not 0 <= args.threshold <= 1:
    print('Brightness threshold ranges from 0 to 1')
    exit()
files = glob.glob(args.sourcedir + "/*." + args.extension)
# Create array for statistics
stats = numpy.zeros(len(files))
# Initialize startrails and keogram image
startrails = None
keogram = None

for idx, file in enumerate(files):
    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if image is None:
        print('Error reading file ' + file)
        exit()
    mean = numpy.amax(cv2.mean(image))
    if image.dtype.name == 'uint8':
        mean /= 255.0
    elif image.dtype.name == 'uint16':
        mean /= 65535.0
    else:
        print('Image ' + file + ' has wrong data type ' + image.dtype.name)
    # Save mean in statistics array
    stats[idx] = mean
    # Add image to startrails image if mean is below threshold
    if mean <= args.threshold:
        status = 'Processing'
        if startrails is None:
            startrails = image.copy()
        else:
            startrails = cv2.max(startrails, image)
    else:
        status = 'Skipping'
    if keogram is None:
        height, width, channels = image.shape
        keogram = numpy.zeros((height, len(files), channels), image.dtype)
    else:
        keogram[:, idx] = image[:, width//2]
    print('%s #%d / %d / %s / %6.4f' % (status, idx, len(files), os.path.basename(file), mean))

#calculate some statistics: Min, Max, Mean and Median of Mean
min_mean, max_mean, min_loc, max_loc = cv2.minMaxLoc(stats)
mean_mean = numpy.mean(stats)
median_mean = numpy.median(stats)

print("Minimum: %6.4f / maximum: %6.4f / mean: %6.4f / median: %6.4f" % (min_mean, max_mean, mean_mean, median_mean))

#If we still don't have an image (no images below threshold), copy the minimum mean image so we see why
if startrails is None:
    print('No images below threshold, writing the minimum image only')
    startrails = cv2.imread(files[min_loc], cv2.IMREAD_UNCHANGED)
if args.extension == "png":
    args.fileoptions = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
elif args.extension == "jpg":
    args.fileoptions = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
else: args.fileoptions = []

status = cv2.imwrite(args.startrailsoutput, startrails, args.fileoptions)
print("Image", args.startrailsoutput, " written to file-system : ", status)
status = cv2.imwrite(args.keogramoutput, keogram, args.fileoptions)
print("Image", args.keogramoutput, " written to file-system : ", status)
