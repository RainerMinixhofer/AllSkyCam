#!/usr/bin/env python3
"""
Image processing Script for selected images from the AllSkyCam
Segmentation inspired by the tutorial
http://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
"""
# pylint: disable=C0301,C0103
import glob
import os
import argparse
import numpy as np
import scipy
import skimage
from skimage.morphology import watershed
import imageio

__author__ = 'Rainer Minixhofer'
__version__ = '0.0.1'
__license__ = 'MIT'

parser = argparse.ArgumentParser(description='parse a selected interval of images and improve their quality through computational methods')

# Argument to specify location of ASI SDK Library (default specified in env_filename
parser.add_argument('--sourcedir',
                    default='.',
                    help='source directory (default ".")')
parser.add_argument('--extension',
                    default='png',
                    help='file extension of image files to load (default png)')
parser.add_argument('--outputextension',
                    default='png',
                    help='file extension of image files to save (default png)')
parser.add_argument('--outputmod',
                    default='_mod',
                    help='modifier to add to base of output file (default "_mod")')
parser.add_argument('--start',
                    default=1,
                    type=int,
                    help='Start index for file range (default 1')
parser.add_argument('--stop',
                    default=-1,
                    type=int,
                    help='Stop index for file range. Negative for counting from end of file range. (default -1')
args = parser.parse_args()
# Check validity of parameters
files = glob.glob(args.sourcedir + "/*." + args.extension)
if not files:
    print('No files of extension', args.extension, 'found in directory', args.sourcedir)
    exit()
if not 1 <= args.start <= len(files):
    print('Start ranges between 1 and %d' % len(files))
    exit()
if not 1 <= args.stop <= len(files):
    print('Stop ranges between 1 and %d' % len(files))
    exit()
if not args.start <= args.stop:
    print('Start mus be smaller than stop')
    exit()
# Create array for statistics
stats = np.zeros(len(files))

#Get image dimensions of first image (assuming the others have the same dimensions)
image = skimage.io.imread(files[args.start-1])
height, width, channels = image.shape
mask = np.zeros((height, width), dtype='uint8')
#Define mask for circle with origin in center of image and radius given by the smaller side of the image
row, col = np.ogrid[:height, :width]
cnt_row, cnt_col = height / 2, width / 2
outer_disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 > (height / 2)**2)


for idx, file in enumerate(files[args.start-1:args.stop]):

    image = skimage.io.imread(file)
    if image is None:
        print('Error reading file ' + file)
        exit()
    print('Processing #%d/%d / %s' % (idx + args.start, args.stop - args.start + 1, os.path.basename(file)))
    # Apply circular outer_disk_mask
    image[outer_disk_mask] = [0, 0, 0]
    # Contrast Stretching
    p2, p98 = np.percentile(image, (2, 98))
    imageo = skimage.exposure.rescale_intensity(image, in_range=(p2, p98))
    # Adaptive Equalization (CLAHE) does make the sky very noisy Contrast stretching is better
    #imageo = skimage.exposure.equalize_adapthist(image, clip_limit=0.03)

    # Do segmentation on grayscale image converted to 16bit
    gray = skimage.img_as_ubyte(skimage.color.rgb2gray(imageo))
    markers = np.zeros_like(gray)
    markers[gray < 30] = 1
    markers[gray > 150] = 2
    elevation_map = skimage.filters.sobel(gray)

    segmentation = watershed(elevation_map, markers)
    segmentation = scipy.ndimage.binary_fill_holes(segmentation - 1)
    imageo = image
    imageo[segmentation] = [0, 0, 0]

    print('image shape:', imageo.shape, 'datatype:', imageo.dtype, 'min:', np.amin(imageo), 'max:', np.amax(imageo))

    filename = os.path.splitext(file)[0] + args.outputmod + '.' + args.outputextension

    if args.outputextension == "png":
        imageio.imwrite(filename, imageo, optimize=True)
    elif args.outputextension == "jpg":
        imageio.imwrite(filename, imageo, optimize=True, quality=99)
    else:
        imageio.imwrite(filename, imageo)

#    if not status:
#        print("Image", filename, " writing to file-system failed ")
#        exit()
    os.system("scp -q " + filename + " rainer@server03:/var/www/html/kameras/allskycam/test/")
