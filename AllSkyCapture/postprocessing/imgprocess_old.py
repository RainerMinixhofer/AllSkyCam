#!/usr/bin/env python3
"""
Image processing Script for selected images from the AllSkyCam
inspired by the tutorial https://docs.opencv.org/3.4.3/d3/db4/tutorial_py_watershed.html
"""
# pylint: disable=C0301,C0103
import glob
import os
import argparse
import cv2
import numpy as np
import skimage

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

if args.extension == "png":
    args.fileoptions = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
elif args.extension == "jpg":
    args.fileoptions = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
else: args.fileoptions = []

#Get image dimensions of first image (assuming the others have the same dimensions)
image = cv2.imread(files[args.start-1], cv2.IMREAD_UNCHANGED)
height, width, channels = image.shape
mask = np.zeros((height, width), dtype='uint8')
#Define circle with origin in center of image and radius given by the smaller side of the image
cv2.circle(mask, (width//2, height//2), min([width//2, height//2]), 1, -1)


for idx, file in enumerate(files[args.start-1:args.stop]):
    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if image is None:
        print('Error reading file ' + file)
        exit()
    print('Processing #%d/%d / %s' % (idx + args.start, args.stop - args.start + 1, os.path.basename(file)))
    grayorg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert 16 bit image to 8 bit because cv2.threshold cannot handle 16bit
    if image.dtype.name == 'uint16':
        gray = (grayorg/256).astype('uint8')
    else:
        gray = grayorg
    # Equalize histogram for contrast enhancement
    gray = cv2.equalizeHist(gray)
    # Do approximate thresholding with Otsu's binarization
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Noise removal using Morphological closing operation
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Sure background using Dialation
    bg = cv2.dilate(closing, kernel, iterations=1)

    # Finding foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    ret, fg = cv2.threshold(dist_transform, 0.02 * dist_transform.max(), 255, cv2.THRESH_BINARY)

    # Apply circular mask of aperture of camera and found segmentation to image
    # We scale fg to the range [0,1], convert it to uint16 and use broadcasting to get it 
    # to the same channel number than image
    fg *= mask
    image *= (fg>0).astype('uint16')[:, :, np.newaxis]

    # Apply CLAHE to image

    imageo = skimage.exposure.equalize_adapthist(image[:, :, ::-1], clip_limit=0.03)
    #  Convert image to LAB Color model. 
    #  We first convert to float since the 16 bit integers are not supported by cvtColor
#    lab = color.rgb2lab(image)
#    nimg = np.float32(image)
#    nimg *= 1.0/65535
#    nlab = cv2.cvtColor(nimg, cv2.COLOR_BGR2LAB)
#    lab = (nlab * 65535).astype('uint16')
    
    #  Split the lab image into the three channels
#    l, a, b = cv2.split(lab)
#    pa = 'l'
#    print('array',pa,'shape:', eval(pa).shape, 'datatype:',eval(pa).dtype, 'max:',np.amax(eval(pa)), 'min:', np.amin(eval(pa)), 'uniques:', len(np.unique(eval(pa))))
    #  Apply CLAHE to L-channel
#    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#    cl = clahe.apply(l)
    
    #  Merge the CLAHE enhanced L-channel back with the a and b channel
#    lab = cv2.merge((cl,a,b))
    
    # Convert back into RGB color space
#    image = color.lab2rgb(lab)
    
#    nlab = np.float32(lab)
#    nlab *= 1.0/65535
#    pa = 'nlab'
#    print('array',pa,'shape:', eval(pa).shape, 'datatype:',eval(pa).dtype, 'max:',np.amax(eval(pa)), 'min:', np.amin(eval(pa)), 'uniques:', len(np.unique(eval(pa))))
#    nimg = cv2.cvtColor(nlab, cv2.COLOR_LAB2BGR)
#    image = (nimg * 65535).astype('uint16')
#    pa = 'image'
#    print('array',pa,'shape:', eval(pa).shape, 'datatype:',eval(pa).dtype, 'max:',np.amax(eval(pa)), 'min:', np.amin(eval(pa)), 'uniques:', len(np.unique(eval(pa))))

    filename = os.path.splitext(file)[0] + args.outputmod + os.path.splitext(file)[1]

    status = cv2.imwrite(filename, skimage.img_as_uint(imageo)[:, :, ::-1], args.fileoptions)
    if not status:
        print("Image", filename, " writing to file-system failed ")
        exit()
    os.system("scp -q " + filename + " rainer@server03:/var/www/html/kameras/allskycam/test/")
