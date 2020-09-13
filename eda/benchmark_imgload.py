#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 18:00:49 2020

@author: dhanley
"""

from PIL import Image
import cv2
from turbojpeg import TurboJPEG

fname = '/Users/dhanley//Downloads/20200726_120831.jpg'
fout = '/Users/dhanley//Downloads/tmp.jpg'
# using default library installation
jpeg = TurboJPEG()

# decoding input.jpg to BGR array
def turboload(f):
    in_file = open(f, 'rb')
    bgr_array = jpeg.decode(in_file.read())
    in_file.close()
    return bgr_array[:,:,::-1]

def turbodump(f, img):
    # encoding BGR array to output.jpg with default settings.
    out_file = open(f, 'wb')
    out_file.write(jpeg.encode(img[:,:,::-1]))
    out_file.close()

%time for i in range(10): img = turboload(fname)
Image.fromarray(img)
%time for i in range(10): turbodump(fout, img)


%time for i in range(10): img = cv2.imread(fname)
Image.fromarray(img)
%time for i in range(10): cv2.imwrite(fout, img[:,:,::-1])


