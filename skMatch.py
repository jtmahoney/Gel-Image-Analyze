# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 19:39:50 2017

@author: jmahoney
"""

import numpy as np
import skimage as sk
from scipy.ndimage import rotate as ro
import skimage.io as io
from skimage.filters import sobel
from skimage.transform import rescale, rotate
from skimage.morphology import watershed
from matplotlib import pyplot as plt

from fileLocations import imageFiles, templateLocation

def compressImg(files, scaleFactor):
    imgs = []
    for f in files:
        img = io.imread(f)#, as_grey=True)
        #img = sk.img_as_ubyte(img)
        cimg = rescale(img, scaleFactor)
        imgs.append(cimg)
        #io.imshow(cimg)
        #io.show()
    return imgs

def imgHisto(files):
    imgs = compressImg(files, 0.1)
    #rots = [30,0]
    rots = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]

    best_images = []
    for img in imgs:
        rot_imgs = []
        positives = []
        negatives = []
        slopes = []
        for rot in rots:
            crop_img = img[int(img.shape[0]/20):int(img.shape[0]/1.2)]
            ir = ro(crop_img, rot, mode="nearest", output=None, reshape=False)
            av = np.average(ir)
            el_map = sobel(ir)
            markers = np.zeros_like(ir)
            markers[ir < av] = -1
            markers[ir > av*1.4] = 30
            seg = watershed(el_map, markers)
            rot_imgs.append(seg)
            hist = np.sum(seg, axis=0)
            pos = 0
            negs = 0
            #print(hist)
            for ea in range(len(hist)-1):
                if hist[ea] < 0:
                    negs = negs + hist[ea]
                if hist[ea] > 0:
                    pos = pos + hist[ea]
            positives.append(pos)
            negatives.append(negs)

        #print(rots[np.argmax(positives)])
        selected_im = rot_imgs[np.argmin(negatives)]
        io.imshow(selected_im)
        io.show()
        best_images.append(selected_im)
                #print(hist[ea])
            #slopes.append(np.average(hist))
            #io.imshow(seg)
            #io.show()
    print(best_images)
    #return best_images
    
    

imgHisto(imageFiles)

        
        

