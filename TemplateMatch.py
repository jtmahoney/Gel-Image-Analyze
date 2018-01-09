import cv2
import numpy as np
from matplotlib import pyplot as plt

imageFiles = ['R5887 Plate 1 Column 2 (8).tif', 
              'R5885 QC Column 1 and 2 (9).tif',
              'R5887 Plate 1 Column 1 (8).tif',
              'R5887 Plate 1 Column 3 (8).tif',
              'R5887 Plate 1 Column 4 (6).tif',
              'R5807 Plate 3 Column 1 (8).tif', 
              'R5807 Plate 3 Column 2 (8).tif', 
              'R5807 Plate 3 Column 3 (2) and 1-B08 (1) and 2-B08 (1).tif', 
              'R5807 ReQC (2).tif', 
              'R5876 ReQC (1) and R5877 ReQC (4).tif', 
              '10D_R5807 Plate 3 Column 1 (8).tif']

templateLocation = ['lrg_ladder4_R5887 Plate 1 Column 2 (8).tif']

SZ = 352
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def compressImages(files, compressionFactor = 5):
    returnImages = []
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        #print(img.dtype)
        height, width = img.shape[:2]
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small_img = cv2.resize(img, (int(width/compressionFactor),int(height/compressionFactor)), interpolation = cv2.INTER_CUBIC)
        returnImages.append(small_img)
    return returnImages

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1,skew, -0.5*SZ*skew], [0,1,0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

#compressImages(imageFiles)
def tempMatch(files, templateLocation):
    temps = compressImages(templateLocation)
    template = temps[0]
    plt.imshow(template)
    plt.show()
    h,w = template.shape[:2]
    imgs = compressImages(imageFiles)
    for loc in range(len(imgs)):
        #uncomment if you wish to deskew images... kinda works
        #imgs[loc] = deskew(imgs[loc])
        method = cv2.TM_CCOEFF#_NORMED
        res = cv2.matchTemplate(imgs[loc], template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        #these will be used when sectioning gel image
        #print(top_left)
        #print(bottom_right)
        cv2.rectangle(imgs[loc],top_left, bottom_right, 0, 3)
        plt.imshow(imgs[loc])
        plt.show()

tempMatch(imageFiles, templateLocation)
