import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
from sklearn.cluster import KMeans
import random as random


def preprocess(img):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    img = cv2.medianBlur(img, 21)
    img = img/255

    return img

def plotImage(img, title):
    
    plt.imshow(img)
    #plt.title('Clustered Image')
    plt.savefig('results/image_classified/normal_img/'+title)
    # plt.show()
    
def cropOrig(bRect, oimg):
    # x (Horizontal), y (Vertical Downwards) are start coordinates
    # img.shape[0] = height of image
    # img.shape[1] = width of image

    x,y,w,h = bRect

    # print(x,y,w,h)
    pcropedImg = oimg[y:y+h,x:x+w]

    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]

    y2 = int(h1/10)

    x2 = int(w1/10)

    crop1 = pcropedImg[y1+y2:h1-y2,x1+x2:w1-x2]

    #cv2_imshow(crop1)

    ix, iy, iw, ih = x+x2, y+y2, crop1.shape[1], crop1.shape[0]

    croppedImg = oimg[iy:iy+ih,ix:ix+iw]

    return croppedImg, pcropedImg

def overlayImage(croppedImg, pcropedImg):
    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]

    y2 = int(h1/10)

    x2 = int(w1/10)

    new_image = np.zeros((pcropedImg.shape[0], pcropedImg.shape[1], 3), np.uint8)
    new_image[:, 0:pcropedImg.shape[1]] = (117, 13, 205) # (B, G, R)
    
    new_image[ y1+y2:y1+y2+croppedImg.shape[0], x1+x2:x1+x2+croppedImg.shape[1]] = croppedImg
    new_image[np.where((new_image==[246,57,178]).all(axis=2))] = [117, 13, 205]
    new_img = cv2.medianBlur(new_image, 17)
    
    return new_image

def kMeans_cluster(img):

    # For clustering the image using k-means, we first need to convert it into a 2-dimensional array
    # (H*W, N) N is channel = 3
    image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])

    # tweak the cluster size and see what happens to the Output
    kmeans = KMeans(n_clusters=2, random_state=0).fit(image_2D)
    clustOut = kmeans.cluster_centers_[kmeans.labels_]

    # Reshape back the image from 2D to 3D image
    clustered_3D = clustOut.reshape(img.shape[0], img.shape[1], img.shape[2])

    clusteredImg = np.uint8(clustered_3D*255)

    return clusteredImg

def getBoundingBox(img):

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #print(len(contours))
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    
    

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])

    
    return boundRect, contours, contours_poly, img

def drawCnt(bRect, contours, cntPoly, img):

    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)   


    paperbb = bRect

    for i in range(len(contours)):
      color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
      cv2.drawContours(drawing, cntPoly, i, color)
      #cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
              #(int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
    cv2.rectangle(drawing, (int(paperbb[0]), int(paperbb[1])), \
              (int(paperbb[0]+paperbb[2]), int(paperbb[1]+paperbb[3])), color, 2)
    
    return drawing

def paperEdgeDetection(clusteredImage):
  edged1 = cv2.Canny(clusteredImage, 0, 255, 15)
  edged = cv2.dilate(edged1, kernel=(5, 5), iterations=10)
  edged = cv2.erode(edged, kernel=(5, 5), iterations=8)
  
  edged = cv2.fastNlMeansDenoising(edged, None, 20, 7, 21) 
  # edged = cv2.medianBlur(edged, 3)
  return edged

def edgeDetection(clusteredImage):
  edged1 = cv2.Canny(clusteredImage, 0, 255)
  edged = cv2.dilate(edged1, kernel=(5, 5), iterations=10)
  edged = cv2.erode(edged, kernel=(5, 5), iterations=9)
  
  edged = cv2.fastNlMeansDenoising(edged, None, 20, 7, 21) 
  return edged

def footEdgeDetection(clusteredImage):
  edged1 = cv2.Canny(clusteredImage, 0, 255)
  edged = cv2.dilate(edged1, kernel=(5, 5), iterations=10)
  edged = cv2.erode(edged, kernel=(5, 5), iterations=9)
  
  # edged = cv2.fastNlMeansDenoising(edged, None, 20, 7, 21) 
  return edged

def calcFeetSize(pcropedImg, fboundRect):
  x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]

  y2 = int(h1/10)

  x2 = int(w1/10)

  fh = y2 + fboundRect[0][3]
  fw = x2 + fboundRect[0][2]
  ph = pcropedImg.shape[0]
  pw = pcropedImg.shape[1]

  # print("Feet height: ", fh)
  # print("Feet Width: ", fw)

  # print("Paper height: ", ph)
  # print("Paper Width: ", pw)

  opw = 210
  oph = 297

  ofs = 0.0

  if fw>fh:
    ofs = (oph/pw)*fw
  else :
    ofs = (oph/ph)*fh



  return ofs/10, fh, fw, ph, pw