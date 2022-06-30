import skimage
import argparse
from utils import *
import os
import shutil
from tqdm import tqdm
from classifier import *

def main(img_path):
    # parser = argparse.ArgumentParser()
    
    # parser.add_argument('--image', type=str, default='', help='Path to image')
    
    # args = parser.parse_args()
    
    dict_label = {0: 'normal_img',
              1: 'white_background_img',
              2: 'skin_background_img',
              3: 'pattern_background_img',
              4: 'curve_background_img'}
    
    om_img = skimage.io.imread(img_path)
    # test = cv2.imread()
    cls = classifier(om_img)
    
    if os.path.exists('results/image_classified/normal_img/' + img_path.split('/')[-1].split('.')[0]):
        shutil.rmtree('results/image_classified/normal_img/' + img_path.split('/')[-1].split('.')[0])
    os.mkdir('results/image_classified/normal_img/' + img_path.split('/')[-1].split('.')[0])
    
    preprocessed_img = preprocess(og_img,cls)
    plotImage(preprocessed_img, img_path.split('/')[-1].split('.')[0]+'/preprocessed.jpg')
    
    clustered_img = kMeans_cluster(preprocessed_img)
    plotImage(clustered_img, img_path.split('/')[-1].split('.')[0]+'/clustered.jpg')
    
    edge_detected_img = paperEdgeDetection(clustered_img)
    plotImage(edge_detected_img, img_path.split('/')[-1].split('.')[0]+'/edged.jpg')
    
    boundRect, contours, contours_poly, img = getBoundingBox(edge_detected_img)
    pdraw = drawCnt(boundRect[0], contours, contours_poly, img)
    plotImage(pdraw, img_path.split('/')[-1].split('.')[0]+'/boundingBox.jpg')
    
    cropped_img, pcropped_img = cropOrig(boundRect[0], clustered_img)
    plotImage(cropped_img, img_path.split('/')[-1].split('.')[0]+'/cropped.jpg')
    plotImage(pcropped_img, img_path.split('/')[-1].split('.')[0]+'/pcropped.jpg')
    
    new_img = overlayImage(cropped_img, pcropped_img)
    plotImage(new_img, img_path.split('/')[-1].split('.')[0]+'/overlayed.jpg')
    
    fedged = footEdgeDetection(new_img)
    fboundRect, fcnt, fcntpoly, fimg = getBoundingBox(fedged)
    fdraw = drawCnt(fboundRect[0], fcnt, fcntpoly, fimg)
    plotImage(fdraw, img_path.split('/')[-1].split('.')[0]+'/fboundingBox.jpg')
    
    ofs, fh, fw, ph, pw = calcFeetSize(pcropped_img, fboundRect)
    
    txt = f"[INFO] Feet height: {fh} Feet width: {fw} Paper height: {ph} Paper width: {pw} Feet size (cm): {ofs}"
    
    print(txt)
    
    with open('results/image_classified/normal_img/' + img_path.split('/')[-1].split('.')[0] + '/result.txt', 'w') as f:
        f.write(txt)
  
# main('data/data_foot/images/23868.jpeg')

