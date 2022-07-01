# foot-size-estimation

## Introduction
Our project is used to estimate feet size just by 2D image of your feet. To do that, you need to prepare white paper, put your feet in that paper. Take a picture that you can see clearly both your feet and paper like this one:
![feet on paper picture](https://github.com/raisinbl/foot-size-estimation/blob/main/image_classified/normal_img/23809.jpeg width="200" height="200") Wait a minute and you'll get your feet size :smile:

## Installation
Clone our repo: `https://github.com/raisinbl/foot-size-estimation/` <br>
`cd foot-size-estimation`<br>
Install library: `pip install requirements.txt` <br>
Enjoy your app: `streamlit run streamlit_main.py`, you can run it on other divices by go to Network URL:
![image](https://user-images.githubusercontent.com/66005831/176816188-2d4bd5d2-7a94-4852-ba6d-139b67d3db68.png)

## How it worked?
Image segmentation is a importation technique in Machine Learning, Deep Learning in general and Computer Vision in particular.<br>
There are two approaches for image segmentation:<br>
  - Traditional image processing that apply filter, thresolding parameter on image, apply some Machine Learning to get result. No need dataset for training but quite slow in processing time due to high computation 
  - Deep Learning based: Using DL model to get result. This approach show incredible efficient on proccesing time and computation, even if for video

Our approach is Traditional image processing. Our pipeline can broken down as:
  - Image Preprocessing: Convert image to HSV color space, tuning intensity by Gamma Correction, Blur image by Median blur.
  - Segmentation: Using Kmeans Clustering.
  - Draw contour and approximation to get extract paper, and then do the same after crop paper to get feet bounding box.
  - Calculation feet size.

#### What new in our approach
 From our observation on data, we divided 5 kinds of data:
  - Images that have feet same color as background
  - Images that have paper same color as background
  - Images that have curve background
  - Images that have pattern background
  - Normal Images: Paper, feet, background have distinct and homogenious color, pattern
 
 For each type of image, we need particular image preprocessing, The rest of each image type procesing is the same pipline. So for each input image, we use SVM or KNN to classify it.
 
 
