from sklearn import datasets, svm, metrics
import os
import cv2

def classifier(upload):
    """
    Find what type of uploaded image belong for by using SVM
    """
    dict_label = {'normal_img': 0,
                'white_background_img': 1,
                'skin_background_img': 2,
                'pattern_background_img': 3,
                'curve_background_img': 4}
    X = []
    y = []
    for folder in os.listdir('image_classified'):
        for img_path in os.listdir('image_classified/' + folder):
            img = cv2.imread('image_classified/' + folder + '/' + img_path)
            img = cv2.resize(img, (128, 128))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # resize img to (128, 128)
            img = img/255
            # print(img.shape)
                
            # flatten img to 1D array
            img = img.flatten()
            X.append(img)
            y.append(dict_label[folder])
            
    n_samples = len(os.listdir('data/data_foot/images/'))
    clf = svm.SVC(gamma=0.001, C=100)
    clf.fit(X, y)
    
    # test_img = cv2.imread(img_path)
    test_img = cv2.resize(upload, (128, 128))
    # test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    # resize test_img to (128, 128)
    test_img = test_img/255
    # print(test_img.shape)
            
    # flatten test_img to 1D array
    test_img = test_img.flatten()

    predict = clf.predict([test_img])
    
    return predict[0]

if __name__ == '__main__':
    img = cv2.imread('image_classified/normal_img/23789.jpeg')
    predict = classifier(img)
    print(predict)