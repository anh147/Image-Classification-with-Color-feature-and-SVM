import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, datasets

def crop_image(img):
    """c
    This is a function that crops extra white background
    around product.
    """
    mask = img!=255
    mask = mask.any(2)
    mask0,mask1 = mask.any(0),mask.any(1)
    colstart, colend = mask0.argmax(), len(mask0)-mask0[::-1].argmax()+1
    rowstart, rowend = mask1.argmax(), len(mask1)-mask1[::-1].argmax()+1
    return img[rowstart:rowend, colstart:colend]

# tinh histogram cua kenh H tu anh 
def hsv_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0]
    return np.bincount(h.ravel(), minlength=256)


X = []
y = []

for img_dir in glob.glob('leedsbutterfly/images'+'/*.png'):
        # print("count", count, "label", label, img_dir)
        im = cv2.imread(img_dir)
        im_size= np.shape(im)
        
        #Read seg image
        
        # print(img_dir)
        label = img_dir.split("\\")
        # index = label.split(".")
        index = str(label[1][0:7])
        # print(index)
        seg_dir = 'leedsbutterfly/segmentations/' + index + '_seg0.png'
        print("seg dir", seg_dir)
        im_seg = cv2.imread(seg_dir)
        # cv2.imshow("seg", im_seg)
        # cv2.waitKey(0)
        # print("read successfully")

        #Remove background
        x = im_seg.reshape((im_size[0]*im_size[1]*3))
        x = x / 255
        x = x.astype(np.uint8)
        x = x.reshape((np.shape(im_seg)[0],np.shape(im_seg)[1],3))
        new_im = x*im
        # cv2.imshow("seg", new_im)
        # cv2.waitKey(0)
        
        #Extract H channel from HSV
        hist = hsv_histogram(new_im)

        X.append(hist)
        y.append(label[1][1:3])
        # count += 1

#-------------------------------------------------------------------#
#splitting the dataset into 80% training data and 20% testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.10, random_state=0)

#-------------------------------------------------------------------#
#building KNN model
KNN_model = KNeighborsClassifier(n_neighbors=5, p = 2, weights = 'distance')
KNN_model.fit(X_train, Y_train)

#prediction on testing data
Y_predict = KNN_model.predict(X_test)

# print("Y predict", len(Y_predict))

count = 0
for i in range(len(Y_predict)):
    if Y_predict[i] == Y_test[i]:
        count += 1

print("acc knn", count/len(Y_predict))

#---------------------------------------------------------------------#
# khởi tạo SVM classifier
clf = svm.SVC(C = 2, kernel = 'linear')
# Train classifier với dữ liệu
clf.fit(X_train, Y_train)

#prediction on testing data
Y_predict = clf.predict(X_test)

# print("Y predict", len(Y_predict))

count = 0
for i in range(len(Y_predict)):
    if Y_predict[i] == Y_test[i]:
        count += 1

print("acc svm", count/len(Y_predict)*100, "%")

print(clf.score(X_test, Y_test))