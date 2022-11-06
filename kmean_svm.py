import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, datasets
from sklearn.cluster import KMeans


data_bov = []
for img_dir in glob.glob('leedsbutterfly/images'+'/*.png'):
        # print("count", count, "label", label, img_dir)
        im = cv2.imread(img_dir)
        im_size= np.shape(im)
        # im = im.reshape((im_size[0]*im_size[1]*3))
        # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # X = np.append(X, im, axis=0)
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
        x = im_seg.reshape((im_size[0]*im_size[1]*3))
        x = x / 255
        x = x.astype(np.uint8)
        x = x.reshape((np.shape(im_seg)[0],np.shape(im_seg)[1],3))
        
        new_im = x*im
        # cv2.imshow("seg", new_im)
        # cv2.waitKey(0)
        new_im = new_im.reshape(im_size[0]*im_size[1], 3)
        #Lấy 10000 pixel 
        ran= np.random.permutation(im_size[0]*im_size[1])
        # print(im_size[0]*im_size[1])
        sub_data = new_im[ran[0:10000]]

        for i in range (10000):
                data_bov.append(new_im[ran[i]])
        # print("new data")
        # data_bov.append(sub_data)
        # count += 1



mycluster = KMeans(n_clusters=16) # 128 colors
mycluster.fit(data_bov)

X = []
y = []

for img_dir in glob.glob('leedsbutterfly/images'+'/*.png'):
        # print("count", count, "label", label, img_dir)
        im = cv2.imread(img_dir)
        im_size= np.shape(im)
        # im = im.reshape((im_size[0]*im_size[1]*3))
        # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # X = np.append(X, im, axis=0)
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
        x = im_seg.reshape((im_size[0]*im_size[1]*3))
        x = x / 255
        x = x.astype(np.uint8)
        x = x.reshape((np.shape(im_seg)[0],np.shape(im_seg)[1],3))
        
        new_im = x*im
        new_im = new_im.reshape((np.shape(im_seg)[0]*np.shape(im_seg)[1],3))
        # cv2.imshow("seg", new_im)
        # cv2.waitKey(0)

        new_image_vector = mycluster.predict(new_im)
        histogram, bin_edges = np.histogram(new_image_vector, bins=16)
        
        
        X.append(histogram)
        y.append(label[1][1:3])
        # count += 1

#-------------------------------------------------------------------#
#splitting the dataset into 80% training data and 20% testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=0)

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
clf = svm.SVC(C = 1, kernel = 'linear')
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