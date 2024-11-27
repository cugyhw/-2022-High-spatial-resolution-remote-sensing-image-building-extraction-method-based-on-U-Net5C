import cv2
import numpy as np
import os
from time import time


start = time()
predict_path = r"D:\Deep Learning Code\predict"
label_path = r"D:\Deep Learning Code\label"

files = os.listdir(label_path)
list1 = []

for file in files:
    result_name = os.path.join(label_path,file)
    list1.append(result_name)

files = os.listdir(predict_path)
list2 = []
print("lable_len:",len(files))
for file in files:
    result_name = os.path.join(predict_path,file)
    list2.append(result_name)
print("predict_len:",len(files))
Recall = 0
Precision = 0
F1 = 0
for i in range(len(files)):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    img_lable = cv2.imread(list1[i],1)
    img_predict = cv2.imread(list2[i],1)
    x = np.array(img_lable)
    for j in range(x.shape[0]):
        index = np.arange(0, x.shape[1])
        a = index[img_lable[j]==img_predict[j]]
        for k in range(len(a)):
            if img_lable[j][a[k]] == 255:
                TP += 1
            else :
                TN += 1
        b = index[img_lable[j] != img_predict[j]]
        for k in range(len(b)):
            if img_lable[j][b[k]] == 255:
                FN += 1
            else :
                FP += 1
    Precision += TP/(TP+FP)
    Recall += TP/(TP+FN)
    F1 += (2 * (TP/(TP+FP)) * (TP/(TP+FN))/((TP/(TP+FP)) + (TP/(TP+FN))))
    print("第",i+1,"次计算完成")
end = time()
print('running time is :%f'%(end-start))
print("Precision:" ,Precision/len(files),"Recall:",Recall/len(files),"F1:",F1/len(files))
