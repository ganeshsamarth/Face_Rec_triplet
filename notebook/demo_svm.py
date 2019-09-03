import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from imageio import imread
from skimage.transform import resize
from keras.models import load_model


image_dir_basepath = '/home/ganesh/Downloads/IMFDB_Align'
names = os.listdir(image_dir_basepath)
image_size = 160
model_path = '/home/ganesh/keras-facenet/model/keras/facenet_keras.h5'
model = load_model(model_path)

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def load_and_align_images(filepaths, margin):
    #cascade = cv2.CascadeClassifier(cascade_path)
    
    aligned_images = []
    for filepath in filepaths:
        img = cv2.imread(filepath)
        img = cv2.resize(img,(160,160))
        
        '''
        faces = cascade.detectMultiScale(img,
                                         scaleFactor=1.1,
                                         minNeighbors=3)
        (x, y, w, h) = faces[0]
        cropped = img[y-margin//2:y+h+margin//2,
                      x-margin//2:x+w+margin//2, :]
        aligned = resize(cropped, (image_size, image_size), mode='reflect')
        aligned_images.append(aligned)
        '''
        #print(np.array(img).shape)
        aligned_images.append(np.array(img))
            
    return np.array(aligned_images)

def calc_embs(filepaths, margin=10, batch_size=1):
    aligned_images = prewhiten(load_and_align_images(filepaths, margin))
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))
    #print(embs.shape)
    return embs

def train(dir_basepath, names, max_num_img=20):
    labels = []
    embs = []
    for name in names:
        print(name)
        dirpath = os.path.join(dir_basepath, name)
        filepaths = [os.path.join(dirpath, f) for f in os.listdir(dirpath)][:max_num_img]
        embs_ = calc_embs(filepaths)    
        labels.extend([name] * len(embs_))
        embs.append(embs_)
        
    embs = np.concatenate(embs)
    le = LabelEncoder().fit(labels)
    y = le.transform(labels)
    clf = SVC(kernel='linear', probability=True).fit(embs, y)
    return le, clf

def infer(le, clf, filepaths):
    embs = calc_embs(filepaths)
    #pred = le.inverse_transform(clf.predict(embs))
    return embs

le, clf = train(image_dir_basepath, names)
acc=list()
'''
for names in os.listdir(image_dir_basepath):
    test_file_paths = [os.path.join(image_dir_basepath+'/'+names,images) for images in os.listdir(image_dir_basepath+'/'+names)]
    pred = infer(le, clf, test_file_paths[:1])
    count=0
    #print(pred)
    for items in pred:
        if items==names:
            count+=1
    print(names+':'+ str(count/len(pred)))
    acc.append(count/len(pred))
print(sum(acc)/len(acc))
'''

with open('/home/ganesh/Desktop/yaw_values.txt','r') as myfile:
    data=myfile.read()
    yaw_data_list=list()
    actor_list=list()
    frontal_list=list()
    profile_list=list()
    x=data.strip().split('\n')
    for i in range(len(x)):
        y=x[i]
        z=y.split(':')
        actor=z[0].split('_')[0]
        yaw=z[1]
        if abs(float(yaw))<=20:
            frontal_list.append(z)
        elif abs(float(yaw))>=45:
            profile_list.append(z)
        actor_list.append(actor)
    myfile.close()

paired_images_list=list()
x=np.empty((0,128))
y=np.empty((0,128))
filepath_infer_1=list()
filepath_infer_2=list()
for actors in actor_list:
    #print(len(frontal_list))
    actor_dir= image_dir_basepath + '/' + actors
    print(actors)
    res1=list()
    res2=list()
    for i,items in enumerate(frontal_list):
        if (actors in items[0]):
            print(items[0])
            res1.append(items)
    for j,items2 in enumerate(profile_list):
        if(actors in items2[0]):
            res2.append(items2)
    print(res1)
    for k in range(2):
        img=actor_dir +'/' + res1[k][0]
        print(img)
        img2=actor_dir +'/'+ res2[k][0]
        print(img2)
        filepath_infer_1.append(img)
        filepath_infer_2.append(img2)
    y=np.append(y,infer(le,clf,filepath_infer_1),axis=0)
    x=np.append(x,infer(le,clf,filepath_infer_2),axis=0)

