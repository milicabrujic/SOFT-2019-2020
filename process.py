import cv2
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import dlib

from joblib import dump, load
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

    
nbins = 9 # broj binova
cell_size = (8,8) # broj piksela po celiji
block_size = (3, 3) # broj celija po bloku

def train_or_load_model(train_image_paths):
    model = load('svm2.joblib')
    if model != None:
        print('Model postoji')
        return model
    train_X =[]
    labels=[]
    for f in train_image_paths:
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        landmarks = get_landmarks(img)
        try:
            img=img[landmarks[19,1]-30:landmarks[9,1]+10, landmarks[0,0]:landmarks[16,0]]
            img = cv2.resize(img, (100, 100), interpolation = cv2.INTER_AREA)        
        except:
            continue
        hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                  img.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)       
        train_X.append(hog.compute(img))
        print(f)
        labels.append(f[-5])
    
    x = np.array(train_X)
    y = np.array(labels)
    x_train = reshape_data(x)
    print('Train shape: ', x.shape,y.shape)
    clf_svm = SVC(kernel='linear') 
    clf_svm.fit(x_train, y)
    y_train_pred = clf_svm.predict(x_train)
    print("Train accuracy: ", accuracy_score(y, y_train_pred))
    dump(clf_svm, 'svm2.joblib')
    model = clf_svm
    return model
   
def from_image(trained_model, image_path):
        print(image_path)
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)  
        model = trained_model
        r=detector(img,1)
        for i in range(0,len(r)):
            train_X=[]
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)  
            landmarks = np.matrix([[p.x,p.y] for p in predictor(img,r[i]).parts()])
            img=img[landmarks[19,1]-30:landmarks[9,1]+10, landmarks[0,0]:landmarks[16,0]]
            img = cv2.resize(img, (100, 100), interpolation = cv2.INTER_AREA)
            hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                          img.shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)
            train_X.append(hog.compute(img))
            x = np.array(train_X)        
            x_train = reshape_data(x)
            print(model.predict(x_train)[0])
        return 
def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))

def get_landmarks(im):
    r=detector(im,1)
    if len(r)!=1:
        return np.matrix(0)
    return np.matrix([[p.x,p.y] for p in predictor(im,r[0]).parts()])