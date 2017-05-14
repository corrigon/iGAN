from glob import glob
from scipy.misc import imread, imsave, imresize
from model_def.dcgan_theano import Model
import numpy as np
import theano
import theano.tensor as T
from lib.theano_utils import floatX
from lib.rng import np_rng
import os
import shutil
from sklearn.model_selection import train_test_split



#Load model
aa = Model(model_name='handbag_64', model_file='models/handbag_64.dcgan_theano')

#compile f_enc
x_s = T.ftensor4('imgs')
z_tilda_sym = aa.model_P(x_s)
f_enc = theano.function([x_s],z_tilda_sym)

def get_feats(glob_path):
    paths = glob(glob_path)
    z_arr = []
    for path in paths:
        print path
        img = imread(path)
        if img.ndim == 2:
            img = np.repeat(img[:,:,None],3,2)
        img = imresize(img,(64,64))[None,...]
        t_img = aa.transform(img)
        z = f_enc(t_img)
        z_arr.append(z)

    z = np.vstack(z_arr) #1000x100
    return z

z1 = get_feats('/home/ubuntu/Data/bag_446_coach/*')
lbl_1 = np.zeros((len(z1)))
z2 = get_feats('/home/ubuntu/Data/bag_446_tory/*')
lbl_2 = np.ones((len(z2)))

X = np.vstack([z1,z2])
Y = np.hstack([lbl_1, lbl_2])


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier

cross_validate_count = 30
models = []
models.append(KNeighborsClassifier(n_neighbors=3, metric='l2', algorithm='brute'))
models.append(RandomForestClassifier(max_depth=5))
models.append(RidgeClassifier())
for model in models:
    print 'model '+str(type(model))
    for train_size in [10,20,40,80,100,200]:
        accuracy = 0
        for run in range(cross_validate_count):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size)

            model = KNeighborsClassifier(n_neighbors=3, metric='l2', algorithm='brute')
            model = RandomForestClassifier(max_depth=5)
            #model = RidgeClassifier()
            model.fit(X_train, Y_train)

            preds = model.predict(X_test)
            accuracy = accuracy + accuracy_score(Y_test, preds)
        print 'accuracy with train size %d: %f'%(train_size, accuracy / cross_validate_count)
