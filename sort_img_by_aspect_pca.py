################### sort images by aspect ################
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
from sklearn.decomposition import PCA

#Load model
aa = Model(model_name='handbag_64', model_file='models/handbag_64.dcgan_theano')

#compile f_enc
x_s = T.ftensor4('imgs')
z_tilda_sym = aa.model_P(x_s)
f_enc = theano.function([x_s],z_tilda_sym)

paths = glob('/home/ubuntu/Data/hand_bags/*.jpg')[:1000]
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
#    shutil.copy(path, 'our_recon/%s'%os.path.basename(path))

z = np.vstack(z_arr) #1000x100
paths = np.asarray(paths)
pca = PCA(n_components=100)
pca.fit(z)
print(pca.explained_variance_ratio_)
z_tr = pca.fit_transform(z)
for i in range(10):
    col_feats = z_tr[:,i]
    idxs = np.argsort(col_feats)
    print 'aspect %d'%i
    base_dir = 'aspect_sorted_pca/%d'%i
    os.mkdir(base_dir)
    for k,j in enumerate(idxs):
        shutil.copy(paths[j], '%s/%03d_%s'%(base_dir, k, os.path.basename(paths[j])))
