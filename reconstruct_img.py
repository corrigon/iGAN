
m glob import glob
from scipy.misc import imread, imsave, imresize
from model_def.dcgan_theano import Model
import numpy as np
import theano
import theano.tensor as T
from lib.theano_utils import floatX
from lib.rng import np_rng
import os
import shutil

#Load model
aa = Model(model_name='handbag_64', model_file='models/handbag_64.dcgan_theano')

#compile f_enc
x_s = T.ftensor4('imgs')
z_tilda_sym = aa.model_P(x_s)
f_enc = theano.function([x_s],z_tilda_sym)

#compile f_dec
s_z = T.fmatrix('z')
gen_img = aa.model_G(s_z)
f_dec = theano.function([s_z], gen_img)


paths = glob('/home/ubuntu/Data/hand_bags/*.jpg')
for path in paths[:10]:
    print path
    shutil.copy(path, 'our_recon/%s'%os.path.basename(path))
    img = imread(path)
    img = imresize(img,(64,64))[None,...]
    t_img = aa.transform(img)
    z = f_enc(t_img)
    x_tilda = f_dec(z)[0]
    imsave('our_recon/%s_recon.jpg'%os.path.basename(path),(x_tilda.transpose(1,2,0)*255).astype('uint8'))

