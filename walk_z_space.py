
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
dim_samples = 11
dim_radius = 0.5
dimensions = 100
for path in paths[:10]:
    print path
    out_path = 'walk_z/'+os.path.basename(path)
    os.mkdir(out_path)
    shutil.copy(path, out_path+'/orig_'+os.path.basename(path))
    img = imread(path)
    img = imresize(img,(64,64))[None,...]
    t_img = aa.transform(img)
    z_orig = f_enc(t_img)
    x_tilda = f_dec(z_orig)[0]
    imsave(out_path+'/reconstructed_%s'%(os.path.basename(path)),(x_tilda.transpose(1,2,0)*255).astype('uint8'))
    for i in range(dimensions):
        out_dim_path = out_path+'/%d'%i
        print out_dim_path
        os.mkdir(out_dim_path)
        v = z_orig[0, i]
        v = max(-1.0+dim_radius, min(1-dim_radius,v))
        samples = np.linspace(max(-1.0, v-dim_radius), min(1, v+dim_radius), dim_samples)
        z = z_orig.copy()
        for j in range(dim_samples):
            z[0, i] = samples[j]
            x_tilda = f_dec(z)[0]
            imsave(out_dim_path+'/%02d_%s'%(j,os.path.basename(path)),(x_tilda.transpose(1,2,0)*255).astype('uint8'))
