from model_def.dcgan_theano import Model
from scipy.misc import imsave
import numpy as np
import theano
import theano.tensor as T
from lib.theano_utils import floatX
from lib.rng import np_rng


aa = Model(model_name='shoes_64', model_file='models/shoes_64.dcgan_theano')

z0 = np_rng.uniform(-1., 1., size=(1, 100))
z = floatX(z0)
x_recon = aa._gen(z)
x_tilda = aa.inverse_transform(x_recon, npx=64, nc=3)
imsave('x_tilda.jpg',x_tilda[0])
x_tilda_256 = (x_tilda*255).astype('uint8')
imsave('x_tilda_256.jpg',x_tilda_256[0])

x_tilda_normed = aa.transform(x_tilda_256)

x_s = T.ftensor4('imgs')
z_tilda_sym = aa.model_P(x_s)
f_enc = theano.function([x_s],z_tilda_sym)

z_tilda = f_enc(x_tilda_normed)
x_tilda_recon = aa._gen(z_tilda)
x_tilda_recon = aa.inverse_transform(x_tilda_recon, npx=64, nc=3)
imsave('x_tilda_recon.jpg',x_tilda_recon[0])

s_z = T.fmatrix('z')
gen_img = aa.model_G(s_z)
f_dec = theano.function([s_z], [gen_img])

img1 = f_dec(np.zeros([1,100],dtype='float32'))[0][0]
imsave('foo.jpg',(img1.transpose(1,2,0)*255).astype('uint8'))


imsave('foo3.jpg',aa.gen_samples()[0])


imgs = aa.gen_samples()

