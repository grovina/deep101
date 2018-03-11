'''Attention visualization'''

from keras.callbacks import Callback
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

net = load_model('net.h5')

img_input = net.inputs[0]
out = net.outputs[0]


# [0  0 0 0 0 0 0 0  0  1  0]
# [.1 .2 .1 .5 0 0 0 0  X .1]
loss = -K.log(1 - out[..., 8])
grad = K.gradients(loss, img_input)

get_grad = K.function([img_input], grad)

img = np.random.rand(1, 28, 28)

for _ in range(1000):
  img_grad = get_grad([img])[0]
  img_grad /= (np.linalg.norm(img_grad) + .00001)
  img_grad -= img_grad.min()
  img_grad /= img_grad.max()
  print(img.mean(), img_grad.mean())
  img -= 1. * img_grad

plt.imshow(img[0])
plt.show()
