'''Weights visualization'''

from keras.callbacks import Callback
from keras.models import load_model
import matplotlib.pyplot as plt


def _draw_weights(net, filename='out.png'):
    weights = net.get_weights()[0]

    for i in range(weights.shape[-1]):
        w = weights[..., i].reshape(28, 28)
        plt.subplot(2, 5, i + 1)
        plt.imshow(w)
        plt.axis('off')

    plt.savefig(filename, bbox_inches='tight')


class DrawWeights(Callback):
    def __init__(self, net):
        self.net = net
        _draw_weights(self.net, 'imgs/%06d.png' % 0)

    def on_epoch_end(self, epoch, logs):
        _draw_weights(self.net, 'imgs/%06d.png' % (epoch + 1))


if __name__ == '__main__':
    net = load_model('net.h5')
    _draw_weights(net)
