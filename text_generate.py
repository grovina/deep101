import pickle as pk

from keras.models import load_model
from keras.utils import to_categorical
import numpy as np


length = 5

net = load_model('text.h5')

chars = pk.load(open('text.pk', 'rb'))
char_to_int = {c: i for i, c in enumerate(chars)}

text = 'Impreſſos'.lower()
for _ in range(1000):
    word = text[-length:]
    x = to_categorical([[char_to_int[w] for w in word]], len(chars))
    y = net.predict(x, verbose=0)[0]
    char = np.random.choice(chars, p=y)
    # char = chars[np.argmax(y)]
    text += char

print(text)
