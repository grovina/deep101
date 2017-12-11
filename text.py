'''Vanilla Text generation net with keras'''

import pickle as pk

from keras.callbacks import TensorBoard
from keras.layers import Input, LSTM, Dense, Activation
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical
import numpy as np


LENGTH = 100

# # # # # # # #
#  D A D O S  #
# # # # # # # #

# carregamos todo o texto
text = open('data/lusiadas.txt').read()

# passamos para minusculas
text = text.lower()

# detectamos os caracteres e mapeamos para int
chars = sorted(set(text))
char_to_int = {c: i for i, c in enumerate(chars)}
pk.dump(chars, open('text.pk', 'wb'))

# geramos os exemplos "sequencia -> proxima letra"
x, y = [], []
word = text[:LENGTH]
for letter in text[LENGTH:]:
    x.append([char_to_int[w] for w in word])
    y.append(char_to_int[letter])
    word = word[1:] + letter

# passamos
x = to_categorical(x, len(chars))
y = to_categorical(y, len(chars))

# embaralhamos o dataset
i = np.random.permutation(len(x))
x, y = x[i], y[i]

# separamos em treino e teste
i = int(.8 * len(x))
x_train, x_test = x[:i], x[i:]
y_train, y_test = y[:i], y[i:]



# # # # # # # # #
#  M O D E L O  #
# # # # # # # # #

# camada de entrada (compatível com a forma de x)
out = entry = Input(shape=x_train.shape[1:])

# camada de memória
out = LSTM(256)(out)

# camada de saída com um neurônio para cada caractere
out = Dense(y_train.shape[1])(out)
# aplicação do softmax para obtermos uma distribuição de probabilidade
out = Activation('softmax')(out)

# definição do modelo em si
net = Model(entry, out)

# imprimimos a descrição do modelo
net.summary()



# # # # # # # # # # # # # #
#  T R E I N A M E N T O  #
# # # # # # # # # # # # # #

# definição do custo e da otimização
# otimizador é a descida de gradientes estocástica
from keras.models import load_model
net = load_model('text.h5')
net.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(lr=0.005, momentum=0.9, nesterov=True),
    metrics=['accuracy'])


net.fit(
    x_train, y_train,
    batch_size=512,
    epochs=10,
    validation_data=(x_test, y_test),
    callbacks=[TensorBoard()])

net.save('text.h5')
