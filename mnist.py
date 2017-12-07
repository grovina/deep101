'''Vanilla MNIST net with keras'''

from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Input, Flatten, Dense, Activation
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical


# # # # # # # #
#  D A D O S  #
# # # # # # # #

# carregamos os dados já embaralhados divididos em train e test
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# passamos as entradas pra float (pra poder manipular)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# passamos os labels pra one-hot encoding (vetor 10-dimensional)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# pré-tratamento
# como temos um intervalo definido de entrada [0, 255], simplesmente
# mudamos a escala para [0, 1], o que é bem comum de se fazer em imagens
x_train /= 255
x_test /= 255



# # # # # # # # #
#  M O D E L O  #
# # # # # # # # #

# camada de entrada (compatível com a forma de x)
out = entry = Input(shape=x_train.shape[1:])
# esticamos a entrada num vetor linear (unidimensional)
# isso é necessário para podermos aplicar uma camada densa
out = Flatten()(out)

# camada de saída com 10 neurônios, cada um responsável por um dígito
out = Dense(10)(out)
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
# custo é a distância euclidiana (norma 2) entre saída e resposta
# otimizador é a descida de gradientes estocástica
net.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
    metrics=['accuracy'])

net.fit(
    x_train, y_train,
    batch_size=60,
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[TensorBoard()])

net.save('net.h5')
