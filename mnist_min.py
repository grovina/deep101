from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


# Geral
tf.set_random_seed(1)
xavier = tf.contrib.layers.xavier_initializer()

# Dados
mnist = input_data.read_data_sets('.')

# Modelo
x = tf.placeholder(tf.float32, [None, 784])

with tf.name_scope('single'):
    W = tf.Variable(xavier([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

# with tf.name_scope('double'):
#     W = tf.Variable(xavier([784, 64]))
#     b = tf.Variable(tf.zeros([64]))
#     y = tf.nn.relu(tf.matmul(x, W) + b)

#     W = tf.Variable(xavier([64, 10]))
#     b = tf.Variable(tf.zeros([10]))
#     y = tf.matmul(y, W) + b

# Resposta
labels = tf.placeholder(tf.int64, [None])

# Custo
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=y)

# Otimizacao
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Sessao do TensorFlow
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Treino
for _ in range(1000):
    xi, yi = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: xi, labels: yi})

# Teste
right_answer = tf.equal(tf.argmax(y, 1), labels)
accuracy = tf.reduce_mean(tf.cast(right_answer, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, labels: mnist.test.labels}))
