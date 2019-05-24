import tensorflow as tf
from tensorflow import keras
import numpy as np


features = np.load('features.npy')
train_f = features[0:15000, :]
test_f = features[15000:20000, :]
targets = np.load('targets.npy')
train_t = targets[0:15000]
test_t = targets[15000:20000]


model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(2,)),
    keras.layers.Dense(3, activation=tf.nn.tanh),
    keras.layers.Dense(2, activation=tf.nn.tanh),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])


model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

inp = model.input
outputs = [layer.output for layer in model.layers]
functor = keras.backend.function([inp, keras.backend.learning_phase()], outputs)


N_SHOW = 100
test = test_f[0:N_SHOW, :]
col = test_t[0:N_SHOW]


transformed_data = []
loss = []
acc = []
for j in range(60):
    model.fit(train_f, train_t, epochs=1, verbose=False)
    layer_outs = functor([test, 1.])
    transformed_data.append(layer_outs)

    test_loss, test_acc = model.evaluate(test_f, test_t)
    loss.append(test_loss)
    acc.append(test_acc)

    print(j, test_loss)


np.save('nn_trans_evol3', [test, transformed_data, col, loss, acc])

test_loss, test_acc = model.evaluate(test_f, test_t)
print('Test accuracy:', test_acc)
