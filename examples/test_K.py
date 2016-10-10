'''This example demonstrates the use of Convolution1D for text classification.

Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.

'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras import backend as K
#from keras.layers.core import K
#import tensorflow as tf
#K._LEARNING_PHASE = tf.constant(0) # try with 1

# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 2

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(X_train[0:3])
print(len(X_test), 'test sequences')
print(X_test[0:3])

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
X_train= X_train[0:1000]
X_test= X_test[0:500]
y_train= y_train[0:1000]
y_test= y_test[0:500]
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
# X_train shape: (25000, 400)
# X_test shape: (25000, 400)



print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
# embedding_dims = 50, each word is mapped to an integer standing for the index among the 5000 words
# then the integer is mapped to length-50 vector ...
# max_features=5000, use the top-5000 words
#  input_length=maxlen, limit the number words up to 400 per sentence/sequence/review
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen,
                    dropout=0.2))

model.summary()


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

          
# after model.fit()
# rowX=X_test[0]
# xxx preds = model.predict_classes(rowX, verbose=0)
#preds = model.predict_classes(X_test, verbose=0)
# K.set_learning_phase(0) 
get_layer_output = K.function([model.input, K.learning_phase()],  [model.output] )
layer_output = get_layer_output([ X_test[0:batch_size],0 ])  #xxx , keras_learning_phase='bob')
print('embedding layer op shape = ', layer_output[0].shape)
print("embedding layer op[0]'s the first 3 rows = ", layer_output[0][0:2])
