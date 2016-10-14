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
import mytime

# set parameters:
max_features = 5000 # number of different words
maxlen = 400 #number of words in a sentence/review, time_step_size
batch_size = 32
embedding_dims = 50 # one word to length-50 vector
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
#X_train= X_train[0:1000]
#X_test= X_test[0:500]
#y_train= y_train[0:1000]
#y_test= y_test[0:500]
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
# X_train shape: (25000, 400)
# X_test shape: (25000, 400)

print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

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

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
# one input to 50(nb_filter) output, each output the result of  conv filter length 3 applied to the input
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# we use max pooling:
model.add(MaxPooling1D(pool_length=model.output_shape[1]))

# We flatten the output of the conv layer,
# so that we can add a vanilla dense layer:
model.add(Flatten())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))
          

