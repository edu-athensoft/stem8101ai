import pandas as pd
from sklearn.model_selection import train_test_split

from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Flatten
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X.reshape(60000, 28, 28, 1)
test_X = test_X.reshape(10000, 28, 28, 1)

print(train_X.shape, test_X.shape)

inputs_1 = Input(shape=(28, 28, 1))
x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs_1)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
x = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
x = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
x = Flatten()(x)
x = Dense(50, kernel_initializer='he_normal', activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(10, activation='softmax')(x)
model = Model([inputs_1], x)
model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, shuffle=True, test_size=0.1)

batch_size = 64
epochs = 20

train_y = pd.get_dummies(train_y)
test_y = pd.get_dummies(test_y)
valid_y = pd.get_dummies(valid_y)

callback = [EarlyStopping(monitor='val_loss', patience=1),
            ReduceLROnPlateau(patience=1, verbose=0),
            ModelCheckpoint('model_cnn_digital_recog.h5', save_best_only=True, verbose=0),
            ]
model.fit(x=train_X, y=train_y,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(valid_X, valid_y),
          callbacks=callback,
          verbose=1
          )
