from __future__ import print_function
import numpy as np
import pickle
import sys
import argparse

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json

from keras.optimizers import *
from keras.layers.normalization import BatchNormalization

parser = argparse.ArgumentParser('')
parser.add_argument('--use_stored_model', action='store_true', help='use_stored_model')
args = parser.parse_args()


def load_data(data_file):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    X = []
    y = []
    for key in sorted(data):
        X.append(data[key])
        if key.find('angry') >= 0:
            y.append([1, 0, 0, 0, 0, 0, 0, 0])
        elif key.find('contemptuous') >= 0:
            y.append([0, 1, 0, 0, 0, 0, 0, 0])
        elif key.find('disgusted') >= 0:
            y.append([0, 0, 1, 0, 0, 0, 0, 0])
        elif key.find('fearful') >= 0:
            y.append([0, 0, 0, 1, 0, 0, 0, 0])
        elif key.find('happy') >= 0:
            y.append([0, 0, 0, 0, 1, 0, 0, 0])
        elif key.find('neutral') >= 0:
            y.append([0, 0, 0, 0, 0, 1, 0, 0])
        elif key.find('sad') >= 0:
            y.append([0, 0, 0, 0, 0, 0, 1, 0])
        elif key.find('surprised') >= 0:
            y.append([0, 0, 0, 0, 0, 0, 0, 1])
        else:
            print('wrong image!')
            sys.exit()

    X, y = np.array(X), np.array(y)
    print(X.shape, y.shape)

    return X, y, data


# Main CNN model with four Convolution layer & two fully connected layer
def baseline_model():
    # Initialising the CNN
    model = Sequential()

    # 1 - Convolution
    model.add(Conv2D(64, 3, padding='same', input_shape=(128, 128, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2nd Convolution layer
    model.add(Conv2D(128, 5, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 3rd Convolution layer
    model.add(Conv2D(512, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 4th Convolution layer
    model.add(Conv2D(512, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 5th Convolution layer
    model.add(Conv2D(512, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    # Flattening
    model.add(Flatten())

    # Fully connected layer 1st layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))


    # Fully connected layer 2nd layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(num_class, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])

    return model


def baseline_model_saved():
    # load json and create model
    json_file = open('my_model_ep17.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights from h5 file
    model.load_weights("my_model_weights_ep17.h5")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])
    return model


if __name__ == '__main__':

    batch_size = 128
    epochs = 50
    label_map = ['angry', 'contemptuous', 'disgusted', 'fearful', 'happy',
                 'neutral', 'sad', 'surprised']
    num_class = len(label_map)

    print('--------------------------loading data-------------------------------')
    X_train, y_train, data_matrix = load_data('img_data.pkl')

    is_model_saved = args.use_stored_model
    # If model is not saved train the CNN model otherwise just load the weights
    if is_model_saved:
        # Load the trained model
        print("Load model from disk")
        model = baseline_model_saved()
    else:
        print('-----------------------Training model----------------------------')
        model = baseline_model()
        # Note : 3259 samples is used as validation data &   28,709  as training samples

        model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  validation_split=0.1111)
        print('-----------------------Training ended----------------------------')
        model_json = model.to_json()
        with open("my_model_ep50.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("my_model_weights_ep50.h5")
        print("Saved model to disk")

    # Model will predict the probability values for 8 labels for a test image
    score = model.predict(X_train)
    model.summary()
    print('score: ', score.shape, score[0:5])
    s = np.sum(score, axis=1)
    print('sum: ', s.shape, s[0:5])
    classes = [np.argmax(item) for item in score]
    print('classes: ', len(classes), classes[0:5])

    con_data = data_matrix
    print('aaaaa: ', len(con_data), con_data['Rafd090_61_Caucasian_female_disgusted_frontal'])
    i = 0
    for key, cla, con in zip(sorted(con_data), classes, score):
        if i < 5:
            print(key, label_map[cla], con)
        con_data[key] = con
        i = i+1
    print('bbbbb: ', len(con_data), con_data['Rafd090_61_Caucasian_female_disgusted_frontal'])

    with open('my_aus_50.pkl', 'wb') as f:
        pickle.dump(con_data, f, pickle.HIGHEST_PROTOCOL)

    y_label = [np.argmax(item) for item in y_train]
    # Calculating categorical accuracy taking label having highest probability
    accuracy = [(x == y) for x, y in zip(classes, y_label)]
    print(" Accuracy on Test set : ", np.mean(accuracy), len(accuracy))

