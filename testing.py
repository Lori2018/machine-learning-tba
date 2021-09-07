import numpy as np
from match import Team, MatchQM
from tba import tba
from keras import models, layers
import tensorflow as tf

data_file = open("training_data.txt", 'r')
raw = data_file.read()
data = raw.split('match')

# number of matches (407), teams per match, 10 features of each team
training_data = np.empty((len(data) - 1, 6, 10))

mtch_cnt = 0
for match in data:
    teams = match.split('\n')
    teams.remove('')
    
    tm_cnt = 0
    for team in teams:
        data = team.split(' ')
        data.remove('') if '' in data else None
        if len(data) > 1:
            features = [float(point) for point in data if point != '']
        if mtch_cnt < 407:
            training_data[mtch_cnt][tm_cnt] = features  
        tm_cnt = tm_cnt + 1 if tm_cnt < 5 else 0
    mtch_cnt += 1

data_file.close()

training_labels = np.loadtxt('training_labels.txt')

def build_model():
    # tf.reshape(training_data, [])
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu',
              input_shape=(training_data.shape[2],)))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

num_epochs = 20
all_mae_histories = []

k = 5
num_val_samples = len(training_data) // k

for i in range(k):
    print('processing fold #', i)
    val_data = training_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_labels = training_labels[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [training_data[:i * num_val_samples],
         training_data[(i + 1) * num_val_samples:]],
         axis=0)
    partial_train_labels = np.concatenate(
        [training_labels[:i * num_val_samples],
         training_labels[(i + 1) * num_val_samples:]],
         axis=0)

    model = build_model()

    history = model.fit(partial_train_data, partial_train_labels,
                        validation_data=(val_data, val_labels),
                        epochs=num_epochs, batch_size=1, verbose=1)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)