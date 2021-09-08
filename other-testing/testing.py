import numpy as np
from match import Team, MatchQM
from tba import tba
from keras import models, layers
import tensorflow as tf

data_file = open("training_data.txt", 'r')
raw = data_file.read()
data = raw.split('match')
data_file_1 = open("scale_data.txt", "w+")

# number of matches (407), teams per match, 10 features of each team
# UPDATE - flat input so (407, 60)
training_data = np.empty((len(data) - 1, 10 * 6))

mtch_cnt = 0
training_labels = np.loadtxt('training_labels.txt')
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
            if tm_cnt < 6:
                np.append(training_data[mtch_cnt], features)
                data_file_1.write(str(features[5]) + " " + str(training_labels[mtch_cnt]) + "\n")
            else: 
                tm_cnt = 0
                mtch_cnt += 1
        # tm_cnt = tm_cnt + 1 if tm_cnt < 5 else 0
    mtch_cnt += 1

data_file.close()

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu',
              input_shape=(training_data.shape[1],)))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['val_mae'])
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

    # history = model.fit(partial_train_data, partial_train_labels,
    #                     validation_data=(val_data, val_labels),
    #                     epochs=num_epochs, batch_size=1, verbose=0)
    history = model.fit(partial_train_data, partial_train_labels,
                        epochs=num_epochs, batch_size = 1, validation_data=(val_data, val_labels))