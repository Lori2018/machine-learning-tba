import numpy as np
from match import Team, MatchQM
from tba import tba
from keras import models, layers

data_file = open("training_data.txt", 'r')
raw = data_file.read()
data = raw.split('match')

# number of matches (407), teams per match, 10 features of each team
# UPDATE - flat input so (407, 60)
training_data = np.empty((len(data) - 1, 9 * 6))

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
            updated_features = np.empty(9)
            # normalize data
            updated_features[0] = features[1] / 7
            updated_features[1] = features[2] / 4
            updated_features[2] = features[3] / 10
            updated_features[3] = features[4] / 30
            updated_features[4] = features[5] / 45
            updated_features[5] = features[6] / 3
            updated_features[6] = features[7] / 3
            updated_features[7] = features[8] / 10
            updated_features[8] = abs(features[9]) / 15
        if mtch_cnt < 407:
            if tm_cnt < 6:
                np.append(training_data[mtch_cnt], updated_features)
            else: 
                tm_cnt = 0
                mtch_cnt += 1
    mtch_cnt += 1

data_file.close()

training_labels = np.loadtxt('training_labels.txt')

mean = training_data.mean(axis=0)
training_data -= mean
std = training_data.std(axis=0)
training_data /= std
print(training_data[0])

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(9, activation='relu',
              input_shape=(training_data.shape[1],)))
    # model.add(layers.Dense(18, activation='relu'))
    # model.add(layers.Dense(9, activation='relu'))
    # model.add(layers.Dense(3, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mae', metrics=['mse'])

    return model

num_epochs = 6
all_mae_histories = []

k = 4
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
                        epochs=num_epochs, batch_size = 1, validation_data=(val_data, val_labels))

predictions = model.predict(training_data)
test_mse_score, test_mae_score = model.evaluate(val_data, val_labels)
print(predictions)