from tba import tba
import numpy as np
from match import Team, MatchQM
import keras
keras.__version__
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

training_labels = np.loadtxt('training_labels.txt')
data_file = open("training_data.txt", 'w+')

print(training_data)

# making a plot (basically see if our data is how we want it)
y = train_targets
x = train_data[:,0] 
plt.plot(x,y,'bo')
plt.xlabel('Per Capita Crime Rate') 
plt.ylabel('Price of House (in kilo$)') 
plt.title('Check if Price Depends on Per Capita Crime Rate') 
plt.show()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1)) 
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# k fold validation (idk if we need this but probably)
num_epochs = 200 # we probably don't need 200 epochs?
all_mae_histories = [] 

k = 3
num_val_samples = len(train_data) // k 

for i in range(k): 
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
plt.plot(average_mae_history)
#plt.plot(average_mae_history[10:])
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

max_epoch = 50 # the number here should be the number of epochs where the validation MAE starts overfitting
model = build_model()
# Train it on the entirety of the data.
model.fit(train_data, train_targets,
          epochs=max_epoch, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score)

# prediction vs target graph - there's more in the original code if we want 
predictions = model.predict(test_data)
plt.figure(figsize=(18,5))
plt.plot(predictions[:,0],'r.',label='Predictions')
plt.plot(test_targets,'bx',label='Targets')
plt.ylabel('Cost of House (k$)')
plt.legend()
plt.show()

new_mae = []  # Set up an empty array to keep the results

for i in np.arange(13): # Loop through the inputs
  permutation = np.copy(test_data) # Make a copy of the test data
  permutation[:,i] = 0.0 # Set a particular column of inputs to zero
  p_mse, p_mae = model.evaluate(permutation, test_targets) # evaluate the modified data
  new_mae.append(p_mae-test_mae_score) # save the change in MAE to the array

plt.bar(np.arange(13)+1,new_mae) # Make a bar graph to show change in result
plt.ylabel('Change in MAE')
plt.xlabel('Parameter Number (see list above)')
plt.show()