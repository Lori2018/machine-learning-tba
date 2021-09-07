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

print(data_file)
print(data_file.shape)
print(training_labels.shape)
