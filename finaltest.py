from tensorflow.keras.models import load_model

import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report

ts = pd.read_csv("test_set.csv", index_col=False)
ts_x = ts.drop('result', axis=1)
ts_y = ts['result']  

# Load the model from the file
best_model = load_model('best_model.h5')
print(best_model.summary())

test_loss, test_accuracy = best_model.evaluate(ts_x, ts_y, verbose=1)

# Print the test accuracy
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")