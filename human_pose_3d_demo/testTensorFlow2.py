# Load the model
from tensorflow.keras.models import Sequential, save_model, load_model
import matplotlib.pyplot as plt
import numpy as np

#input_train = np.load('test_arr.npy', allow_pickle=True)
input_train = np.load('liste.npy', allow_pickle=True)

print("hi")
print(input_train[0].shape)

filepath = './saved_model_test'
model = load_model(filepath, compile = True)

# A few random samples
use_samples = [5, 38, 3939, 27389]
samples_to_predict = []

# Generate plots for samples
for x in range(4):
  # Generate a plot
  reshaped_image = input_train[x].reshape((28, 28))
  plt.imshow(reshaped_image)
  plt.show()
  # Add sample to array for prediction
  samples_to_predict.append(input_train[x])

# Convert into Numpy array
samples_to_predict = np.array(samples_to_predict)
print("to predict shape")
print(samples_to_predict.shape)

# Generate predictions for samples
predictions = model.predict(samples_to_predict)
print(predictions)

# Generate arg maxes for predictions
classes = np.argmax(predictions, axis = 1)
print(classes)