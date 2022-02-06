# import machine learning model
import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import countRepetitions

# filepath = './saved_instance'
# model = load_model(filepath, compile=True)
model = load_model('new_safe2_NEW_NEW.model')

global prediction
prediction = -1

global armsFrontReps
global armsSideReps
global armsHorizontalReps
global armsSkyReps

armsFrontReps = 0
armsHorizontalReps = 0
armsSideReps = 0
armsSkyReps = 0


print(model.summary())

#shoulderAngles gives full array, shoulderAngles[x] gets the 6 angles of row x, shoulderAngles[x][y] gets the yth Angle
#of row x

def predict(shoulderAngles):

    row, col = shoulderAngles.shape
    global armsFrontReps
    global armsSideReps
    global armsHorizontalReps
    global armsSkyReps

    global prediction

    if (row==60):
        predictArray = shoulderAngles[row-60:row]
        prediction, armsFrontReps, armsHorizontalReps, armsSideReps, armsSkyReps = makePrediction(predictArray)

    elif(row != 30 and row%30==0):
        predictArray = shoulderAngles[row - 60:row]
        prediction, armsFrontReps, armsHorizontalReps, armsSideReps, armsSkyReps = makePrediction(predictArray)


    return prediction, armsFrontReps, armsHorizontalReps, armsSideReps, armsSkyReps

def makePrediction(predictArray):

    #using the same scaler applied in the machine learning model
    scaler = joblib.load("scaler.save")

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1 -  only use transform and not fit_transform
    #https://towardsdatascience.com/
    # what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe
    #fit method is calculating mean and variance and transform is actually transforming

    X = scaler.transform(predictArray)

    np.savetxt("predicted_scaled_output_2.csv", X, delimiter=",", fmt='%s')

    _predictArray = np.reshape(X, (1, 60, 6, 1))

    predictions = model.predict([_predictArray])
    classIndex = np.argmax(predictions)

    armsFrontReps, armsHorizontalReps, armsSideReps, armsSkyReps = countRepetitions.reps(classIndex, predictArray)

    #classes = 0


    return classIndex, armsFrontReps, armsHorizontalReps, armsSideReps, armsSkyReps