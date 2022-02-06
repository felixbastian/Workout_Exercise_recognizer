import numpy as np

global confirm
confirm = -1
global count
count = 0
global exerciseCount
exerciseCount = 0
global currentExercise
currentExercise = -1
global potentialChange
potentialChange = False

global armsFrontReps
global armsHorizontalReps
global armsSideReps
global armsSkyReps
global toBeAllocated

armsFrontReps = 0
armsHorizontalReps = 0
armsSideReps = 0
armsSkyReps = 0
toBeAllocated = [0,0,0,0]


def reps(predictionIndex, shoulderAngles):
    global count
    global confirm
    global exerciseCount
    global potentialChange

    global armsFrontReps
    global armsHorizontalReps
    global armsSideReps
    global armsSkyReps

    row, col = shoulderAngles.shape

    justChanged = False
    #look at beginning row
    print(f"prediction: {predictionIndex}")

    if (confirm != -1 and predictionIndex == confirm):
        potentialChange = False
        row = row-30
        countReps(shoulderAngles, confirm, predictionIndex, row, "confirmed and locked")
        count = 0

    if (confirm != -1 and predictionIndex != confirm):

        if (count ==0):
            potentialChange = True
            row = row - 30
            countReps(shoulderAngles, confirm,predictionIndex, row, "might change soon!")
            count = count + 1

            #to not immediately loop over the following if-statement, we need a specific variable
            justChanged = True

        if (count == 1 and justChanged == False):
            potentialChange = False
            confirm = predictionIndex
            exerciseCount = 0
            count = 0

            row = row - 60
            countReps(shoulderAngles, confirm, predictionIndex,row, "things changed")



    if (confirm ==-1 and predictionIndex==-1):
        return 0

    if (confirm == -1 and predictionIndex !=-1):
        potentialChange = False
        confirm = predictionIndex
        row = row-60
        countReps(shoulderAngles, confirm,predictionIndex, row, "first!")

    return armsFrontReps, armsHorizontalReps, armsSideReps, armsSkyReps

def armsFrontRep(_shoulderAngles, row):
    turnOff = False
    turnOn = False
    counter = 0

    print("welcome")
    for i in range(row):
        print("let me introduce")
        print(f"{_shoulderAngles[i][0]} and {_shoulderAngles[i][3]}")
        if (_shoulderAngles[i][0] <110 and _shoulderAngles[i][3] <110):
            turnOn = True
            print("yes")

        if (turnOn == True and _shoulderAngles[i][0] >150 and _shoulderAngles[i][3] >150):
            turnOff = True
            print("yes but then no")

        if (turnOn ==True and turnOff == True):

            counter = counter + 1
            print(f"True true: {counter}")
            print("")
            turnOn =False
            turnOff = False

    return counter



def armsHorizontalRep(_shoulderAngles, row):
    turnOff = False
    turnOn = False
    counter = 0

    for i in range(row):

        if (_shoulderAngles[i][2] > 100 and _shoulderAngles[i][5] > 100):
            turnOn = True

        if (turnOn == True and _shoulderAngles[i][2] < 50 and _shoulderAngles[i][5] < 50):
            turnOff = True

        if (turnOn == True and turnOff == True):
            counter = counter + 1

            turnOn = False
            turnOff = False

    return counter

def armsSideRep(_shoulderAngles, row):
    turnOff = False
    turnOn = False
    counter = 0

    for i in range(row):

        if (_shoulderAngles[i][0] < 140 and _shoulderAngles[i][3] < 140):
            turnOn = True

        if (turnOn == True and _shoulderAngles[i][0] > 160 and _shoulderAngles[i][3] > 160):
            turnOff = True

        if (turnOn == True and turnOff == True):
            counter = counter + 1

            turnOn = False
            turnOff = False

    return counter

def armsSkyRep(_shoulderAngles, row):
    turnOff = False
    turnOn = False
    counter = 0

    for i in range(row):

        if (_shoulderAngles[i][0] < 30 and _shoulderAngles[i][3] < 30):
            turnOn = True

        if (turnOn == True and _shoulderAngles[i][0] > 120 and _shoulderAngles[i][3] > 120):
            turnOff = True

        if (turnOn == True and turnOff == True):
            counter = counter + 1

            turnOn = False
            turnOff = False

    return counter


def countReps(shoulderAngles, confirm,predictionIndex, row, test):
    global armsFrontReps
    global armsHorizontalReps
    global armsSideReps
    global armsSkyReps
    global potentialChange
    global toBeAllocated
    _shoulderAngles = shoulderAngles[row:]

    #add the repititions (when it's clear which exercise)
    if potentialChange == False:
        if confirm == 0:
            armsFrontReps += armsFrontRep(_shoulderAngles, row) + toBeAllocated[0]
            toBeAllocated = [0,0,0,0]

        elif confirm == 1:
            armsHorizontalReps += armsHorizontalRep(_shoulderAngles, row) + toBeAllocated[1]
            toBeAllocated = [0, 0, 0, 0]


        elif confirm == 2:
            armsSideReps += armsSideRep(_shoulderAngles, row) + toBeAllocated[2]
            toBeAllocated = [0, 0, 0, 0]

        elif confirm == 3:
            armsSkyReps += armsSkyRep(_shoulderAngles, row) + toBeAllocated[3]
            toBeAllocated = [0,0,0,0]

    #When a potential change occurs, count the exercises of the former predicted exercise and the current one
    elif potentialChange == True:

        #counting former predicted exercise
        if (confirm == 0):
            toBeAllocated[0] = armsFrontRep(_shoulderAngles, row)
        elif (confirm == 1):
            toBeAllocated[1] = armsHorizontalRep(_shoulderAngles, row)
        elif (confirm == 2):
            toBeAllocated[2] = armsSideRep(_shoulderAngles, row)
        elif (confirm == 3):
            toBeAllocated[3] = armsSkyRep(_shoulderAngles, row)

        #count current predicted exercise
        if (predictionIndex == 0):
            toBeAllocated[0] = armsFrontRep(_shoulderAngles, row)
        elif (predictionIndex == 1):
            toBeAllocated[1] = armsHorizontalRep(_shoulderAngles, row)
        elif (predictionIndex == 2):
            toBeAllocated[2] = armsSideRep(_shoulderAngles, row)
        elif (predictionIndex == 3):
            toBeAllocated[3] = armsSkyRep(_shoulderAngles, row)


    print(test)
    print(_shoulderAngles.shape)
    print(f"armsFrontReps: {armsFrontReps} + {toBeAllocated[0]}")
    print(f"armsHorizontalReps: {armsHorizontalReps} + {toBeAllocated[1]}")
    print(f"armsSideReps: {armsSideReps} + {toBeAllocated[2]}")
    print(f"armsSkyReps: {armsSkyReps} + {toBeAllocated[3]}")


    return 0