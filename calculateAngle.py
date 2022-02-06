import numpy as np
import math
# calculate angle between vectors
def calculateVectorAngle(vector1, vector2):
    # the two vectors still have to be subtracted by their common starting point in order to calc. degree
    v1_squeeze = np.squeeze(np.asarray(vector1))
    v2_squeeze = np.squeeze(np.asarray(vector2))
    unit_vector_1 = v1_squeeze / np.linalg.norm(v1_squeeze)
    unit_vector_2 = v2_squeeze / np.linalg.norm(v2_squeeze)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angleInDeg = math.degrees(np.arccos(dot_product))
    return angleInDeg


def calculateAngles(relativeBasis, vector, center):
    relativeVector = np.subtract(vector, center)

    # calculate linear combination of basis to reach (e.g. l_elbow) in the order z,x,y
    vec_new = np.linalg.inv(np.array([relativeBasis[3, :], relativeBasis[1, :], relativeBasis[0, :]])).dot(
        relativeVector)

    # calculate angle of Y&X plane (taking z out)
    vectorYX = np.array([0, vec_new[1], vec_new[2]])
    # the angle between Y-Dimension (relativeBasis[0, :]) and the reference vector will be measured
    angleInDegYX = calculateVectorAngle(vectorYX, relativeBasis[0, :])

    # calculate angle of Y&Z plane (taking x out)
    vectorYZ = np.array([vec_new[0], 0, vec_new[2]])
    # the angle between Y-Dimension (relativeBasis[0, :]) and the reference vector will be measured
    angleInDegYZ = calculateVectorAngle(vectorYZ, relativeBasis[0, :])

    # calculate angle of X&Z plane (taking y out)
    vectorXZ = np.array([vec_new[0], vec_new[1], 0])
    # the angle between X-Dimension (relativeBasis[1, :]) and the reference vector will be measured
    angleInDegXZ = calculateVectorAngle(vectorXZ, relativeBasis[1, :])

    # print(angleInDegYX)
    # print(angleInDegYZ)
    # print(angleInDegXZ)

    return angleInDegYX, angleInDegYZ, angleInDegXZ


# norming basis to be located in center and one unit for each axis has same length as one unit in original coord. system
def norm(relativeBasis, center):
    for i in range(relativeBasis.shape[0]):
        relativeBasis[i, :] = np.subtract(relativeBasis[i, :], center)

    # norm basis
    _relativeBasis = []
    for i in range(relativeBasis.shape[0]):
        norm = math.sqrt((relativeBasis[i, 0]) ** 2 + (relativeBasis[i, 1]) ** 2 + (relativeBasis[i, 2]) ** 2)  #

        if (norm != 0):
            # print(np.true_divide(relativeBasis[i, :], norm))
            _relativeBasis.append(np.true_divide(relativeBasis[i, :], norm))
        elif (norm == 0):
            _relativeBasis.append([0, 0, 0])
        # relativeBasis[i,:] = np.divide(relativeBasis[i,:], norm)

    arr = np.array(_relativeBasis)
    return arr


def calculateRelativeCoordinates(poses_3d_reference, basis):
    IDs = poses_3d_reference.shape[0]
    anglesArray = []
    basisArray = []

    for ID in range(IDs):
        ########## l_shoulder
        # send l_shoulder to center
        _basis = np.copy(basis[ID])
        center = np.copy(_basis[2, :])

        # norming the relative basis
        basisArray.append(norm(_basis, center))
        # calculating the angles (between center of rel. basis and e.g. l_elbow) in the relative basis
        anglesArray.append(calculateAngles(basisArray[ID], poses_3d_reference[ID], center))

    return anglesArray
