#THIS FILE IS NOT PART OF THE SCRIPT YET!!!

import numpy as np

#calculation of coordinates relative basis dimensions that maps the key points of each ID independent
    #from their position in the room
    #the array "arr" (=baseCoordinates) contains 4 points with three coordinates each -> size = (IDs,4,3)
    #arr[0,:] = Y-direction - neck center
    #arr[1,:] = X-direction - l_hip
    #arr[2,:] = center (0,0,0) - perpendicularOnDim1
    #arr[3,:] = Z-direction - crossP
def calculateRelativeBasis(poses_3d):

    baseCoordinates = []
    for IDs in range(poses_3d.shape[0]):
        iDCoordArray = np.empty((0, 3), int)

        # dim1 = Y consists of the line [hip_center, center]
        neck_center = np.array([[poses_3d[IDs, 0, 0], poses_3d[IDs, 0, 1], poses_3d[IDs, 0, 2]]])
        iDCoordArray = np.append(iDCoordArray, neck_center, axis=0)

        # Calculate hip center - output is a (1,3)-matrix - NOT PART OF FINAL ARRAY
        hip_center = ((np.add(poses_3d[IDs, 6, :], poses_3d[IDs, 12, :])) / 2).reshape((1, 3))
        #iDCoordArray = np.append(iDCoordArray, hip_center, axis=0)

        # add r-hip as reference point for Dim2 = X
        l_hip = np.array([poses_3d[IDs, 6, :]])
        iDCoordArray = np.append(iDCoordArray, l_hip, axis=0)

        # determine perpendicular point on line (center, hip_center) to l_hip
        # (this is required to keep the second dimension always perpendicular to the first)
        a = hip_center
        d = np.subtract(neck_center, a)
        p = np.subtract(l_hip, a)

        # need to be turned into a (,3)-array in order to calculate the dot product
        a_squeeze = np.squeeze(np.asarray(a))
        d_squeeze = np.squeeze(np.asarray(d))
        p_squeeze = np.squeeze(np.asarray(p))

        perpendicularOnDim1 = np.array([np.add(((np.dot(p_squeeze, d_squeeze) / np.dot(d_squeeze, d_squeeze)) * d_squeeze), a_squeeze)])
        iDCoordArray = np.append(iDCoordArray, perpendicularOnDim1, axis=0)

        # calculate Dim3 = Z
        aMinusC = np.subtract(neck_center,perpendicularOnDim1)
        bMinusC = np.subtract(l_hip, perpendicularOnDim1)

        a_b = np.subtract(aMinusC, bMinusC)

        crossP = np.cross(a_b, aMinusC)
            #make sure the Z-coordinate is positive, if not -> *-1
        if (crossP[0, 2] < 0):
            crossP = crossP * -1

            #verify perpendicularness
        # crossP_squeeze = np.squeeze(np.asarray(crossP))
        # y_squeeze = np.squeeze(np.asarray(aMinusC))
        # x_squeeze = np.squeeze(np.asarray(bMinusC))
        #
        # y_dotP = np.dot(y_squeeze, crossP_squeeze)
        # x_dotP = np.dot(x_squeeze, crossP_squeeze)
        # xy_dotP = np.dot(x_squeeze, y_squeeze)
        iDCoordArray = np.append(iDCoordArray, crossP, axis=0)
        baseCoordinates.append(iDCoordArray)
    arr = np.array(baseCoordinates)
    return arr, perpendicularOnDim1, crossP
