import numpy as np
import math
np.random.seed(2021)



relBasis = np.random.randint(10, size=(2,4,3))
poses_3d = np.random.randint(10, size =(2, 5, 3 ))
print("relBasis")
print(relBasis)

l_shoulderBasis = np.empty_like (relBasis)
l_shoulderBasis[:] = relBasis

r_shoulderBasis = np.empty_like (relBasis)
r_shoulderBasis[:] = relBasis


baseList = [l_shoulderBasis, r_shoulderBasis]

translationBasis = np.random.randint(10, size=(2,3,3))
print("translationBasis")
print(translationBasis)
rotationBasis = np.array([[-1,1,1],[-1,1,1]])

def sendToCenter(relativeBasis):
   for i in range(relativeBasis.shape[1]):
       relativeBasis[i,:] = np.subtract(relativeBasis[i,:], relativeBasis[2,:])
   return relativeBasis

def norm(relativeBasis):

   for i in [x for x in range(relativeBasis.shape[1]) if x != 2]:
       norm = math.sqrt((relativeBasis[i,0])**2 +(relativeBasis[i,1])**2 + (relativeBasis[i,2])**2)#

       print("norm")
       print(norm)
       print("rel")
       print(relativeBasis[i, :])
       print("theroy")

       print(np.divide(relativeBasis[i,:], norm))
       relativeBasis[i,:] = np.divide(relativeBasis[i,:], norm)
       print(relativeBasis[i,:])


   print("normed basis")
   print(relativeBasis)

   return relativeBasis

def calculateRelativeCoordinates(poses_3d,l_shoulderBasis):
   for ID in range(poses_3d.shape[0]):
       l_shoulderBasis[ID] = sendToCenter(l_shoulderBasis[ID])
       l_shoulderBasis[ID] = norm(l_shoulderBasis[ID])



   return 0

def changeAxisOrientation(relativeBasis, axis):
    if (axis == "X"): a=1
    elif (axis == "Y"): a =0
    elif (axis == "Z"): a = 3

    for ID in range(relativeBasis.shape[0]):

        relativeBasis[ID, a] = np.subtract(relativeBasis[ID,2,:], np.subtract(relativeBasis[ID,a,:], relativeBasis[ID,2,:]))


    return relativeBasis

for i in range(len(baseList)):
    for ID in range(translationBasis.shape[0]):
        baseList[i][ID,:, 0] += translationBasis[ID,i,0]
        baseList[i][ID,:, 1] += translationBasis[ID,i,1]
        baseList[i][ID,:, 2] += translationBasis[ID,i,2]



print("l_shoulderBasis")

print(l_shoulderBasis)

#print("ll")
#print(l_shoulderBasis[0])

print("r_shoulderBasis")

print(r_shoulderBasis)

print("poses_3d")
print(poses_3d)


r_shoulderBasis = changeAxisOrientation(r_shoulderBasis, "X")

#calculate relative Coordinates with elbow of respective arm as input
#poses_3d[:,x_elbowPoint,:].shape=(ID,3)
calculateRelativeCoordinates(poses_3d[:,4,:], l_shoulderBasis)







# perpendicularOnDim1 = np.random.randint(10, size=(3))
# l_hip = np.random.randint(10, size=(3))
#
# aMinusC = np.subtract(neck_center, perpendicularOnDim1)
# bMinusC = np.subtract(l_hip, perpendicularOnDim1)
#
# a_b = np.subtract(aMinusC, bMinusC)
#
# crossP = np.cross(a_b, aMinusC)
# # make sure the Z-coordinate is positive, if not -> *-1
# if (crossP[2] < 0):
#     crossP = crossP * -1
#
# print(crossP)
#
#     # verify perpendicularness
# crossP_squeeze = np.squeeze(np.asarray(crossP))
# y_squeeze = np.squeeze(np.asarray(aMinusC))
# x_squeeze = np.squeeze(np.asarray(bMinusC))
#
# y_dotP = np.dot(y_squeeze, crossP_squeeze)
# x_dotP = np.dot(x_squeeze, crossP_squeeze)
#
# print(y_dotP)
# print(x_dotP)


# def angle_between(p1, p2):
#     ang1 = np.arctan2(*p1[::-1])
#     ang2 = np.arctan2(*p2[::-1])
#     return np.rad2deg((ang1 - ang2) % (2 * np.pi))
#
# A = (1, 0,1)
# B = (1, -1,1)
#
# ang = acos(dot(v1,v2)/(|v1|.|v2|))
#
# print(angle_between(A, B))

#ang = acos( (x1*x2 + y1*y2 + z1*z1) / sqrt( (x1*x1 + y1*y1 + z1*z1)*(x2*x2+y2*y2+z2*z2) ) )
#
# print(int(arr.shape[1])/2)
#
# for i in range(int(int(arr.shape[1])/2)):
#     print(i)