#!/usr/bin/env python3
"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import json
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
import math



import cv2
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from modules.inference_engine import InferenceEngine
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
import  modules.monitors
from modules.images_capture import open_images_capture

def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(poses_3d.shape[0]):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3] = np.dot(R_inv, pose_3d[0:3] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d


if __name__ == '__main__':
    parser = ArgumentParser(description='Lightweight 3D human pose estimation demo. '
                                        'Press esc to exit, "p" to (un)pause video or process next image.',
                            add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model',
                      help='Required. Path to an .xml file with a trained model.',
                      type=Path, required=True)
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('--loop', default=False, action='store_true',
                      help='Optional. Enable reading the input in a loop.')
    args.add_argument('-o', '--output', required=False,
                      help='Optional. Name of output to save.')
    args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                      help='Optional. Number of frames to store in output. '
                           'If 0 is set, all frames are stored.')
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on: CPU, GPU, FPGA, HDDL or MYRIAD. '
                           'The demo will look for a suitable plugin for device specified '
                           '(by default, it is CPU).',
                      type=str, default='CPU')
    args.add_argument('--height_size', help='Optional. Network input layer height size.', type=int, default=256)
    args.add_argument('--extrinsics_path',
                      help='Optional. Path to file with camera extrinsics.',
                      type=Path, default=None)
    args.add_argument('--fx', type=np.float32, default=-1, help='Optional. Camera focal length.')
    args.add_argument('--no_show', help='Optional. Do not display output.', action='store_true')
    args.add_argument("-u", "--utilization_monitors", default='', type=str,
                      help="Optional. List of monitors to show initially.")
    args = parser.parse_args()



    global dataFrame
    dataFrame = []
    #dataframe containing the entire coordinates table with max 3 ID's per layer ((2,19,3) somehow leads
    #to 3 layers in the first dimension
    #The dataFrame can ce accessed as following: dataFrame[0][0][0][0]
    #=dataFrame[#ofFrame][IdInFrame][key point][Coordinate(x,y,z)]

    def takeCoordinates():
        global dataFrame
        frame = np.full((0, 19, 3), np.nan)
        xRow, xCol = x.shape
        for n in range(xRow):
            coordArray = np.empty((0, 3), int)
            for m in range(xCol):
                coordArray = np.append(coordArray, np.array([[x[n, m], y[n, m], z[n, m]]]), axis=0)
            frame = np.insert(frame,n,coordArray,axis=0)
        dataFrame.append(frame)

    # Return the min, max and abs(max-min) values for coordinates of all key points
    #SOMEHOW NOT WORKING CORRECTLY
    global Three_D_array
    Three_D_array = np.empty((0, 3), int)

    def testDistances():
        global Three_D_array
        for i in range(19):
            for d in range(3):
               min = dataFrame[0][0][i][d]
               max = dataFrame[0][0][i][d]

               for x in range(len(dataFrame)):
                    if (dataFrame[x][0][i][d] < min ): min = dataFrame[x][0][i][d]
                    elif (dataFrame[x][0][i][d] > max ): max =  dataFrame[x][0][i][d]
               Three_D_array = np.append(Three_D_array, np.array([[min, max, abs(max-min)]]), axis=0)
            Three_D_array = np.append(Three_D_array, np.array([[0,i,0]]), axis=0)

    def calculateRelativeCoordinates():
        #1) take base coordinates as input
        #2)send basis to real center
        #3) norm the base coordinates
        #4) send to be found key point to center as well
        #4)calculate change of basis
        return 0

    # calculate angle between vectors
    def calculateVectorAngle(vector1, vector2):
        #the two vectors still have to be subtracted by their common starting point in order to calc. degree
        v1_squeeze = np.squeeze(np.asarray(vector1))
        v2_squeeze = np.squeeze(np.asarray(vector2))
        unit_vector_1 = v1_squeeze / np.linalg.norm(v1_squeeze)
        unit_vector_2 = v2_squeeze / np.linalg.norm(v2_squeeze)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angleInDeg = math.degrees(np.arccos(dot_product))
        return angleInDeg

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


    def changeAxisOrientation(relativeBasis, axis):
        if (axis == "X"):a = 1
        elif (axis == "Y"):a = 0
        elif (axis == "Z"):a = 3

        for ID in range(relativeBasis.shape[0]):
            relativeBasis[ID, a] = np.subtract(relativeBasis[ID, 2, :],
                                               np.subtract(relativeBasis[ID, a, :], relativeBasis[ID, 2, :]))

        return relativeBasis

    #output is a (ID, 6,3) array
    def translateBasisTo(poses_3d, perpendicularOnDim1, relBasis):
        _translationBasis = []

        #define rotation for defined each basis in order
        #rotationBasis = np.array([[1, 1, 1], [-1, 1, 1]])

        for IDs in range(poses_3d.shape[0]):
            translationArray = np.empty((0, 3), int)

            # determine the key points the basis should be translated to
            # to the relative basis (REMINDER: poses_3d is sorted as (z,x,y))

            #l_shoulder
            l_shoulder = np.subtract(np.array([[poses_3d[IDs, 3, 0], poses_3d[IDs, 3, 1], poses_3d[IDs, 3, 2]]]), perpendicularOnDim1)
            translationArray = np.append(translationArray, l_shoulder, axis=0)

            # r_shoulder
            r_shoulder = np.subtract(np.array([[poses_3d[IDs, 9, 0], poses_3d[IDs, 9, 1], poses_3d[IDs, 9, 2]]]),
                                     perpendicularOnDim1)
            translationArray = np.append(translationArray, r_shoulder, axis=0)


            _translationBasis.append(translationArray)

        translationBasis = np.array(_translationBasis)


        # copy relative basis that the new basis will be added on top (translation)
        l_shoulderBasis = np.empty_like(relBasis)
        l_shoulderBasis[:] = relBasis

        r_shoulderBasis = np.empty_like(relBasis)
        r_shoulderBasis[:] = relBasis

        baseList = [l_shoulderBasis, r_shoulderBasis]

        # Change the relative basis according to the parameters for each basis
        for i in range(len(baseList)):
            for ID in range(translationBasis.shape[0]):
                baseList[i][ID, :, 0] += translationBasis[ID, i, 0]
                baseList[i][ID, :, 1] += translationBasis[ID, i, 1]
                baseList[i][ID, :, 2] += translationBasis[ID, i, 2]

        #invert X-Axis orientation
        r_shoulderBasis = changeAxisOrientation(r_shoulderBasis, "X")

        return l_shoulderBasis, r_shoulderBasis

    stride = 8
    inference_engine = InferenceEngine(args.model, args.device, stride)
    canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    plotter = Plotter3d(canvas_3d.shape[:2])
    canvas_3d_window_name = 'Canvas 3D'
    if not args.no_show:
        cv2.namedWindow(canvas_3d_window_name)
        cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)

    file_path = args.extrinsics_path
    if file_path is None:
        file_path = Path(__file__).parent / 'data/extrinsics.json'
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)

    cap = open_images_capture(args.input, args.loop)
    is_video = cap.get_type() in ('VIDEO', 'CAMERA')
    frame = cap.read()
    if frame is None:
        raise RuntimeError("Can't read an image from the input")

    video_writer = cv2.VideoWriter()
    if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                             cap.fps(), (frame.shape[1], frame.shape[0])):
        raise RuntimeError("Can't open video writer")

    base_height = args.height_size
    fx = args.fx

    frames_processed = 0
    delay = 1
    esc_code = 27
    p_code = 112
    space_code = 32
    mean_time = 0
    presenter = modules.monitors.Presenter(args.utilization_monitors, 0)

    coordinateArray = np.empty((0,2), dtype=object)

    counter = 0
    while frame is not None:
        current_time = cv2.getTickCount()
        input_scale = base_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        if fx < 0:  # Focal length is unknown
            fx = np.float32(0.8 * frame.shape[1])

        inference_result = inference_engine.infer(scaled_img)

        #poses_3d is an array of coordinates just like dataFrame. However, poses_3d only saves the coordinates
        # of the specific Frame in format (x, 19,3) with x IDs, 19 key points & 3 coordinates (in the order z,x,y)
        poses_3d, poses_2d = parse_poses(inference_result, input_scale, stride, fx, is_video)
        edges = []
        if len(poses_3d) > 0:
            poses_3d = rotate_poses(poses_3d, R, t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]

            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]

            takeCoordinates()
            relBasis, perpendicularOnDim1, crossP = calculateRelativeBasis(poses_3d)
            l_shoulderBasis, r_shoulderBasis = translateBasisTo(poses_3d, perpendicularOnDim1, relBasis)
            calculateRelativeCoordinates(poses_3d, l_shoulderBasis)

            #SKELETON-EDGES is a (17,2)-array defining the 17 edges that need to be connected with 2 key points
            #poses 3D.shape[0] returns the number of stacks (IDs) (for a (2,19,3)-matrix; .shape = 2
            # np.arrange returns evenly spaced values np.arange(2) = [0,1]

            #(Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1)))
            #Returns a (ID,17,2) matrix
            #(0,17,2) are the edges as defined in skeleton
            #(1,17,2) are the edges +19
            #(2,17,2) are the edges +38
            #->(ID,18,2) are the edges +19*ID

            #(Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
            #Returns a matrix (17*ID,2) so if one person is tracked = (17,2), 2 = (34,2)
            #Is equal to the reshaping part above but just all below each other instead of stacks in dim3
            edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
            #
            # print(Plotter3d.SKELETON_EDGES)
            # print("-------------")
            # print(Plotter3d.SKELETON_EDGES.shape)
            # print("----------")

        #def plot(self, img, vertices, edges): - function in draw.py -> The call here links the coordinates to the drawing
        plotter.plot(canvas_3d, poses_3d, edges, relBasis, l_shoulderBasis, r_shoulderBasis)

        presenter.drawGraphs(frame)
        draw_poses(frame, poses_2d)
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05
        cv2.putText(frame, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                    (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

        frames_processed += 1
        if video_writer.isOpened() and (args.output_limit <= 0 or frames_processed <= args.output_limit):
            video_writer.write(frame)

        if not args.no_show:
            cv2.imshow(canvas_3d_window_name, canvas_3d)
            cv2.imshow('3D Human Pose Estimation', frame)

            key = cv2.waitKey(delay)
            if key == esc_code:
                break
            if key == p_code:
                if delay == 1:
                    delay = 0
                else:
                    delay = 1
            else:
                presenter.handleKey(key)
            if delay == 0 or not is_video:  # allow to rotate 3D canvas while on pause
                key = 0
                while (key != p_code
                       and key != esc_code
                       and key != space_code):
                    plotter.plot(canvas_3d, poses_3d, edges, relBasis, l_shoulderBasis, r_shoulderBasis)
                    cv2.imshow(canvas_3d_window_name, canvas_3d)
                    key = cv2.waitKey(33)
                if key == esc_code:
                    break
                else:
                    delay = 1
        frame = cap.read()


    #np.savetxt(Path("coordinate_output.csv"), poses_3d_copy, delimiter=",", fmt='%s')
    print(presenter.reportMeans())

