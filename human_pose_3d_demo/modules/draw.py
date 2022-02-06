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

import math

import cv2
import numpy as np


previous_position = []
theta, phi = math.pi / 4, -math.pi / 6
should_rotate = False
scale_dx = 800
scale_dy = 800


class Plotter3d:

    #Defining all edges that need to be drawn
    SKELETON_EDGES = np.array([[11, 10], [10, 9], [9, 0], [0, 3], [3, 4], [4, 5], [0, 6], [6, 7], [7, 8], [0, 12],
                               [12, 13], [13, 14], [0, 1], [1, 15], [15, 16], [1, 17], [17, 18]])

    # defining the indices of the lines that need to be drawn between the relBasis_2D points
    global baseSkeleton
    baseSkeleton = np.array([[2, 1], [2, 0], [2, 3]])

    def __init__(self, canvas_size, origin=(0.5, 0.5), scale=1):
        self.origin = np.array([origin[1] * canvas_size[1], origin[0] * canvas_size[0]], dtype=np.float32)  # x, y
        self.scale = np.float32(scale)
        self.theta = 0
        self.phi = 0
        axis_length = 200
        axes = [
            np.array([[-axis_length/2, -axis_length/2, 0], [axis_length/2, -axis_length/2, 0]], dtype=np.float32),
            np.array([[-axis_length/2, -axis_length/2, 0], [-axis_length/2, axis_length/2, 0]], dtype=np.float32),
            np.array([[-axis_length/2, -axis_length/2, 0], [-axis_length/2, -axis_length/2, axis_length]], dtype=np.float32)]
        step = 20
        for step_id in range(axis_length // step + 1):  # add grid
            axes.append(np.array([[-axis_length / 2, -axis_length / 2 + step_id * step, 0],
                                  [axis_length / 2, -axis_length / 2 + step_id * step, 0]], dtype=np.float32))
            axes.append(np.array([[-axis_length / 2 + step_id * step, -axis_length / 2, 0],
                                  [-axis_length / 2 + step_id * step, axis_length / 2, 0]], dtype=np.float32))
        self.axes = np.array(axes)

    #the final plotting - plotter.plot(canvas_3d, poses_3d, edges) - Call in the demo.py-file, poses_3d (vertices) is the (x,19,3) array
    def plot(self, img, vertices, edges, relBasis, l_shoulderBasis, r_shoulderBasis):
        global theta, phi
        img.fill(0)
        R = self._get_rotation(theta, phi)
        self._draw_axes(img, R)

        #call the relative base plotter
        self._plot_relative_base(img, vertices, R, relBasis)
        self._plot_l_shoulder_base(img, vertices, R, l_shoulderBasis)
        self._plot_r_shoulder_base(img, vertices, R, r_shoulderBasis)

        if len(edges) != 0:
            self._plot_edges(img, vertices, edges, R)

    #Drawing the 3D axes
    def _draw_axes(self, img, R):
        axes_2d = np.dot(self.axes, R)
        axes_2d = axes_2d * self.scale + self.origin
        for axe in axes_2d:
            axe = axe.astype(int)
            cv2.line(img, tuple(axe[0]), tuple(axe[1]), (255, 255, 255), 1, cv2.LINE_AA)

    #Drawing the body
    def _plot_edges(self, img, vertices, edges, R):

        #creates the literal 2D-image on the screen based on the 3D-coordinates (vertices) and the relative rotation (R)
        #dot product of vertices (x,19,3) . (3,2) -> (x,19,2)-matrix
        vertices_2d = np.dot(vertices, R)

        #reshaping the 2D-vertice array in size (-1=length of array,2) -> edges_vertices has only 2 columns for (x,y)
        #The vertices_2d.reshape function turns the (x,19,2)-matrix into a (19x,2) matrix -> all vertices below each other

        #vertices_2d.reshape((-1, 2))[edges] -> Returns a (17*ID,2,2) Matrix
        #Bascially what it does it is maps the vertices with what edges need to be connected
        #At the end 17 lines need to be drawn. Since some vertices are used twice, mapping logic is required
        #Each (2,2)-matrix represents one line edge_vertices[0] is the start (x,y)-tuple na dedge_vertices[1] the end
        #Therefore, it does not have implications on the actual coordinates (hopefully)
        #e.g. First element in skeleton is the edge = [5,6], looking at the vertices_2d, list at index 5&6
        #should show the coordinates that is the first (2,2) matrix in edges_vertices (excluding scale & origin part)
        edges_vertices = vertices_2d.reshape((-1, 2))[edges] * self.scale + self.origin
        for edge_vertices in edges_vertices:
            edge_vertices = edge_vertices.astype(int)
            cv2.line(img, tuple(edge_vertices[0]), tuple(edge_vertices[1]), (255, 0, 25), 1, cv2.LINE_AA)


    def _plot_relative_base(self, img, vertices, R, relBasis):

        global baseSkeleton
        relBasis_2d = np.dot(relBasis, R)


        edges = (baseSkeleton + 4 * np.arange(relBasis_2d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
        edges_vertices = relBasis_2d.reshape((-1, 2))[edges] * self.scale + self.origin


        for edge_vertices in edges_vertices:
            edge_vertices = edge_vertices.astype(int)
            cv2.line(img, tuple(edge_vertices[0]), tuple(edge_vertices[1]), (40, 252, 3), 1, cv2.LINE_AA)

    #plot base for l_shoulder
    def _plot_l_shoulder_base(self, img, vertices, R, l_shoulderBasis):

        global baseSkeleton
        relBasis_2d = np.dot(l_shoulderBasis, R)


        edges = (baseSkeleton + 4 * np.arange(relBasis_2d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
        edges_vertices = relBasis_2d.reshape((-1, 2))[edges] * self.scale + self.origin


        for edge_vertices in edges_vertices:
            edge_vertices = edge_vertices.astype(int)
            cv2.line(img, tuple(edge_vertices[0]), tuple(edge_vertices[1]), (40, 252, 3), 1, cv2.LINE_AA)

    # plot base for r_shoulder
    def _plot_r_shoulder_base(self, img, vertices, R, r_shoulderBasis):

        global baseSkeleton
        relBasis_2d = np.dot(r_shoulderBasis, R)

        edges = (baseSkeleton + 4 * np.arange(relBasis_2d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
        edges_vertices = relBasis_2d.reshape((-1, 2))[edges] * self.scale + self.origin

        for edge_vertices in edges_vertices:
            edge_vertices = edge_vertices.astype(int)
            cv2.line(img, tuple(edge_vertices[0]), tuple(edge_vertices[1]), (40, 252, 3), 1, cv2.LINE_AA)


    def _get_rotation(self, theta, phi):
        sin, cos = math.sin, math.cos
        return np.array([
            [ cos(theta), sin(theta) * sin(phi)],
            [-sin(theta), cos(theta) * sin(phi)],
            [ 0,                      -cos(phi)]
        ], dtype=np.float32)  # transposed

    @staticmethod
    def mouse_callback(event, x, y, flags, params):
        global previous_position, theta, phi, should_rotate, scale_dx, scale_dy
        if event == cv2.EVENT_LBUTTONDOWN:
            previous_position = [x, y]
            should_rotate = True
        if event == cv2.EVENT_MOUSEMOVE and should_rotate:
            theta += (x - previous_position[0]) / scale_dx * 2 * math.pi
            phi -= (y - previous_position[1]) / scale_dy * 2 * math.pi * 2
            phi = max(min(math.pi / 2, phi), -math.pi / 2)
            previous_position = [x, y]
        if event == cv2.EVENT_LBUTTONUP:
            should_rotate = False


body_edges = np.array(
    [[0, 1],  # neck - nose
     [1, 16], [16, 18],  # nose - l_eye - l_ear
     [1, 15], [15, 17],  # nose - r_eye - r_ear
     [0, 3], [3, 4], [4, 5],     # neck - l_shoulder - l_elbow - l_wrist
     [0, 9], [9, 10], [10, 11],  # neck - r_shoulder - r_elbow - r_wrist
     [0, 6], [6, 7], [7, 8],        # neck - l_hip - l_knee - l_ankle
     [0, 12], [12, 13], [13, 14]])  # neck - r_hip - r_knee - r_ankle


def draw_poses(img, poses_2d):
    for pose in poses_2d:
        pose = np.array(pose[0:-1]).reshape((-1, 3)).transpose()
        was_found = pose[2] > 0
        for edge in body_edges:
            if was_found[edge[0]] and was_found[edge[1]]:
                cv2.line(img, tuple(pose[0:2, edge[0]].astype(np.int32)), tuple(pose[0:2, edge[1]].astype(np.int32)),
                         (255, 255, 0), 4, cv2.LINE_AA)
        for kpt_id in range(pose.shape[1]):
            if pose[2, kpt_id] != -1:
                cv2.circle(img, tuple(pose[0:2, kpt_id].astype(np.int32)), 3, (0, 255, 255), -1, cv2.LINE_AA)
