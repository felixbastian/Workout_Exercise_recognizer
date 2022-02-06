Base Folder
C:\Users\felix\OneDrive\Dokumente\Python Projects\PersonalTrainer\human_pose_3d_demo

REQUIRES PYTHON 3.6
Initialize environment - RUN EVERYTIME WHEN CMD IS RESTARTED
1) Run CMD as administrator
2) cd C:\Program Files (x86)\Intel\openvino_2021\bin
3) setupvars.bat

python human_pose_estimation_3d_demo.py -m "..\..\..\..\..\..\..\Program Files (x86)\Intel\openvino_2021\deployment_tools\tools\model_downloader\public\human-pose-estimation-3d-0001\FP16\human-pose-estimation-3d-0001.xml" -i ..\Videos\testing\firstTest.mp4

'''bash'''
cd C:\Program Files (x86)\Intel\openvino_2021\bin
setupvars.bat
cd C:\Users\felix\OneDrive\Dokumente\Python Projects\PersonalTrainer\human_pose_3d_demo
python human_pose_estimation_3d_demo.py -m "..\..\..\..\..\..\..\Program Files (x86)\Intel\openvino_2021\deployment_tools\tools\model_downloader\public\human-pose-estimation-3d-0001\FP16\human-pose-estimation-3d-0001.xml" -i ..\Videos\testing\firstTest.mp4
'''

######################################################################

Body map


body_edges = np.array(
    [[0, 1],  # neck - nose
     [1, 16], [16, 18],  # nose - l_eye - l_ear
     [1, 15], [15, 17],  # nose - r_eye - r_ear
     [0, 3], [3, 4], [4, 5],     # neck - l_shoulder - l_elbow - l_wrist
     [0, 9], [9, 10], [10, 11],  # neck - r_shoulder - r_elbow - r_wrist
     [0, 6], [6, 7], [7, 8],        # neck - l_hip - l_knee - l_ankle
     [0, 12], [12, 13], [13, 14]])  # neck - r_hip - r_knee - r_ankle

######################################################################
poses_3d - output

In human_pose_estimtion_3d_demo.py
the variables x,y,z are created from poses_3d.
Each of them has 19 dimensions (one for each limb) and n outer dimensions e.g. x.shape= [1,19] or x.shape[2,19] based on the number
of people detected. This is the same for all three variables x,y,z.



######################################################################

3D Mapping logic

QUESTION -> How to handle multiple people???
The 2D array contains 19 inner dimensions (one per limb) that each contain 3 further dimensions (one per coordinate x,y,z)
Each row entrance represents one frame
[[[x0,y0,z0], [x1,y1,z1], [x2,y2,z2], ... [x19,y19,z19]]
...
]


SUGGESTION FOR FUTHER DEVELOPMENT
Since the focus of the 3D representation is the positioning within the body (arms are stretched forward, rest straight = push-up position)
it could make sense to create a relative coordinate system within the body with one center point.
(e.g. the neck represents (0,0,0) and all other body parts are relative seen relative to the neck.)
In order to also include the surrounding position (is the body laying on the floor or standing) the neck could be the only point
that has two coordinates (0,0,0) as center and (x,y,z) relative to the position in the room.




