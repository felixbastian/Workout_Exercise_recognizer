# PersonalTrainer
Personal Project: Implementation of prototype to identify the type of training exercise (push-ups, pull-ups, etc) with a RGB-camera.

## Implementation

- Usage of OpenCV library to convert video into 2D-limb annotation
- Implementation of OpenVino Module to convert 2D-limb annotation into 3D-limb annotation
- Conversion of data to relative basis to account for movements of subject in 3D space
- Conversion of coordinate data to angle data (Altitude, Azimuth) of arms and legs
- Creation of feature windows for prediction with training type label
- Development of feed-forward Convolutional Deep Learning Network

## Result
- Prototype was able to successfully annotate a video time series with changing exercises
- The accuracy was good generally but very dependent on the exercise (Conversion difficulties of 2D to 3D coordinates)
- Further accuracy could be achieved with multiple camera setups.

## Mutliple camera implementation
- Three phone camera setup
- Usage of VoxelPose library to merge and calibrate multiple camera data
- Data processing on an AWS Linux-based EC2 unit
- Implementation on hold due to unresolved library issues
