# Stimulation Behavior

Behavioral analysis is performed on animals such as Octopus, Zebrafish and Mice using transfer learning tools like DeepLabCut and steaming dimension reductions tools like proSVD.

## Rhesus-Macaque

Applying DeepLabCut for real time hand tracking in collaboration with the Chestek Lab.
- Scripts
- - cameraCtrl.py and camera.py: current camera interfaces for recording and accessed Imaging Source camera frames during NHP experiments. Use python cameraCtrl.py -h for help
- - dataLoader.py: Class used to access DLC models for training and refining. dlcLiveTesting is an example notebook that walks through how to use DLC with exisiting NHP data. DataLoader class is used to preprocess and analyze DLC and SLEAP csv files. Use dmpFatigueFigure for an example notebook walking through the class.

