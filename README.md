# Stimulation Behavior

Behavioral analysis is performed on animals such as Octopus, Monkey and Mice using transfer learning tools like [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) and steaming dimension reductions tools like [proSVD](https://github.com/draeloslab/prosvd).

## Quickstart
```
# download the repo
git clone https://github.com/draeloslab/stim_behavior
cd stim_behavior

# install dependencies
conda env create --file=environment.yml

# install the repo locally
conda activate stim_behavior
pip install -e .
```

## Musall-mice project
* Link to the Paper -> [https://pubmed.ncbi.nlm.nih.gov/31551604/](https://pubmed.ncbi.nlm.nih.gov/31551604/)
* Download (raw) data from -> [https://labshare.cshl.edu/shares/library/repository/38599/2pData/Animals/](https://labshare.cshl.edu/shares/library/repository/38599/2pData/Animals/)

### How to execute
1. The scripts required to perform the following are inside `scripts/musall-mice/`.
2. The raw data contains the SVD components of the video data. Use `download_data.py` to download files more easily.
3. Once the files are downloaded to the local system, the video svd files needs to be combined to restore the original video. Run `construct_video.py`

## Rhesus-Macaque
Applying DeepLabCut for real time hand tracking in collaboration with the Chestek Lab.
- Scripts
- - cameraCtrl.py and camera.py: current camera interfaces for recording and accessed Imaging Source camera frames during NHP experiments. Use python cameraCtrl.py -h for help
- - dataLoader.py: Class used to access DLC models for training and refining. dlcLiveTesting is an example notebook that walks through how to use DLC with exisiting NHP data. DataLoader class is used to preprocess and analyze DLC and SLEAP csv files. Use dmpFatigueFigure for an example notebook walking through the class.

## Sarvestani-treeshrew project
Tree shrew with cranial window over V1, with Gcamp8s injected. Eye tracking camera records face and eye. 2P recording of neurons in V1. Original experiment and data collection by: Madineh Sarvestani @ MPFI/Cornell.

### Setup data
1. Download the data from this dropbox link here: [Link](https://www.dropbox.com/scl/fo/xmvk5pmog323oppam80tn/h?rlkey=tl4u08bmos38lpaf0f79zn57c&dl=0).
2. Unzip and move it to a folder of your choice

### Execute
1. Run `python scripts/sarvestani_treeshrew/main.py` to extract the flourescence signals from the raw data using the suite2p package.
2. Suffix `-h` to understand command-line args.

## Pelled-octopus project
### Setup data
1. Download sample videos of the octopus experiement conducted in Dr.Pelled lab (MSU) from this dropbox link here: [Link](https://www.dropbox.com/scl/fo/2fmw5bhcjtio235lebxqs/AGMrKvDIy7BczYjE1wn9m0I?rlkey=4jq9k1e260vx0jn2srwx68hbw&dl=0).

### Execute
1. Execute `scripts/pelled_octopus/dlc_compute_octopus.py` to perform deeplabcut analysis on the octopus videos.
2. Execute `scripts/pelled_octopus/prosvd_compute_octopus.py` to perform prosvd analysis on the octopus videos.
