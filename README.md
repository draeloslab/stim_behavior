# Stimulation Behavior

Behavioral analysis is performed on animals such as Octopus, Zebrafish and Mice using transfer learning tools like DeepLabCut and steaming dimension reductions tools like proSVD.

## Musall-mice project
Paper: https://pubmed.ncbi.nlm.nih.gov/31551604/
Raw Data: Download it from https://labshare.cshl.edu/shares/library/repository/38599/2pData/Animals/${mouse_id}/SpatialDisc

The scripts required to perform the following are inside `scripts/musall-mice/`.
Step 1. The raw data contains the SVD components of the video data. Use `download_data.py` to download files more easily.
Step 2. Once the files are downloaded to the local system, the video svd files needs to be combined to restore the original video. Run `construct_video.py`
