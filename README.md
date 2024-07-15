# Stimulation Behavior

Behavioral analysis is performed on animals such as Octopus, Monkey and Mice using transfer learning tools like [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) and steaming dimension reductions tools like [proSVD](https://github.com/draeloslab/prosvd).

## Musall-mice project
* Link to the Paper -> [https://pubmed.ncbi.nlm.nih.gov/31551604/](https://pubmed.ncbi.nlm.nih.gov/31551604/)
* Download (raw) data from -> [https://labshare.cshl.edu/shares/library/repository/38599/2pData/Animals/](https://labshare.cshl.edu/shares/library/repository/38599/2pData/Animals/)

### How to execute
1. The scripts required to perform the following are inside `scripts/musall-mice/`.
2. The raw data contains the SVD components of the video data. Use `download_data.py` to download files more easily.
3. Once the files are downloaded to the local system, the video svd files needs to be combined to restore the original video. Run `construct_video.py`
