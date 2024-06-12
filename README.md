# sleps
Supervised Learning of Enhancer Promoter Specificity.

## Overview
Sleps software is a pipeline that utilizes machine learning and graph relationships to predict the effect of an enhancer on a particular promoter. 

Currently, the software is in alpha and needs further development to generalize enough for training on new datasets. However, our experiment and trained sleps model can be reproduced using this guide. There are three main steps needed to complete the training of the model. 

1. [Download Data](https://drive.google.com/drive/folders/1270MmEk8oF3VpJ5llkaKSCqPy0i8-qLW?usp=drive_link)
 - First, download requisite data. Training data includes Hi-C, DHS, H3K27ac, and CRISPRi datasets as well as a suite of ChIPSeq data from ENCODE. 
2. [Generate Enhancer Networks](https://github.com/HanLabUNLV/abic/blob/master/network_generation_process.md)
 - Next generate networks using Hi-C data and tools. This step requires a singularity environment that we have provided as well as HPC capabilities. 
3. [Train Model & Apply](https://github.com/HanLabUNLV/abic/blob/master/learning.md)
 - Lastly train the model again using singularity and an HPC. With the trained model in hand, you can apply the model as well to new data. 

## Singularity Containers
- [Network singularity container](https://drive.google.com/drive/folders/13WP9gLttNaa3HQLAs5Of-PB4ZVqbmwUJ?usp=sharing)
- [Machine Learning singularity container](https://drive.google.com/drive/folders/1QTNEvYx6T5kXfspyx4w_OEKo8dJ8cJTQ?usp=sharing)


