# oculoplastic_segmentation


# Oculoplastic Disease Segmentation

## Overview
This repository contains code for segmentation of oculoplastic diseases using U-Net and DeepLabV3 (DLV3) architectures. The project aims to develop robust segmentation models trained on the CelebA-HQ dataset, specifically focusing on segmenting the sclera and brow from images of the left and right eye. 

## Key Features
1. **Generic Segmentation of Oculoplastic Disease**:
   - Focused on segmenting regions relevant to oculoplastic diseases.
   
2. **Dataset**:
   - Trained on the CelebA-HQ dataset.
   - Images were preprocessed by splitting them into left and right eye regions.
   - Segmentation targets include the sclera and brow.

3. **Hyperparameter Tuning**:
   - Hyperparameters were tuned using Weights & Biases (wandb) to optimize model performance.

4. **Iris Segmentation**:
   - Performed using Segment Anything Model (SAM) for accurate iris segmentation.

### NOTE

Work is under construction still. Code needs cleaning up. For final version will publish with optimal hyperparameters.
