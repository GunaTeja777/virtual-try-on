# Model Checkpoints

This folder contains the trained model checkpoints for the Virtual Try-On system.

## Structure

After training on Kaggle, download and place your `.pth` files here:

```
checkpoints/
├── best_segmentation.pth  (207 MB) - Segmentation U-Net
├── best_pose.pth          (14 MB)  - Pose Estimation
├── best_warp.pth          (50 MB)  - Cloth Warping TPS
├── best_tryon.pth         (78 MB) - Try-On GAN
└── best_refine.pth        (2 MB)  - Refinement Network
```

## Training Order

1. **Segmentation** (`1_Train_Segmentation.ipynb`) → `best_segmentation.pth`
2. **Pose** (`2_Train_Pose.ipynb`) → `best_pose.pth`
3. **Warp** (`3_Train_Warp.ipynb`) → `best_warp.pth`
4. **Try-On** (`4_Train_TryOn.ipynb`) → `best_tryon.pth`
5. **Refine** (`5_Train_Refine.ipynb`) → `best_refine.pth`

## Usage

Once all checkpoints are downloaded from Kaggle and placed here, run:

```bash
py inference.py --demo
```

This will load all 5 models and generate virtual try-on results.
