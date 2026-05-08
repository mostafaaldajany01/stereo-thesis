# StereoProject

Stereo vision pipeline for 3D point cloud reconstruction from calibrated stereo camera pairs. Supports two disparity estimation backends:

- **SGBM** — OpenCV's Semi-Global Block Matching (no GPU required)
- **FoundationStereo (FS)** — deep learning stereo model ([repo](https://github.com/NVlabs/FoundationStereo))

## Requirements

- **Python 3.12**

For the FoundationStereo backend, install PyTorch with CUDA **first** (check your CUDA version with `nvidia-smi`):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Then install the remaining dependencies:

```
pip install -r requirements.txt
```

For FS you also need:
- A local clone of [FoundationStereo](https://github.com/NVlabs/FoundationStereo) with pretrained weights

## Image Folder Structure

The pipeline expects a specific directory layout under `images/`. Each **baseline** is a subfolder (e.g. named by the distance between cameras in mm):

```
images/
└── <baseline>/            # e.g. "188"
    ├── cam0/              # Left camera calibration images
    │   ├── 001.jpg
    │   ├── 002.jpg
    │   └── ...
    ├── cam1/              # Right camera calibration images
    │   ├── 001.jpg
    │   ├── 002.jpg
    │   └── ...
    └── 3d/                # Stereo pairs for 3D reconstruction
        ├── cam0/          # Left images
        │   ├── 001.jpg
        │   └── ...
        └── cam1/          # Right images (matching filenames)
            ├── 001.jpg
            └── ...
```

### Calibration images (`cam0/` and `cam1/`)

- Chessboard pattern photos taken **simultaneously** by each camera
- Default pattern: **8 x 11** inner corners, **60 mm** square size
- These are used for intrinsic calibration of each camera and stereo calibration (R, T)
- Left and right images must be taken of the **same chessboard poses** in the **same order** (filenames are sorted and paired)

### 3D stereo pairs (`3d/cam0/` and `3d/cam1/`)

- Synchronized left/right image pairs of the scene you want to reconstruct
- Filenames are sorted and matched — `cam0/001.jpg` pairs with `cam1/001.jpg`
- These are rectified using the calibration, then fed to the disparity algorithm

## Usage

### Batch mode (processes all baselines)

```bash
# SGBM — no GPU needed
python run_sgbm.py

# FoundationStereo — requires GPU + FS repo
python run_fs.py
```

Each script auto-discovers all baseline folders under `images/`, runs calibration (or loads cached `.npz` files), and outputs disparity maps + point clouds to `out_sgbm/` or `out_fs/`.

### Interactive mode (single image pair)

The `*_model.py` files work on a single stereo pair with interactive parameter tuning:

- `sgbm_model.py` — interactive SGBM with trackbar sliders for tuning parameters
- `fs_model.py` — single-pair FoundationStereo inference

Edit the `IMG_SET` and `IMG_NAME` variables at the top of each file to point to your image pair.

## Configuration

All tunable parameters are defined as constants at the top of each script:

| Parameter | Description |
|---|---|
| `N_ROWS`, `N_COLS` | Chessboard inner corner count (default: 8 x 11) |
| `SQUARE_MM` | Chessboard square size in mm (default: 60) |
| `ZFAR` | Max depth cutoff in mm |
| `X/Y_FILTER_ENABLED/RANGE` | Optional spatial filtering of the point cloud |
| `SOR_ENABLED` | Statistical outlier removal |
| `DENOISE_CLOUD` | Voxel downsampling + radius outlier removal |
| `NUM_DISP`, `BLOCK_SIZE`, ... | SGBM-specific parameters |
| `VALID_ITERS`, `MAX_DISP`, `SCALE` | FS-specific parameters |

## Output

- `disparity_N.png` — color-mapped disparity visualization
- `pointcloud_N.ply` — 3D point cloud (open with MeshLab, CloudCompare, or Open3D)

## Setup for FS backend

Update the path constants in the relevant files to point to your local clone:

- `run_fs.py` / `fs_model.py`: set `FS_ROOT` to your FoundationStereo directory

Also update `MODEL_PATH` to point to your downloaded pretrained weights.
