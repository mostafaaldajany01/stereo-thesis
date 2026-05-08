import os
import glob
import numpy as np
import cv2 as cv
import open3d as o3d

IMAGES_ROOT = 'images'
OUT_ROOT    = 'out_sgbm'

N_ROWS    = 8
N_COLS    = 11
SQUARE_MM = 60

NUM_DISP    = 160
BLOCK_SIZE  = 5
MIN_DISP    = 0
UNIQUENESS  = 5
SPECKLE_WIN = 200
SPECKLE_RNG = 2
P1_SCALE    = 8
P2_SCALE    = 32

ZFAR              = 22000

X_FILTER_ENABLED  = False
X_FILTER_RANGE    = 1000

Y_FILTER_ENABLED  = False
Y_FILTER_RANGE    = 1000

SOR_ENABLED       = False
SOR_NB_NEIGHBORS  = 30
SOR_STD_RATIO     = 1.2

DENOISE_CLOUD      = False
DENOISE_VOXEL_SIZE = 1.0
DENOISE_NB_POINTS  = 30
DENOISE_RADIUS     = 30.0

def _single_calibrate(img_dir: str):
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    if not img_paths:
        raise FileNotFoundError(f'No images found in {img_dir}')
    term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    obj_tpl = np.zeros((N_ROWS * N_COLS, 3), np.float32)
    obj_tpl[:, :2] = np.mgrid[0:N_ROWS, 0:N_COLS].T.reshape(-1, 2) * SQUARE_MM
    obj_list, img_list, img_size = [], [], None
    for p in img_paths:
        gray = cv.cvtColor(cv.imread(p), cv.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]
        found, corners = cv.findChessboardCorners(
            gray, (N_ROWS, N_COLS),
            cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE | cv.CALIB_CB_FAST_CHECK)
        if found:
            obj_list.append(obj_tpl)
            img_list.append(cv.cornerSubPix(gray, corners, (13, 13), (-1, -1), term))
    if not img_list:
        raise RuntimeError(f'No chessboard corners found in {img_dir}')
    err, cam_mtx, dist, _, _ = cv.calibrateCamera(obj_list, img_list, img_size, None, None)
    print(f'    reprojection error: {err:.4f} px  ({len(img_list)} images used)')
    return cam_mtx, dist


def _stereo_calibrate(cam0, dist0, cam1, dist1, baseline_dir: str):
    left_paths  = sorted(glob.glob(os.path.join(baseline_dir, 'cam0', '*.jpg')))
    right_paths = sorted(glob.glob(os.path.join(baseline_dir, 'cam1', '*.jpg')))
    term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    obj_tpl = np.zeros((N_ROWS * N_COLS, 3), np.float32)
    obj_tpl[:, :2] = np.mgrid[0:N_ROWS, 0:N_COLS].T.reshape(-1, 2) * SQUARE_MM
    obj_list, left_list, right_list, img_size = [], [], [], None
    for lp, rp in zip(left_paths, right_paths):
        gl = cv.cvtColor(cv.imread(lp), cv.COLOR_BGR2GRAY)
        gr = cv.cvtColor(cv.imread(rp), cv.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gl.shape[::-1]
        fl, cl = cv.findChessboardCorners(gl, (N_ROWS, N_COLS),
                     cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE)
        fr, cr = cv.findChessboardCorners(gr, (N_ROWS, N_COLS),
                     cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE)
        if fl and fr:
            obj_list.append(obj_tpl)
            left_list.append(cv.cornerSubPix(gl, cl, (13, 13), (-1, -1), term))
            right_list.append(cv.cornerSubPix(gr, cr, (13, 13), (-1, -1), term))
    ret, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
        obj_list, left_list, right_list,
        cam0, dist0, cam1, dist1,
        img_size, criteria=term, flags=cv.CALIB_FIX_INTRINSIC)
    print(f'    stereo error: {ret:.4f} px   baseline: {abs(T[0][0]):.2f} mm')
    return R, T, E, F


def get_calibration(baseline: str, baseline_dir: str):
    calib_file = f'calib_{baseline}.npz'
    if os.path.isfile(calib_file):
        print(f'  [calib] Loaded {calib_file} (skipping calibration)')
        d = np.load(calib_file)
        return d['camMatrix0'], d['distCoeff0'], d['camMatrix1'], d['distCoeff1'], d['R'], d['T']
    print(f'  [calib] {calib_file} not found — calibrating ...')
    cam0, dist0 = _single_calibrate(os.path.join(baseline_dir, 'cam0'))
    cam1, dist1 = _single_calibrate(os.path.join(baseline_dir, 'cam1'))
    R, T, E, F  = _stereo_calibrate(cam0, dist0, cam1, dist1, baseline_dir)
    np.savez(calib_file, camMatrix0=cam0, distCoeff0=dist0,
             camMatrix1=cam1, distCoeff1=dist1, R=R, T=T, E=E, F=F)
    print(f'  [calib] Saved {calib_file}')
    return cam0, dist0, cam1, dist1, R, T


def process_pair(idx, rect_l, rect_r, Q, cam_matrix, out_dir):
    gray_l = cv.cvtColor(rect_l, cv.COLOR_BGR2GRAY)
    gray_r = cv.cvtColor(rect_r, cv.COLOR_BGR2GRAY)

    stereo = cv.StereoSGBM_create(
        minDisparity=MIN_DISP,
        numDisparities=NUM_DISP,
        blockSize=BLOCK_SIZE,
        P1=P1_SCALE * 3 * BLOCK_SIZE ** 2,
        P2=P2_SCALE * 3 * BLOCK_SIZE ** 2,
        uniquenessRatio=UNIQUENESS,
        speckleWindowSize=SPECKLE_WIN,
        speckleRange=SPECKLE_RNG,
        disp12MaxDiff=1,
        mode=cv.STEREO_SGBM_MODE_HH
    )

    disp_raw = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
    valid    = disp_raw > MIN_DISP

    disp_vis = disp_raw.copy()
    disp_vis[~valid] = 0
    norm = np.zeros(disp_vis.shape, dtype=np.uint8)
    mask = disp_vis > MIN_DISP
    if mask.any():
        norm[mask] = cv.normalize(disp_vis[mask], None, 0, 255,
                                   cv.NORM_MINMAX, cv.CV_8U).flatten()
    color_disp = cv.applyColorMap(norm, cv.COLORMAP_TURBO)
    color_disp[~mask] = 0
    cv.imwrite(os.path.join(out_dir, f'disparity_{idx}.png'), color_disp)

    fx = cam_matrix[0, 0]
    cx = cam_matrix[0, 2]
    cy = cam_matrix[1, 2]

    H, W = disp_raw.shape
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    pts3d = cv.reprojectImageTo3D(disp_raw, Q, handleMissingValues=False)
    z_mm  = pts3d[:, :, 2]

    x_mm = (uu - cx) * z_mm / fx
    y_mm = (vv - cy) * z_mm / fx

    pts_mm     = np.stack([x_mm, y_mm, z_mm], axis=-1).reshape(-1, 3)
    colors     = rect_l[:, :, ::-1].reshape(-1, 3).astype(np.float64) / 255.0
    valid_flat = valid.reshape(-1)

    pts_mm = pts_mm[valid_flat]
    colors = colors[valid_flat]

    keep = np.isfinite(pts_mm).all(axis=1) & (pts_mm[:, 2] > 0) & (pts_mm[:, 2] <= ZFAR)
    if X_FILTER_ENABLED:
        keep &= (pts_mm[:, 0] >= -X_FILTER_RANGE) & (pts_mm[:, 0] <= X_FILTER_RANGE)
    if Y_FILTER_ENABLED:
        keep &= (pts_mm[:, 1] >= -Y_FILTER_RANGE) & (pts_mm[:, 1] <= Y_FILTER_RANGE)

    pts_mm = pts_mm[keep]
    colors = colors[keep]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_mm)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if SOR_ENABLED and len(pcd.points) > SOR_NB_NEIGHBORS:
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO)
        pcd = pcd.select_by_index(ind)

    if DENOISE_CLOUD and len(np.asarray(pcd.points)) > DENOISE_NB_POINTS:
        pcd = pcd.voxel_down_sample(voxel_size=DENOISE_VOXEL_SIZE)
        _, ind = pcd.remove_radius_outlier(nb_points=DENOISE_NB_POINTS, radius=DENOISE_RADIUS)
        pcd = pcd.select_by_index(ind)

    ply_path = os.path.join(out_dir, f'pointcloud_{idx}.ply')
    o3d.io.write_point_cloud(ply_path, pcd)
    n_pts = len(np.asarray(pcd.points))
    print(f'    [{idx}] disparity_{idx}.png  pointcloud_{idx}.ply  ({n_pts} pts)')


def process_baseline(baseline: str):
    baseline_dir = os.path.join(os.getcwd(), IMAGES_ROOT, baseline)
    pairs_dir    = os.path.join(baseline_dir, '3d')
    out_dir      = os.path.join(OUT_ROOT, baseline)

    left_paths  = sorted(glob.glob(os.path.join(pairs_dir, 'cam0', '*.jpg')))
    right_paths = sorted(glob.glob(os.path.join(pairs_dir, 'cam1', '*.jpg')))
    if not left_paths:
        print(f'  [skip] No stereo pairs found in {pairs_dir}/cam0 — skipping.')
        return

    os.makedirs(out_dir, exist_ok=True)
    cam0, dist0, cam1, dist1, R, T = get_calibration(baseline, baseline_dir)

    size = cv.imread(left_paths[0]).shape[:2][::-1]
    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(cam0, dist0, cam1, dist1, size, R, T, alpha=0)
    map_lx, map_ly = cv.initUndistortRectifyMap(cam0, dist0, R1, P1, size, cv.CV_32FC1)
    map_rx, map_ry = cv.initUndistortRectifyMap(cam1, dist1, R2, P2, size, cv.CV_32FC1)

    print(f'  Processing {len(left_paths)} pairs -> {out_dir}')
    for idx, (lp, rp) in enumerate(zip(left_paths, right_paths), start=1):
        rect_l = cv.remap(cv.imread(lp), map_lx, map_ly, cv.INTER_LINEAR)
        rect_r = cv.remap(cv.imread(rp), map_rx, map_ry, cv.INTER_LINEAR)
        process_pair(idx, rect_l, rect_r, Q, cam0, out_dir)


def main():
    root = os.getcwd()
    images_root = os.path.join(root, IMAGES_ROOT)

    baselines = sorted([
        d for d in os.listdir(images_root)
        if os.path.isdir(os.path.join(images_root, d))
        and os.path.isdir(os.path.join(images_root, d, 'cam0'))
        and os.path.isdir(os.path.join(images_root, d, 'cam1'))
    ])

    if not baselines:
        print(f'No baseline folders found under {images_root}')
        return

    print(f'Found {len(baselines)} baseline(s): {baselines}')
    os.makedirs(OUT_ROOT, exist_ok=True)

    for baseline in baselines:
        print(f'\n{"=" * 60}')
        print(f'Baseline: {baseline}')
        print('=' * 60)
        process_baseline(baseline)

    print(f'\nAll done. Results in: {OUT_ROOT}/')


if __name__ == '__main__':
    main()
