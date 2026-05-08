import numpy as np
import open3d as o3d
import cv2 as cv
import glob
import os

ZFAR              = 30000

X_FILTER_ENABLED  = False
X_FILTER_RANGE    = 2000

Y_FILTER_ENABLED  = False
Y_FILTER_RANGE    = 2000

SOR_ENABLED       = False    # statistical outlier removal
SOR_NB_NEIGHBORS  = 30
SOR_STD_RATIO     = 1.5

DENOISE_CLOUD      = True  # voxel downsample + radius outlier removal
DENOISE_VOXEL_SIZE = 1.0
DENOISE_NB_POINTS  = 30
DENOISE_RADIUS     = 30.0


def calibrate(showPics=True, imgPath=None):

    print('--------- single ' + str(imgPath) + ' camera calibration ---------')

    root = os.getcwd()

    if imgPath is None:
        print("Error: No imgPath.")
        return None, None

    calibrationDir = os.path.join(root, 'images', imgPath)
    imgPathList = glob.glob(os.path.join(calibrationDir, "*.jpg"))

    if len(imgPathList) == 0:
        print("Error: No images found in directory.")
        return None, None

    nRows = 8
    nCols = 11
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2) * 60

    worldPtsList = []
    imgPtsList = []
    validImgPaths = []
    imageSize = None

    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)

        if imageSize is None:
            imageSize = imgGray.shape[::-1]

        cornersFound, cornersOrg = cv.findChessboardCorners(
            imgGray, (nRows, nCols),
            cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE | cv.CALIB_CB_FAST_CHECK
        )

        if cornersFound:
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (13, 13), (-1, -1), termCriteria)
            imgPtsList.append(cornersRefined)
            validImgPaths.append(curImgPath)

    if not imgPtsList:
        print("Error: Images found, but no chessboard corners in any of them.")
        return None, None

    repError, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(
        worldPtsList, imgPtsList, imageSize, None, None
    )

    print('Camera Matrix:\n', camMatrix)
    print('Distortion Coeff:\n', distCoeff)
    print('Overall Reproj Error (px): {:.4f}'.format(repError))

    if showPics:
        for i, curImgPath in enumerate(validImgPaths):
            imgBGR = cv.imread(curImgPath)
            cornersRefined = imgPtsList[i]
            cv.drawChessboardCorners(imgBGR, (nRows, nCols), cornersRefined, True)

            imgPtsProjected, _ = cv.projectPoints(worldPtsCur, rvecs[i], tvecs[i], camMatrix, distCoeff)
            error = cv.norm(cornersRefined, imgPtsProjected, cv.NORM_L2) / len(imgPtsProjected)

            imgName = os.path.basename(curImgPath)
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(imgBGR, f'File: {imgName}', (20, 40), font, 1, (255, 255, 0), 2, cv.LINE_AA)
            cv.putText(imgBGR, f'Image Error: {error * 10:.4f} px', (20, 80), font, 1, (0, 255, 0), 2, cv.LINE_AA)

            cv.imshow('img', imgBGR)
            cv.waitKey(0)

        cv.destroyAllWindows()

    return camMatrix, distCoeff


def stereoCalibration(camMatrix0, distCoeff0, camMatrix1, distCoeff1, imgSet):
    print('--------- stereo calibration ---------')
    root = os.getcwd()

    leftPathList  = sorted(glob.glob(os.path.join(root, 'images', imgSet, 'cam0', "*.jpg")))
    rightPathList = sorted(glob.glob(os.path.join(root, 'images', imgSet, 'cam1', "*.jpg")))

    nRows = 8
    nCols = 11
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2) * 60

    worldPtsList = []
    imgPtsLeft = []
    imgPtsRight = []
    imageSize = None

    for leftPath, rightPath in zip(leftPathList, rightPathList):
        imgLeft  = cv.imread(leftPath)
        imgRight = cv.imread(rightPath)

        grayLeft  = cv.cvtColor(imgLeft,  cv.COLOR_BGR2GRAY)
        grayRight = cv.cvtColor(imgRight, cv.COLOR_BGR2GRAY)

        if imageSize is None:
            imageSize = grayLeft.shape[::-1]

        foundLeft, cornersLeft = cv.findChessboardCorners(
            grayLeft, (nRows, nCols),
            cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
        )
        foundRight, cornersRight = cv.findChessboardCorners(
            grayRight, (nRows, nCols),
            cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
        )

        if foundLeft and foundRight:
            worldPtsList.append(worldPtsCur)
            refinedLeft  = cv.cornerSubPix(grayLeft,  cornersLeft,  (13, 13), (-1, -1), termCriteria)
            refinedRight = cv.cornerSubPix(grayRight, cornersRight, (13, 13), (-1, -1), termCriteria)
            imgPtsLeft.append(refinedLeft)
            imgPtsRight.append(refinedRight)

    retStereo, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
        worldPtsList, imgPtsLeft, imgPtsRight,
        camMatrix0, distCoeff0,
        camMatrix1, distCoeff1,
        imageSize, criteria=termCriteria, flags=cv.CALIB_FIX_INTRINSIC)

    print(f'Stereo Reprojection Error: {retStereo:.4f}')
    print('\nRotation Matrix (R):\n', R)
    print('\nTranslation Vector (T):\n', T)

    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder, 'stereo_calibration.npz')
    np.savez(paramPath, R=R, T=T, E=E, F=F,
             camMatrix0=camMatrix0, distCoeff0=distCoeff0,
             camMatrix1=camMatrix1, distCoeff1=distCoeff1)

    print(f"Baseline is: {abs(T[0][0]):.2f} mm")
    return R, T


def rectify_images(camMatrix0, distCoeff0, camMatrix1, distCoeff1, R, T, imgSet, imgName):
    print('--------- stereo rectification ---------')

    imgLeft  = cv.imread(f'images/{imgSet}/cam0/cam0_{imgName}.jpg')
    imgRight = cv.imread(f'images/{imgSet}/cam1/cam1_{imgName}.jpg')

    if imgLeft is None or imgRight is None:
        print("Error loading rectification test images.")
        return

    imageSize = imgLeft.shape[:2][::-1]

    R1, R2, P1, P2, Q, roi_left, roi_right = cv.stereoRectify(
        camMatrix0, distCoeff0,
        camMatrix1, distCoeff1,
        imageSize, R, T,
        flags=cv.CALIB_ZERO_DISPARITY, alpha=0)

    mapLeftX,  mapLeftY  = cv.initUndistortRectifyMap(camMatrix0, distCoeff0, R1, P1, imageSize, cv.CV_32FC1)
    mapRightX, mapRightY = cv.initUndistortRectifyMap(camMatrix1, distCoeff1, R2, P2, imageSize, cv.CV_32FC1)

    rectifiedLeft  = cv.remap(imgLeft,  mapLeftX,  mapLeftY,  cv.INTER_LINEAR)
    rectifiedRight = cv.remap(imgRight, mapRightX, mapRightY, cv.INTER_LINEAR)

    combinedImg = np.hstack((rectifiedLeft, rectifiedRight))
    for y in range(0, combinedImg.shape[0], 40):
        cv.line(combinedImg, (0, y), (combinedImg.shape[1], y), (0, 255, 0), 1)

    cv.namedWindow('Rectified Stereo Pair', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.resizeWindow('Rectified Stereo Pair', 1920, 720)
    cv.imshow('Rectified Stereo Pair', combinedImg)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return mapLeftX, mapLeftY, mapRightX, mapRightY, Q


def main(imgSet, imgName):
    try:
        calib = np.load('stereo_calibration.npz')
    except FileNotFoundError:
        print("Error: Could not find stereo_calibration.npz.")
        return

    imgLeft  = cv.imread(f'images/{imgSet}/cam0/cam0_{imgName}.jpg')
    imgRight = cv.imread(f'images/{imgSet}/cam1/cam1_{imgName}.jpg')

    if imgLeft is None or imgRight is None:
        print("Error: Check your image paths.")
        return

    size = imgLeft.shape[:2][::-1]

    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
        calib['camMatrix0'], calib['distCoeff0'],
        calib['camMatrix1'], calib['distCoeff1'],
        size, calib['R'], calib['T'], alpha=0
    )

    mapLX, mapLY = cv.initUndistortRectifyMap(calib['camMatrix0'], calib['distCoeff0'], R1, P1, size, cv.CV_32FC1)
    mapRX, mapRY = cv.initUndistortRectifyMap(calib['camMatrix1'], calib['distCoeff1'], R2, P2, size, cv.CV_32FC1)

    rectL = cv.remap(imgLeft,  mapLX, mapLY, cv.INTER_LINEAR)
    rectR = cv.remap(imgRight, mapRX, mapRY, cv.INTER_LINEAR)
    grayL = cv.cvtColor(rectL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(rectR, cv.COLOR_BGR2GRAY)

    disparity, min_disp = interactive_tuner(grayL, grayR)

    if disparity is not None:
        camMatrix = calib['camMatrix0']
        reconstruct_point_cloud(disparity, Q, rectL, min_disp, camMatrix)


def interactive_tuner(grayL, grayR):
    h, w = grayL.shape[:2]
    target_width = 1456
    scale = min(1.0, target_width / (w * 2))
    disp_w = int(w * scale)
    disp_h = int(h * scale)
    print(f"  Display scale: {scale:.2f} ({w}x{h} -> {disp_w}x{disp_h} per panel)")

    win_name = 'SGBM'
    cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)

    def nothing(x):
        pass

    cv.createTrackbar('Num Disparities (x16)', win_name, 16, 32, nothing)
    cv.createTrackbar('Block Size (Odd)', win_name, 2, 15, nothing)
    cv.createTrackbar('Min Disparity', win_name, 0, 32, nothing)
    cv.createTrackbar('Uniqueness Ratio', win_name, 10, 100, nothing)
    cv.createTrackbar('Speckle Window', win_name, 200, 300, nothing)
    cv.createTrackbar('Speckle Range', win_name, 1, 50, nothing)
    cv.createTrackbar('P1 Scale', win_name, 8, 32, nothing)
    cv.createTrackbar('P2 Scale', win_name, 32, 128, nothing)
    cv.createTrackbar('PreFilter Cap', win_name, 63, 127, nothing)

    disparity = None
    min_disp  = 0
    valid_mask = None

    while True:
        num_disp    = max(16, cv.getTrackbarPos('Num Disparities (x16)', win_name) * 16)
        block_size  = max(3,  cv.getTrackbarPos('Block Size (Odd)', win_name) * 2 + 1)
        min_disp    = cv.getTrackbarPos('Min Disparity', win_name)
        uniqueness  = cv.getTrackbarPos('Uniqueness Ratio', win_name)
        speckle_win = cv.getTrackbarPos('Speckle Window', win_name)
        speckle_rng = max(1, cv.getTrackbarPos('Speckle Range', win_name))
        p1_scale    = max(1, cv.getTrackbarPos('P1 Scale', win_name))
        p2_scale    = max(p1_scale + 1, cv.getTrackbarPos('P2 Scale', win_name))
        prefilter   = max(1, cv.getTrackbarPos('PreFilter Cap', win_name))

        stereo = cv.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=p1_scale * 3 * block_size ** 2,
            P2=p2_scale * 3 * block_size ** 2,
            uniquenessRatio=uniqueness,
            speckleWindowSize=speckle_win,
            speckleRange=speckle_rng,
            disp12MaxDiff=1,
            preFilterCap=prefilter,
            mode=cv.STEREO_SGBM_MODE_HH
        )

        disp_raw   = stereo.compute(grayL, grayR)
        disparity  = disp_raw.astype(np.float32) / 16.0
        valid_mask = disparity > min_disp

        disp_clean = disparity.copy()
        disp_clean[~valid_mask] = 0
        mask = disp_clean > min_disp
        norm_disp = np.zeros(disp_clean.shape, dtype=np.uint8)
        if mask.any():
            norm_disp[mask] = cv.normalize(
                disp_clean[mask], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U
            ).flatten()
        color_disp = cv.applyColorMap(norm_disp, cv.COLORMAP_TURBO)
        color_disp[~mask] = [0, 0, 0]

        panel_left = cv.resize(cv.cvtColor(grayL, cv.COLOR_GRAY2BGR), (disp_w, disp_h), interpolation=cv.INTER_AREA)
        panel_disp = cv.resize(color_disp, (disp_w, disp_h), interpolation=cv.INTER_AREA)
        cv.imshow(win_name, np.hstack((panel_left, panel_disp)))

        key = cv.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):
            break
    cv.destroyAllWindows()

    if valid_mask is not None:
        disparity[~valid_mask] = 0
    return disparity, min_disp


def reconstruct_point_cloud(disparity, Q, rectLeft, min_disp, camMatrix):
    """Back-projects disparity to a point cloud. All spatial values are in mm."""
    fx = camMatrix[0, 0]
    cx = camMatrix[0, 2]
    cy = camMatrix[1, 2]

    H, W = disparity.shape
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))

    valid = disparity > min_disp

    pts3d = cv.reprojectImageTo3D(disparity, Q, handleMissingValues=False)
    z_mm  = pts3d[:, :, 2]

    x_mm = (uu - cx) * z_mm / fx
    y_mm = (vv - cy) * z_mm / fx

    pts_mm  = np.stack([x_mm, y_mm, z_mm], axis=-1).reshape(-1, 3)
    colors  = rectLeft[:, :, ::-1].reshape(-1, 3).astype(np.float64) / 255.0
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

    print(f"  Z  <= {ZFAR} mm")
    if X_FILTER_ENABLED:
        print(f"  X  in [{-X_FILTER_RANGE}, +{X_FILTER_RANGE}] mm")
    if Y_FILTER_ENABLED:
        print(f"  Y  in [{-Y_FILTER_RANGE}, +{Y_FILTER_RANGE}] mm")
    print(f"  {len(pts_mm)} points after filtering")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_mm)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if SOR_ENABLED and len(pcd.points) > SOR_NB_NEIGHBORS:
        print("Running Statistical Outlier Removal...")
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO)
        pcd = pcd.select_by_index(ind)
        print(f"  SOR done. {len(np.asarray(pcd.points))} points remaining.")

    output_path = os.path.join(os.getcwd(), 'point_cloud.ply')
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved point cloud to: {output_path}")

    try:
        o3d.visualization.draw_geometries(
            [pcd],
            window_name='SGBM 3D Point Cloud',
            width=1456,
            height=816
        )
    except Exception as e:
        print(f"[WARNING] Open3D viewer failed ({e}). "
              f"Point cloud saved to: {output_path} — open it in MeshLab or CloudCompare.")


if __name__ == '__main__':
    useCalib = False
    imgSet   = '188'
    imgSet3d = '188/3d'
    imgName  = '20260410_170144'

    if useCalib:
        camMatrix0, distCoeff0 = calibrate(showPics=False, imgPath=f'{imgSet}/cam0')
        camMatrix1, distCoeff1 = calibrate(showPics=False, imgPath=f'{imgSet}/cam1')

        if camMatrix0 is not None and camMatrix1 is not None:
            R, T = stereoCalibration(camMatrix0, distCoeff0, camMatrix1, distCoeff1, imgSet)
            rectify_images(camMatrix0, distCoeff0, camMatrix1, distCoeff1, R, T, imgSet3d, imgName)

    main(imgSet3d, imgName)
