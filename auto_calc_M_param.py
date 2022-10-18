import pathlib
import cv2
import numpy as np
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)
import argparse
import torch
from models.matching import Matching
import matplotlib.cm as cm
from models.utils import make_matching_plot_fast

torch.set_grad_enabled(False)

default_M = [[2.34594784e+00, -1.06349344e-02,  1.59308517e+02],
     [-5.28646524e-02,  2.41858003e+00,  1.58510310e+02],
     [-3.03681653e-05, -1.43832960e-04,  1.00000000e+00]]
default_M = np.mat(default_M)
default_MINV = np.linalg.inv(default_M)

def match_pair(img_path0, img_path1, resize=[1800,1600]):

    opt = argparse.Namespace(
        nms_radius=4,
        keypoint_threshold=0.005,
        max_keypoints=1024,
        # superglue='indoor', # or 'outdoor
        superglue='outdoor', # or 'indoor
        sinkhorn_iterations=20,
        match_threshold=0.2,
        resize=resize,
        resize_float=True,
    )

    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    rot0, rot1 = 0, 0
    image0, inp0, scales0 = read_image(
        img_path0, device, opt.resize, rot0, opt.resize_float)
    image1, inp1, scales1 = read_image(
        img_path1, device, opt.resize, rot1, opt.resize_float)

    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    # Write the matches to disk.
    out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                    'matches': matches, 'match_confidence': conf}

    return out_matches


# J:/workspace/GitHub/YOLOX/minimal_set/convert_rects.py
if __name__ == "__main__":
    index = "05363"
    # index = "04927"
    index = '00283'
    index = '00283'
    index = '00284'
    # index = '00320'

    fir_path = f"J:/dataset/flir/FLIR_ADAS_1_3/train/thermal_8_bit/FLIR_{index}.jpeg"
    rgb_path = f"J:/dataset/flir/FLIR_ADAS_1_3/train/RGB/FLIR_{index}.jpg"

    
    rgb_img = cv2.imread(rgb_path)
    fir_img = cv2.imread(fir_path)
    resize=[rgb_img.shape[1],rgb_img.shape[0],]
    rgb_gray_img = 255 - cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    tmp_p= pathlib.Path(rgb_path)
    gray_rgb_path = pathlib.Path(tmp_p.stem+'gray_tmp'+tmp_p.suffix)
    cv2.imwrite(str(gray_rgb_path),rgb_gray_img)
    

    # out_matches = match_pair(gray_rgb_path, fir_path, resize)
    out_matches = match_pair(gray_rgb_path, gray_rgb_path, resize)
    gray_rgb_path.unlink()

    rgb_pt = out_matches["keypoints0"].astype(int)
    fir_pt = out_matches["keypoints1"].astype(int)
    print(rgb_pt.shape)
    
    vaild = out_matches["matches"] > -1
    rgb_pt = rgb_pt[vaild]
    fir_pt = fir_pt[out_matches["matches"][vaild]]

    # print(rgb_pt)
    # print(fir_pt)

    M, mask = cv2.findHomography(fir_pt, rgb_pt, cv2.RANSAC, 5.0)

    M = default_M

    # rgb_img_convert = cv2.warpPerspective(cv2.resize(fir_img, resize), M, (rgb_gray_img.shape[1], rgb_gray_img.shape[0]))
    rgb_img_convert = cv2.warpPerspective(fir_img, M, (rgb_gray_img.shape[1], rgb_gray_img.shape[0]))
    blend_img = cv2.addWeighted(rgb_img, 0.5, rgb_img_convert, 0.5, 1)

    
    cv2.imshow('match', cv2.resize(blend_img, (blend_img.shape[1]//2,blend_img.shape[0]//2)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # img3 = cv2.drawMatches(rgb_gray_img, kp1, fir_img, kp2, matches[:10], None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # mmcv.visualization.imshow(
    #     img3,
    #     '',
    #     wait_time=0
    # )

    # mmcv.visualization.imshow(
    #     fir_img,
    #     '',
    #     wait_time=0
    # )

    # mmcv.visualization.imshow(
    #     cv2.resize(rgb_gray_img, (rgb_gray_img.shape[1]//2,rgb_gray_img.shape[0]//2)),
    #     '',
    #     wait_time=0
    # )
