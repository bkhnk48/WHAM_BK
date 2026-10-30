import os
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict

import cv2
import torch
import joblib
import numpy as np
from loguru import logger
from progress.bar import Bar

from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify
import pdb
import json
import shutil


try: 
    from lib.models.preproc.slam import SLAMModel
    _run_global = True
except: 
    logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
    _run_global = False

#hàm tạo và xuất calibration
def export_calibration(video, output_pth, calib=None):
    """
    Export camera intrinsics used by WHAM/DPVO.

    Format:
        fx fy cx cy

    Priority:
        1. User-provided calibration file (if valid).
        2. Existing output_pth/calib.txt (if valid).
        3. WHAM heuristic based on video resolution.

    IMPORTANT:
        WHAM heuristic is NOT physical camera calibration.
    """

    os.makedirs(output_pth, exist_ok=True)

    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid video resolution: {width}x{height}"
        )

    output_calib = osp.join(output_pth, "calib.txt")
    metadata_file = osp.join(
        output_pth, "calib_metadata.json"
    )

    def read_calibration(path):
        values = np.loadtxt(path, dtype=np.float64).reshape(-1)

        if len(values) != 4:
            raise ValueError(
                f"Calibration must contain exactly 4 values: {path}"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Calibration contains invalid values: {path}"
            )

        fx, fy, cx, cy = values

        if fx <= 0 or fy <= 0:
            raise ValueError(
                f"Focal lengths must be positive: {path}"
            )

        return values

    # 1. User-provided calibration
    if calib is not None and osp.isfile(calib):

        intrinsics = read_calibration(calib)
        source = "user_provided"

        if osp.abspath(calib) != osp.abspath(output_calib):
            np.savetxt(
                output_calib,
                intrinsics.reshape(1, 4),
                fmt="%.8f"
            )

    # 2. Existing calibration in output directory
    elif osp.isfile(output_calib):

        intrinsics = read_calibration(output_calib)
        source = "existing_output_calibration"

    # 3. WHAM heuristic
    else:

        focal_length = np.sqrt(
            width ** 2 + height ** 2
        )

        fx = focal_length
        fy = focal_length
        cx = width / 2.0
        cy = height / 2.0

        intrinsics = np.array(
            [fx, fy, cx, cy],
            dtype=np.float64
        )

        source = "wham_heuristic"

        np.savetxt(
            output_calib,
            intrinsics.reshape(1, 4),
            fmt="%.8f"
        )

    fx, fy, cx, cy = intrinsics

    metadata = {
        "video": osp.abspath(video),
        "resolution": {
            "width": width,
            "height": height
        },
        "fps": fps,
        "frame_count": frame_count,
        "intrinsics": {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy)
        },
        "calibration_source": source,
        "coordinate_unit": "pixels",
        "format": "fx fy cx cy",
        "is_physical_calibration": (
            source == "user_provided"
        ),
        "description": (
            "WHAM/DPVO camera intrinsics. "
            "WHAM heuristic values are assumed intrinsics, "
            "not physically calibrated camera parameters."
        )
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(
        f"Camera calibration saved: {output_calib}"
    )

    logger.info(
        f"Calibration source: {source}"
    )

    logger.info(
        f"Intrinsics: fx={fx:.4f}, fy={fy:.4f}, "
        f"cx={cx:.4f}, cy={cy:.4f}"
    )

    return output_calib, intrinsics


def run(cfg,
        video,
        output_pth,
        network,
        calib=None,
        run_global=True,
        save_pkl=False,
        visualize=False,
        mp4=False):
    
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Faild to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # Use the input video filename as the output prefix.
    video_name = osp.splitext(osp.basename(video))[0]

    tracking_file = osp.join(
        output_pth, f'tracking_results_{video_name}.pth'
    )
    slam_file = osp.join(
        output_pth, f'slam_result_{video_name}.pth'
    )
    wham_file = osp.join(
        output_pth, f'wham_{video_name}.pkl'
    )
    rendered_file = osp.join(
        output_pth, f'output_{video_name}.mp4'
    )
    
    # Whether or not estimating motion in global coordinates
    run_global = run_global and _run_global
    
    # Preprocess
    with torch.no_grad():
        if not (osp.exists(tracking_file) and osp.exists(slam_file)):
            
            detector = DetectionModel(cfg.DEVICE.lower())
            extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
            
            if run_global: slam = SLAMModel(video, output_pth, width, height, calib)
            else: slam = None
            
            bar = Bar('Preprocess: 2D detection and SLAM', fill='#', max=length)
            while (cap.isOpened()):
                flag, img = cap.read()
                if not flag: break
                
                # 2D detection and tracking
                detector.track(img, fps, length)
                
                # SLAM
                if slam is not None: 
                    slam.track()
                
                bar.next()

            tracking_results = detector.process(fps)
            
            if slam is not None: 
                slam_results = slam.process()
            else:
                slam_results = np.zeros((length, 7))
                slam_results[:, 3] = 1.0    # Unit quaternion
        
            # Extract image features
            # TODO: Merge this into the previous while loop with an online bbox smoothing.
            tracking_results = extractor.run(video, tracking_results)
            logger.info('Complete Data preprocessing!')
            
            # Save the processed data
            joblib.dump(tracking_results, tracking_file)
            joblib.dump(slam_results, slam_file)

            logger.info(f'Saved tracking results: {tracking_file}')
            logger.info(f'Saved SLAM results: {slam_file}')
        
        # If the processed data already exists, load the processed data
        else:
            tracking_results = joblib.load(tracking_file)
            slam_results = joblib.load(slam_file)

            logger.info(f'Loaded tracking results: {tracking_file}')
            logger.info(f'Loaded SLAM results: {slam_file}')
            logger.info(f'Already processed data exists at {output_pth} ! Load the data .')
    
    # Build dataset
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)
    
    # run WHAM
    results = defaultdict(dict)
    
    n_subjs = len(dataset)
    for subj in range(n_subjs):

        with torch.no_grad():
            if cfg.FLIP_EVAL:
                # Forward pass with flipped input
                flipped_batch = dataset.load_data(subj, True)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = flipped_batch
                flipped_pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # Forward pass with normal input
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # Merge two predictions
                flipped_pose, flipped_shape = flipped_pred['pose'].squeeze(0), flipped_pred['betas'].squeeze(0)
                pose, shape = pred['pose'].squeeze(0), pred['betas'].squeeze(0)
                flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(-1, 24, 6)
                avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
                avg_pose = avg_pose.reshape(-1, 144)
                avg_contact = (flipped_pred['contact'][..., [2, 3, 0, 1]] + pred['contact']) / 2
                
                # Refine trajectory with merged prediction
                network.pred_pose = avg_pose.view_as(network.pred_pose)
                network.pred_shape = avg_shape.view_as(network.pred_shape)
                network.pred_contact = avg_contact.view_as(network.pred_contact)
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
            
            else:
                # data
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                
                # inference
                pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
        
        # if False:
        if args.run_smplify:
            smplify = TemporalSMPLify(smpl, img_w=width, img_h=height, device=cfg.DEVICE)
            input_keypoints = dataset.tracking_results[_id]['keypoints']
            pred = smplify.fit(pred, input_keypoints, **kwargs)
            
            with torch.no_grad():
                network.pred_pose = pred['pose']
                network.pred_shape = pred['betas']
                network.pred_cam = pred['cam']
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
        
        # ========= Store results ========= #
        pred_body_pose = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)
        pred_root = matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)
        pred_root_world = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)
        pred_pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
        pred_pose_world = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
        pred_trans = (pred['trans_cam'] - network.output.offset).cpu().numpy()
        
        results[_id]['pose'] = pred_pose
        results[_id]['trans'] = pred_trans
        results[_id]['pose_world'] = pred_pose_world
        results[_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
        results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
        results[_id]['verts'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
        results[_id]['frame_ids'] = frame_id
    
    if save_pkl:
        joblib.dump(results, wham_file)
        logger.info(f'Saved WHAM results: {wham_file}')
     
    # Visualize
    if visualize:
        from lib.vis.run_vis import run_vis_on_demo
        with torch.no_grad():
            run_vis_on_demo(
                cfg,
                video,
                results,
                output_pth,
                network.smpl,
                vis_global=run_global,
                mp4=mp4,
            )

        # Rename the rendered video to output_<video_name>.mp4.
        if mp4:
            # Chỉ đổi tên khi thực sự xuất video.
            default_rendered_file = osp.join(output_pth, 'output.mp4')
            if osp.isfile(default_rendered_file):
                os.replace(default_rendered_file, rendered_file)
                logger.info(f'Saved rendered video: {rendered_file}')
            else:
                logger.warning(
        	    f'Rendered video was not found: {default_rendered_file}'
        	)
        else:
            logger.info(
        	f'Rendered JPG frames saved to: {osp.join(output_pth, "mp4")}'
            )
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str, 
                        default='examples/demo_video.mp4', 
                        help='input video path or youtube link')

    parser.add_argument('--output_pth', type=str, default='output/demo', 
                        help='output folder to write results')
    
    parser.add_argument('--calib', type=str, default=None, 
                        help='Camera calibration file path')

    parser.add_argument('--estimate_local_only', action='store_true',
                        help='Only estimate motion in camera coordinate if True')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the output mesh if True')
                        
    parser.add_argument('--mp4', action='store_true',
                        help='Save rendered results as an MP4 video. Without this flag, save rendered frames as JPG images.')
    
    parser.add_argument('--save_pkl', action='store_true',
                        help='Save output as pkl file')
    
    parser.add_argument('--run_smplify', action='store_true',
                        help='Run Temporal SMPLify for post processing')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo_cpu.yaml')
    
    if torch.cuda.is_available():
        logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
        logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')
    else:
        logger.info('CUDA unavailable; attempting CPU inference.') 
    
    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    
    # Output folder
    sequence = osp.splitext(osp.basename(args.video))[0]
    output_pth = osp.join(args.output_pth, sequence)
    os.makedirs(output_pth, exist_ok=True)
    
    run(cfg, 
        args.video, 
        output_pth, 
        network, 
        args.calib, 
        run_global=not args.estimate_local_only, 
        save_pkl=args.save_pkl,
        visualize=args.visualize,
        mp4=args.mp4)
        
    print()
    logger.info('Done !')
