# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import errno
import os
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import torch
import torch.nn as nn
import torch.optim as optim

from common.arguments import parse_args
from common.camera import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from common.loss import *
from common.model_poseformer import *
from common.utils import *

args = parse_args()

checkpoint_dir = os.path.join(args.checkpoint, args.exp_name)

try:
    # Create checkpoint directory if it does not exist
    if not args.evaluate:
        os.makedirs(checkpoint_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError("Unable to create checkpoint directory:", checkpoint_dir)

print("Loading dataset...")
dataset_path = "data/data_3d_" + args.dataset + ".npz"
if args.dataset == "h36m":
    from common.h36m_dataset import Human36mDataset

    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith("humaneva"):
    from common.humaneva_dataset import HumanEvaDataset

    dataset = HumanEvaDataset(dataset_path)
elif args.dataset.startswith("force_pose"):
    from common.force_pose_dataset import ForcePoseDataset

    dataset = ForcePoseDataset(dataset_path)
elif args.dataset.startswith("parkour"):
    from common.parkour_dataset import ParkourDataset

    dataset = ParkourDataset(dataset_path)
elif args.dataset.startswith("custom"):
    from common.custom_dataset import CustomDataset

    dataset = CustomDataset(
        "data/data_2d_" + args.dataset + "_" + args.keypoints + ".npz"
    )
else:
    raise KeyError("Invalid dataset")

titles = ["Fx1", "Fy1", "Fz1", "Fx2", "Fy2", "Fz2"]
title_groups = {
    "Medio-Lateral (Fx)": ["Fx1", "Fx2"],
    "Vertical (Fy)": ["Fy1", "Fy2"],
    "Anterior-Posterior (Fz)": ["Fz1", "Fz2"],
}

# Thresholds for force_mse loss
thresholds = [0]  # N/kg
num_thresh = args.num_force_thresh  # Defaults to 1
force_res = 5  # N/kg, resolution between thresholds
for idx in range(1, num_thresh):
    if idx == 1:
        thresholds.append(1)
    else:
        thresholds.append((idx - 1) * force_res)
print(f"Thresholds: {thresholds}")

print("Preparing data...")
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        if "positions" in anim:
            positions_3d = []
            for cam in anim["cameras"]:
                if (
                    args.input_pose_type in ["2d", "3d"]
                    and "positions_triangulated" in anim
                ):
                    pos_3d = world_to_camera(
                        anim["positions_triangulated"],
                        R=cam["orientation"],
                        t=cam["translation"],
                    )

                    # from tools.visualization import draw_pose
                    # draw_pose('cam0', subject, action, np.copy(pos_3d))
                else:
                    pos_3d = world_to_camera(
                        anim["positions"], R=cam["orientation"], t=cam["translation"]
                    )
                    # if not args.dataset.startswith('parkour'):
                    #    pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position

                pos_3d /= 1000  # scale down values
                positions_3d.append(pos_3d)
            anim["positions_3d"] = positions_3d

print("Loading 2D detections...")
keypoints = np.load(
    "data/data_2d_" + args.dataset + "_" + args.keypoints + ".npz", allow_pickle=True
)
keypoints_metadata = keypoints["metadata"].item()
keypoints_symmetry = keypoints_metadata["keypoints_symmetry"]
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = (
    list(dataset.skeleton().joints_left()),
    list(dataset.skeleton().joints_right()),
)
if "cam_names" in keypoints:
    cam_names = keypoints["cam_names"].tolist()
else:
    cam_names = None
keypoints = keypoints["positions_2d"].item()

# Enable only certain cameras
filter_cameras = None if not args.filter_cameras else args.filter_cameras
cam_idxs_to_remove = []
if filter_cameras is not None:
    print(f"Filter cameras: {filter_cameras}")
    subject = list(dataset.subjects())[0]
    action = list(dataset[subject].keys())[0]
    cameras = dataset[subject][action]["cameras"]
    for idx, cam in enumerate(cameras):
        if cam["id"] not in filter_cameras:
            cam_idxs_to_remove.append(idx)
    cam_idxs_to_remove = sorted(cam_idxs_to_remove, reverse=True)

###################
for subject in dataset.subjects():
    assert subject in keypoints, (
        f"Subject {subject} is missing from the 2D detections dataset"
    )
    for idx in cam_idxs_to_remove:  # The camera list is a class variable shared between all subjects - can only be deleted once
        del dataset[subject][list(dataset[subject].keys())[0]]["cameras"][idx]

    for action in dataset[subject].keys():
        assert action in keypoints[subject], (
            f"Action {action} of subject {subject} is missing from the 2D detections dataset"
        )
        if "positions_3d" not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):
            if cam_names is not None and filter_cameras is not None:
                cam_name = cam_names[cam_idx]
                if cam_name not in filter_cameras:
                    continue

            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]["positions_3d"][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][
                    cam_idx
                ][:mocap_length]

        for idx in cam_idxs_to_remove:
            del dataset[subject][action]["positions_3d"][idx]
            del keypoints[subject][action][idx]

        assert len(keypoints[subject][action]) == len(
            dataset[subject][action]["positions_3d"]
        )

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(
                kps[..., :2], w=cam["res_w"], h=cam["res_h"]
            )
            keypoints[subject][action][cam_idx] = kps

subjects_train = args.subjects_train.split(",")
subjects_semi = (
    [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(",")
)
if not args.render:
    subjects_test = args.subjects_test.split(",")
else:
    subjects_test = [args.viz_subject]


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_forces = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i][:, :, :2])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), "Camera count mismatch"
                for cam in cams:
                    if "intrinsic" in cam:
                        out_camera_params.append(cam["intrinsic"])

            if parse_3d_poses and "positions_3d" in dataset[subject][action]:
                poses_3d = dataset[subject][action]["positions_3d"]
                assert len(poses_3d) == len(poses_2d), "Camera count mismatch"
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

            if "forces" in dataset[subject][action]:
                forces = dataset[subject][action]["forces"]
                for i in range(
                    len(poses_2d)
                ):  # Iterate across cameras (repeat same forces for each view)
                    assert forces.shape[0] == len(poses_2d[i]), "Sequence mismatch"
                    out_forces.append(forces)

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None
    if len(out_forces) == 0:
        out_forces = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            start = deterministic_random(
                0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i]))
            )
            out_poses_2d[i] = out_poses_2d[i][start : start + n_frames : stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start : start + n_frames : stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d, out_forces


action_filter = None if args.actions == "*" else args.actions.split(",")
if action_filter is not None:
    print("Selected actions:", action_filter)

cameras_valid, poses_valid, poses_valid_2d, forces_valid = fetch(
    subjects_test, action_filter
)


receptive_field = args.number_of_frames
print(f"INFO: Receptive field: {receptive_field} frames")
pad = (receptive_field - 1) // 2  # Padding on each side
min_loss = 100000
width = cam["res_w"]
height = cam["res_h"]
if args.input_pose_type == "mocap":
    num_joints = 47  # 47 MoCap Keypoints
else:
    num_joints = keypoints_metadata["num_joints"]

#########################################PoseTransformer
if args.input_pose_type in ["3d", "mocap"]:
    in_chans = 3
else:
    in_chans = 2
model_pos_train = PoseTransformer(
    num_frame=receptive_field,
    num_joints=num_joints,
    in_chans=in_chans,
    embed_dim_ratio=32,
    depth=4,
    num_heads=8,
    mlp_ratio=2.0,
    qkv_bias=True,
    qk_scale=None,
    drop_path_rate=0.1,
    pred_force=True,
    multitask=args.multitask,
)

model_pos = PoseTransformer(
    num_frame=receptive_field,
    num_joints=num_joints,
    in_chans=in_chans,
    embed_dim_ratio=32,
    depth=4,
    num_heads=8,
    mlp_ratio=2.0,
    qkv_bias=True,
    qk_scale=None,
    drop_path_rate=0,
    pred_force=True,
    multitask=args.multitask,
)

#################
causal_shift = 0
model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print("INFO: Trainable parameter count:", model_params)

if torch.cuda.is_available():
    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()
    model_pos_train = nn.DataParallel(model_pos_train)
    model_pos_train = model_pos_train.cuda()


if args.resume or args.evaluate or args.pretrained:
    if args.resume:
        chk_name = args.resume
    elif args.pretrained:
        chk_name = args.pretrained
    else:
        chk_name = args.evaluate
    chk_filename = os.path.join(args.checkpoint, chk_name)
    print("Loading checkpoint", chk_filename)
    checkpoint = torch.load(
        chk_filename, map_location=lambda storage, loc: storage, weights_only=False
    )
    print("Epoch", checkpoint["epoch"])
    model_pos_train.load_state_dict(checkpoint["model_pos"], strict=False)
    model_pos.load_state_dict(checkpoint["model_pos"], strict=False)

if in_chans == 3:  # using 3D keypoints as input
    poses_valid_2d = poses_valid

test_generator = UnchunkedGenerator(
    cameras_valid,
    poses_valid,
    poses_valid_2d,
    pad=pad,
    causal_shift=causal_shift,
    augment=False,
    kps_left=kps_left,
    kps_right=kps_right,
    joints_left=joints_left,
    joints_right=joints_right,
    forces=forces_valid,
)
print(f"INFO: Testing on {test_generator.num_frames()} frames")


def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = inputs_3d.permute(1, 0, 2, 3)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(
        out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2]
    )
    for i in range(out_num):
        eval_input_2d[i, :, :, :] = inputs_2d_p[i : i + receptive_field, :, :]
    return eval_input_2d, inputs_3d_p


###################

if not args.evaluate:
    cameras_train, poses_train, poses_train_2d, forces_train = fetch(
        subjects_train, action_filter, subset=args.subset
    )
    if in_chans == 3:  # Using 3D keypoints as input
        poses_train_2d = poses_train

    lr = args.learning_rate
    optimizer = optim.AdamW(model_pos_train.parameters(), lr=lr, weight_decay=0.1)

    lr_decay = args.lr_decay
    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    train_generator = ChunkedGenerator(
        args.batch_size // args.stride,
        cameras_train,
        poses_train,
        poses_train_2d,
        args.stride,
        pad=pad,
        causal_shift=causal_shift,
        shuffle=True,
        augment=args.data_augmentation,
        kps_left=kps_left,
        kps_right=kps_right,
        joints_left=joints_left,
        joints_right=joints_right,
        forces=forces_train,
    )
    train_generator_eval = UnchunkedGenerator(
        cameras_train,
        poses_train,
        poses_train_2d,
        pad=pad,
        causal_shift=causal_shift,
        augment=False,
        forces=forces_train,
    )
    print(f"INFO: Training on {train_generator_eval.num_frames()} frames")

    if args.resume:
        epoch = checkpoint["epoch"]
        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            train_generator.set_random_state(checkpoint["random_state"])
        else:
            print(
                "WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized."
            )

        lr = checkpoint["lr"]

    print("** Note: reported losses are averaged over all frames.")
    print("** The final evaluation will be carried out after the last training epoch.")

    # Pos model only
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_traj_train = 0
        epoch_loss_2d_train_unlabeled = 0
        N = 0
        N_semi = 0
        model_pos_train.train()

        for (
            cameras_train,
            batch_3d,
            batch_2d,
            batch_grf,
        ) in train_generator.next_epoch():
            cameras_train = torch.from_numpy(cameras_train.astype("float32"))
            inputs_3d = torch.from_numpy(batch_3d.astype("float32"))
            inputs_2d = torch.from_numpy(batch_2d.astype("float32"))
            targ_grf = torch.from_numpy(batch_grf.astype("float32"))

            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
                targ_grf = targ_grf.cuda()
                cameras_train = cameras_train.cuda()

            optimizer.zero_grad()

            # Predict 6-axis Ground Reaction Force
            if args.multitask and in_chans == 2:  # Only multitask when input is 2D
                predicted_3d_pos, predicted_grf = model_pos_train(inputs_2d)
                loss_3d_pos = args.multitask_alpha * mpjpe(
                    predicted_3d_pos, inputs_3d
                ) + force_mse(predicted_grf, targ_grf, thresholds)
            else:
                predicted_3d_pos = None
                predicted_grf = model_pos_train(inputs_2d)
                loss_3d_pos = force_mse(predicted_grf, targ_grf, thresholds)

            del inputs_2d
            torch.cuda.empty_cache()

            epoch_loss_3d_train += (
                inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            )
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            loss_total = loss_3d_pos

            loss_total.backward()

            optimizer.step()
            del inputs_3d, loss_3d_pos, predicted_3d_pos, targ_grf, predicted_grf
            torch.cuda.empty_cache()

        losses_3d_train.append(epoch_loss_3d_train / N)
        torch.cuda.empty_cache()

        # End-of-epoch evaluation
        with torch.no_grad():
            model_pos.load_state_dict(model_pos_train.state_dict(), strict=False)
            model_pos.eval()

            epoch_loss_3d_valid = 0
            epoch_loss_traj_valid = 0
            epoch_loss_2d_valid = 0
            N = 0
            if not args.no_eval:
                # Evaluate on test set
                for cam, batch, batch_2d, batch_grf in test_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch.astype("float32"))
                    inputs_2d = torch.from_numpy(batch_2d.astype("float32"))
                    targ_grf = torch.from_numpy(batch_grf.astype("float32"))

                    ##### apply test-time-augmentation (following Videopose3d)
                    inputs_2d_flip = inputs_2d.clone()
                    if in_chans == 2:  # Only flip 2D inputs
                        inputs_2d_flip[:, :, :, 0] *= -1
                        inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[
                            :, :, kps_right + kps_left, :
                        ]

                    ##### convert size
                    inputs_2d, inputs_3d = eval_data_prepare(
                        receptive_field, inputs_2d, inputs_3d
                    )
                    if in_chans == 2:  # Only flip 2D inputs
                        inputs_2d_flip, _ = eval_data_prepare(
                            receptive_field, inputs_2d_flip, inputs_3d
                        )

                    if torch.cuda.is_available():
                        inputs_2d = inputs_2d.cuda()
                        inputs_2d_flip = inputs_2d_flip.cuda()
                        targ_grf = targ_grf.cuda()
                        inputs_3d = inputs_3d.cuda()
                    # inputs_3d[:, :, 0] = 0

                    if (
                        args.multitask and in_chans == 2
                    ):  # Only multitask when input is 2D
                        predicted_3d_pos, predicted_grf = model_pos_train(inputs_2d)
                        predicted_3d_pos_flip, predicted_grf_flip = model_pos(
                            inputs_2d_flip
                        )

                        predicted_3d_pos_flip[:, :, :, 0] *= -1
                        predicted_3d_pos_flip[:, :, joints_left + joints_right] = (
                            predicted_3d_pos_flip[:, :, joints_right + joints_left]
                        )
                        predicted_3d_pos = torch.mean(
                            torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1),
                            dim=1,
                            keepdim=True,
                        )
                    else:
                        predicted_3d_pos = None
                        predicted_grf = model_pos(inputs_2d)
                        if in_chans == 2:  # Only flip 2D inputs
                            predicted_grf_flip = model_pos(inputs_2d_flip)

                    if in_chans == 2:  # Only flip 2D inputs
                        grf_copy = torch.clone(predicted_grf_flip)
                        predicted_grf_flip[:, :, :3], predicted_grf_flip[:, :, 3:] = (
                            grf_copy[:, :, 3:],
                            grf_copy[:, :, :3],
                        )

                        predicted_grf = torch.mean(
                            torch.cat((predicted_grf, predicted_grf_flip), dim=1),
                            dim=1,
                            keepdim=True,
                        )

                    del inputs_2d, inputs_2d_flip
                    torch.cuda.empty_cache()

                    if args.multitask and in_chans == 2:
                        loss_3d_pos = args.multitask_alpha * mpjpe(
                            predicted_3d_pos, inputs_3d
                        ) + force_mse(predicted_grf, targ_grf, thresholds)
                    else:
                        loss_3d_pos = force_mse(predicted_grf, targ_grf, thresholds)
                    epoch_loss_3d_valid += (
                        inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    )
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    del (
                        inputs_3d,
                        loss_3d_pos,
                        predicted_3d_pos,
                        targ_grf,
                        predicted_grf,
                    )
                    torch.cuda.empty_cache()

                losses_3d_valid.append(epoch_loss_3d_valid / N)

                # Evaluate on training set, this time in evaluation mode
                epoch_loss_3d_train_eval = 0
                epoch_loss_traj_train_eval = 0
                epoch_loss_2d_train_labeled_eval = 0
                N = 0
                for (
                    cam,
                    batch,
                    batch_2d,
                    batch_grf,
                ) in train_generator_eval.next_epoch():
                    if batch_2d.shape[1] == 0:
                        # This can only happen when downsampling the dataset
                        continue

                    inputs_3d = torch.from_numpy(batch.astype("float32"))
                    inputs_2d = torch.from_numpy(batch_2d.astype("float32"))
                    targ_grf = torch.from_numpy(batch_grf.astype("float32"))
                    inputs_2d, inputs_3d = eval_data_prepare(
                        receptive_field, inputs_2d, inputs_3d
                    )

                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                        targ_grf = targ_grf.cuda()

                    # Compute 3D poses
                    if args.multitask and in_chans == 2:
                        predicted_3d_pos, predicted_grf = model_pos_train(inputs_2d)
                        loss_3d_pos = args.multitask_alpha * mpjpe(
                            predicted_3d_pos, inputs_3d
                        ) + force_mse(predicted_grf, targ_grf, thresholds)
                    else:
                        predicted_grf = model_pos(inputs_2d)
                        loss_3d_pos = force_mse(predicted_grf, targ_grf, thresholds)

                    del inputs_2d
                    torch.cuda.empty_cache()

                    epoch_loss_3d_train_eval += (
                        inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    )
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    del inputs_3d, loss_3d_pos, targ_grf, predicted_grf
                    torch.cuda.empty_cache()

                losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)

                # Evaluate 2D loss on unlabeled training set (in evaluation mode)
                epoch_loss_2d_train_unlabeled_eval = 0
                N_semi = 0

        elapsed = (time() - start_time) / 60

        if args.no_eval:
            print(
                "[%d] time %.2f lr %f 3d_train %f"
                % (epoch + 1, elapsed, lr, losses_3d_train[-1] * 1000)
            )
        else:
            print(
                "[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f"
                % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000,
                    losses_3d_train_eval[-1] * 1000,
                    losses_3d_valid[-1] * 1000,
                )
            )

        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group["lr"] *= lr_decay
        epoch += 1

        # Decay BatchNorm momentum
        # momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
        # model_pos_train.set_bn_momentum(momentum)

        # Save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.bin")
            print("Saving checkpoint to", chk_path)

            torch.save(
                {
                    "epoch": epoch,
                    "lr": lr,
                    "random_state": train_generator.random_state(),
                    "optimizer": optimizer.state_dict(),
                    "model_pos": model_pos_train.state_dict(),
                    # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                    # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
                },
                chk_path,
            )

        #### save best checkpoint
        best_chk_path = os.path.join(checkpoint_dir, "best_epoch.bin")
        if losses_3d_valid[-1] * 1000 < min_loss:
            min_loss = losses_3d_valid[-1] * 1000
            print("save best checkpoint")
            torch.save(
                {
                    "epoch": epoch,
                    "lr": lr,
                    "random_state": train_generator.random_state(),
                    "optimizer": optimizer.state_dict(),
                    "model_pos": model_pos_train.state_dict(),
                    # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                    # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
                },
                best_chk_path,
            )

        # Save training curves after every epoch, as .png images (if requested)
        if args.export_training_curves and epoch > 3:
            if "matplotlib" not in sys.modules:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], "--", color="C0")
            plt.plot(epoch_x, losses_3d_train_eval[3:], color="C0")
            plt.plot(epoch_x, losses_3d_valid[3:], color="C1")
            plt.legend(["3d train", "3d train (eval)", "3d valid (eval)"])
            plt.ylabel("MPJPE (m)")
            plt.xlabel("Epoch")
            plt.xlim((3, epoch))
            plt.savefig(os.path.join(checkpoint_dir, "loss_3d.png"))

            plt.close("all")


# Evaluate
def evaluate(
    test_generator, action=None, return_predictions=False, use_trajectory_model=False
):
    epoch_loss = []
    cam_losses = {}
    cam_group_losses = {}
    cam_stats_pred = {}
    cam_stats_gt = {}
    with torch.no_grad():
        if not use_trajectory_model:
            model_pos.eval()
        N = 0
        for cam, batch, batch_2d, batch_grf in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype("float32"))
            inputs_3d = torch.from_numpy(batch.astype("float32"))
            targ_grf = torch.from_numpy(batch_grf.astype("float32"))

            ##### apply test-time-augmentation (following Videopose3d)
            inputs_2d_flip = inputs_2d.clone()
            if in_chans == 2:  # Only flip 2D inputs
                inputs_2d_flip[:, :, :, 0] *= -1
                inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[
                    :, :, kps_right + kps_left, :
                ]

            ##### convert size
            inputs_2d, inputs_3d = eval_data_prepare(
                receptive_field, inputs_2d, inputs_3d
            )
            if in_chans == 2:
                inputs_2d_flip, _ = eval_data_prepare(
                    receptive_field, inputs_2d_flip, inputs_3d
                )

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_2d_flip = inputs_2d_flip.cuda()
                targ_grf = targ_grf.cuda()
                inputs_3d = inputs_3d.cuda()

            if args.multitask and in_chans == 2:  # Only multitask when input is 2D
                _, predicted_grf = model_pos_train(inputs_2d)
                _, predicted_grf_flip = model_pos(inputs_2d_flip)
            else:
                predicted_grf = model_pos(inputs_2d)
                if in_chans == 2:
                    predicted_grf_flip = model_pos(inputs_2d_flip)

            if in_chans == 2:
                grf_copy = torch.clone(predicted_grf_flip)
                predicted_grf_flip[:, :, :3], predicted_grf_flip[:, :, 3:] = (
                    grf_copy[:, :, 3:],
                    grf_copy[:, :, :3],
                )

                predicted_grf = torch.mean(
                    torch.cat((predicted_grf, predicted_grf_flip), dim=1),
                    dim=1,
                    keepdim=True,
                )

            del inputs_2d, inputs_2d_flip
            torch.cuda.empty_cache()

            if return_predictions:
                return predicted_grf.squeeze(0).cpu().numpy()

            if False:
                title = action + " - " + cam[0]["id"]
                plt.figure(figsize=(18, 12))
                plt.suptitle(title)
                y_min = -400
                y_max = 1600
                x_max = 400

                mass = 88.37
                for idx, title in enumerate(titles):
                    pr = predicted_grf[:, :, idx] * mass
                    gt = targ_grf[:, :, idx] * mass
                    err = rmse(pr, gt).item()

                    ax = plt.subplot(2, int(len(titles) / 2), idx + 1)

                    ax.plot(pr.cpu().numpy(), "r", label="NN Prediction")  # Predicted
                    ax.plot(gt.cpu().numpy(), "k--", label="Force Plate")  # Groundtruth
                    ax.set_title(f"{title}: rmse: {err:.3f}")
                    ax.set_ylim([y_min, y_max])

                    if idx == 0 or idx == 3:
                        ax.set_ylabel("N/kg")
                    ax.set_xlabel("seconds")
                plt.savefig(f"out_files/{title}.png")

            error = rmse(predicted_grf, targ_grf)

            cam_losses[cam[0]["id"]] = error.item()
            epoch_loss.append(error.item())

            range_forces = torch.max(targ_grf, dim=0)[0] - torch.min(targ_grf, dim=0)[0]
            cam_group_losses[cam[0]["id"]] = {}
            mass = 88.37  # average mass of training set
            stat_pred = []
            stat_gt = []
            for group, g_titles in title_groups.items():
                cam_group_losses[cam[0]["id"]][group] = {}
                prs = []
                gts = []
                errs_norm = []

                for title in g_titles:
                    idx = titles.index(title)
                    prs.append(predicted_grf[:, :, idx] * mass)
                    gts.append(targ_grf[:, :, idx] * mass)

                    err = rmse(predicted_grf[:, :, idx], targ_grf[:, :, idx])
                    if range_forces[:, idx].item() == 0:
                        # GT is completely zero, so no range. Default to avg_mass/100, since already normalized by mass
                        errs_norm.append((err / 0.8837).item())
                    else:
                        errs_norm.append((err / range_forces[:, idx]).item())

                prs = torch.stack(prs)
                gts = torch.stack(gts)
                cam_group_losses[cam[0]["id"]][group]["rmse"] = rmse(prs, gts)
                cam_group_losses[cam[0]["id"]][group]["nrmse"] = np.mean(errs_norm)

                # stats on characteristics of curve
                sum_prs = torch.sum(prs.squeeze(), dim=0)
                sum_gts = torch.sum(gts.squeeze(), dim=0)

                # Find extrema points
                k = 5  # top k peaks and valleys
                peak_idxs = sig.argrelextrema(sum_prs.cpu().numpy(), np.greater)[0]
                vall_idxs = sig.argrelextrema(sum_prs.cpu().numpy(), np.less)[0]
                peak_idxs = np.pad(
                    peak_idxs, (0, np.clip(k - len(peak_idxs), 0, None)), mode="edge"
                )
                vall_idxs = np.pad(
                    vall_idxs, (0, np.clip(k - len(vall_idxs), 0, None)), mode="edge"
                )
                _idxs_pr = np.concatenate([peak_idxs, vall_idxs])
                _pr, new_idx = torch.sort(sum_prs[_idxs_pr].cpu(), descending=True)
                _idxs_pr = _idxs_pr[new_idx]

                peak_idxs = sig.argrelextrema(sum_gts.cpu().numpy(), np.greater)[0]
                vall_idxs = sig.argrelextrema(
                    sum_gts.cpu().numpy() + 0.1 * np.random.rand(len(targ_grf)), np.less
                )[0]  # plus some jitter to capture flat points
                peak_idxs = np.pad(
                    peak_idxs, (0, np.clip(k - len(peak_idxs), 0, None)), mode="edge"
                )
                vall_idxs = np.pad(
                    vall_idxs, (0, np.clip(k - len(vall_idxs), 0, None)), mode="edge"
                )
                _idxs_gt = np.concatenate([peak_idxs, vall_idxs])
                _gt, new_idx = torch.sort(sum_gts[_idxs_gt].cpu(), descending=True)
                _idxs_gt = _idxs_gt[new_idx]

                # statistic will be top-k peaks and valleys and their indices
                stat_pred.append(
                    np.stack([_pr[:k], _pr[-k:], _idxs_pr[:k], _idxs_pr[-k:]], axis=1)
                )
                stat_gt.append(
                    np.stack([_gt[:k], _gt[-k:], _idxs_gt[:k], _idxs_gt[-k:]], axis=1)
                )

            cam_stats_pred[cam[0]["id"]] = np.stack(stat_pred)
            cam_stats_gt[cam[0]["id"]] = np.stack(stat_gt)

    if action is None:
        print("----------")
    else:
        print("----" + action + "----")
    print("average loss across cameras:", np.mean(epoch_loss))

    return {
        "cam_losses": cam_losses,
        "cam_group_losses": cam_group_losses,
        "cam_stats_pred": cam_stats_pred,
        "cam_stats_gt": cam_stats_gt,
    }


if args.render:
    print("Rendering...")

    input_keypoints = keypoints[args.viz_subject][args.viz_action][
        args.viz_camera
    ].copy()
    # In this repo, `run_poseformer_force.py` evaluates GRF (6-axis) checkpoints.
    # The previous pose-animation render path was inconsistent with GRF-only checkpoints.
    # For render mode, we generate a GRF curve visualization (pred vs GT) for a chosen sequence.

    import os

    os.makedirs("out_files", exist_ok=True)

    # Fetch GT 3D positions (for windowing) and GT forces for this sequence.
    # ForcePose stores one GRF signal per trial (not per camera), and provides triangulated 3D joints.
    positions_3d = None
    forces_gt = None
    if (
        args.viz_subject in dataset.subjects()
        and args.viz_action in dataset[args.viz_subject]
    ):
        if "positions_triangulated" in dataset[args.viz_subject][args.viz_action]:
            positions_3d = dataset[args.viz_subject][args.viz_action][
                "positions_triangulated"
            ].copy()
        if "forces" in dataset[args.viz_subject][args.viz_action]:
            forces_gt = dataset[args.viz_subject][args.viz_action]["forces"].copy()

    if forces_gt is None:
        raise RuntimeError(
            "No ground-truth forces found for the requested subject/action; cannot render GRF curves."
        )

    gen = UnchunkedGenerator(
        None,
        [positions_3d] if positions_3d is not None else [None],
        [input_keypoints],
        pad=pad,
        causal_shift=causal_shift,
        augment=False,  # flip augmentation handled inside evaluate(); keep generator simple
        kps_left=kps_left,
        kps_right=kps_right,
        joints_left=joints_left,
        joints_right=joints_right,
        forces=[forces_gt],
    )

    pred_grf = evaluate(gen, return_predictions=True)
    # `evaluate(..., return_predictions=True)` returns shape (T_out, 1, 6) for GRF.
    if pred_grf.ndim == 3 and pred_grf.shape[1] == 1:
        pred_grf = pred_grf[:, 0, :]

    # Align GT length to model output length.
    # In this codebase, `UnchunkedGenerator` pads the 2D inputs by `pad` frames on both sides.
    # As a result, `evaluate(..., return_predictions=True)` typically returns one prediction per
    # original frame (length == forces_gt length). Fall back to center-frame alignment only if needed.
    rf = receptive_field
    t_out = pred_grf.shape[0]
    if forces_gt is not None and t_out == forces_gt.shape[0]:
        start = 0
        forces_gt_aligned = forces_gt
    else:
        start = (rf - 1) // 2
        forces_gt_aligned = forces_gt[start : start + t_out]

    # Save raw arrays for downstream analysis
    export_path = "out_files/grf_pred_gt.npz"
    np.savez(
        export_path,
        pred_grf=pred_grf,
        gt_grf=forces_gt_aligned,
        gt_grf_full=forces_gt,
        gt_start_index=start,
        receptive_field=rf,
    )
    print("Exported GRF arrays to", export_path)

    # Plot predicted vs GT GRF curves
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, writers

    titles = ["Fx1", "Fy1", "Fz1", "Fx2", "Fy2", "Fz2"]
    # If user requested a video, render 2D skeleton + GRF overlay animation.
    if args.viz_output and (
        args.viz_output.endswith(".mp4") or args.viz_output.endswith(".gif")
    ):
        cam = dataset.cameras()[args.viz_subject][args.viz_camera]
        w, h = cam["res_w"], cam["res_h"]

        # The prediction corresponds to center frames in the receptive field window.
        keypoints_center = input_keypoints[start : start + t_out, :, :2]
        keypoints_center = image_coordinates(keypoints_center, w=w, h=h)

        # Aggregate forces per axis across both plates for a simpler overlay.
        gt_total = forces_gt_aligned[:, 0:3] + forces_gt_aligned[:, 3:6]
        pr_total = pred_grf[:, 0:3] + pred_grf[:, 3:6]

        # COCO-17 skeleton edges (index pairs). Works with `pt_coco` layout.
        edges = [
            (5, 7),
            (7, 9),  # left arm
            (6, 8),
            (8, 10),  # right arm
            (11, 13),
            (13, 15),  # left leg
            (12, 14),
            (14, 16),  # right leg
            (5, 6),
            (11, 12),  # shoulders, hips
            (5, 11),
            (6, 12),  # torso sides
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),  # head
        ]

        fig = plt.figure(figsize=(14, 7))
        ax_pose = fig.add_subplot(1, 2, 1)
        ax_force = fig.add_subplot(1, 2, 2)
        fig.suptitle(
            f"{args.viz_subject} / {args.viz_action} / cam {args.viz_camera} — 2D skeleton + GRF (GT vs Pred)"
        )

        # Pose axis setup
        ax_pose.set_xlim(0, w)
        ax_pose.set_ylim(h, 0)
        ax_pose.set_aspect("equal")
        ax_pose.set_title("2D pose (input keypoints)")
        ax_pose.axis("off")

        # Force axis setup (total Fx/Fy/Fz)
        t = np.arange(t_out)
        force_titles = ["Fx total", "Fy total", "Fz total"]
        colors = ["C0", "C1", "C2"]
        for i in range(3):
            ax_force.plot(
                t,
                gt_total[:t_out, i],
                color=colors[i],
                linestyle="--",
                linewidth=1,
                label=f"GT {force_titles[i]}",
            )
            ax_force.plot(
                t,
                pr_total[:t_out, i],
                color=colors[i],
                linestyle="-",
                linewidth=1,
                label=f"Pred {force_titles[i]}",
            )
        vline = ax_force.axvline(0, color="k", alpha=0.4, linewidth=1)
        ax_force.set_title("GRF totals (both plates)")
        ax_force.set_xlabel("center-frame index")
        ax_force.set_ylabel("N/kg")
        ax_force.grid(True, alpha=0.25)
        ax_force.legend(fontsize=8, ncol=1, loc="upper right")

        # Artists for pose
        scat = ax_pose.scatter(
            [], [], s=12, c="lime", edgecolors="black", linewidths=0.4, zorder=3
        )
        lines = [
            ax_pose.plot([], [], color="lime", linewidth=2, zorder=2)[0] for _ in edges
        ]
        txt = ax_pose.text(
            10,
            20,
            "",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.6),
            fontsize=9,
        )

        def update(i):
            pts = keypoints_center[i]
            scat.set_offsets(pts)
            for li, (a, b) in zip(lines, edges):
                li.set_data([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]])
            vline.set_xdata([i, i])
            txt.set_text(
                f"Frame: {i + start} (center)\n"
                f"GT total  Fx/Fy/Fz: {gt_total[i, 0]:.2f}, {gt_total[i, 1]:.2f}, {gt_total[i, 2]:.2f}\n"
                f"Pred total Fx/Fy/Fz: {pr_total[i, 0]:.2f}, {pr_total[i, 1]:.2f}, {pr_total[i, 2]:.2f}"
            )
            return [scat, *lines, vline, txt]

        anim = FuncAnimation(
            fig, update, frames=t_out, interval=1000 / dataset.fps(), blit=True
        )

        if args.viz_output.endswith(".mp4"):
            Writer = writers["ffmpeg"]
            writer = Writer(fps=dataset.fps(), metadata={}, bitrate=args.viz_bitrate)
            anim.save(args.viz_output, writer=writer)
        else:
            anim.save(args.viz_output, dpi=80, writer="imagemagick")

        plt.close(fig)
        print("Saved 2D skeleton + GRF video to", args.viz_output)
    else:
        # Default to a static GRF plot image
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(
            f"{args.viz_subject} / {args.viz_action} / cam {args.viz_camera} — GRF (pred vs GT)"
        )
        for i, tname in enumerate(titles):
            ax = fig.add_subplot(2, 3, i + 1)
            ax.plot(forces_gt_aligned[:, i], "k--", linewidth=1, label="GT")
            ax.plot(pred_grf[:, i], "r", linewidth=1, label="Pred")
            ax.set_title(tname)
            ax.set_xlabel("center-frame index")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()

        out_png = (
            args.viz_output
            if (args.viz_output and args.viz_output.endswith(".png"))
            else "out_files/grf_plot.png"
        )
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        print("Saved GRF plot to", out_png)

else:
    print("Evaluating...")
    all_actions = {}
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}

        for action in dataset[subject].keys():
            action_name = action.split(" ")[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))

    def fetch_actions(actions):
        out_camera_params = []
        out_poses_3d = []
        out_poses_2d = []
        out_forces = []
        out_seq_names = []

        for subject, action in actions:
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i][:, :, :2])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), "Camera count mismatch"
                for cam in cams:
                    out_camera_params.append(cam)

            poses_3d = dataset[subject][action]["positions_3d"]
            assert len(poses_3d) == len(poses_2d), "Camera count mismatch"
            for i in range(len(poses_3d)):  # Iterate across cameras
                out_poses_3d.append(poses_3d[i])

            if "seq_name" in dataset[subject][action].keys():
                out_seq_names.append(dataset[subject][action]["seq_name"])

            if "forces" in dataset[subject][action]:
                forces = dataset[subject][action]["forces"]
                for i in range(
                    len(poses_2d)
                ):  # Iterate across cameras (repeat same forces for each view)
                    assert forces.shape[0] == len(poses_2d[i]), "Sequence mismatch"
                    out_forces.append(forces)

        stride = args.downsample
        if stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_forces) == 0:
            print("No forces detected from dataset")
            out_forces = None

        return out_camera_params, out_poses_3d, out_poses_2d, out_forces, out_seq_names

    def run_evaluation(actions, action_filter=None):
        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        errors_vel = []

        cam_losses = {}
        cam_groups_losses = {}
        seq_force_stats = {1: [], 3: [], 5: []}
        for action_key in actions.keys():
            print(f"Subject: {actions[action_key][0][0]}")
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            cameras_act, poses_act, poses_2d_act, forces_act, seq_names = fetch_actions(
                actions[action_key]
            )
            if in_chans == 3:  # Using 3D keypoints as input
                poses_2d_act = poses_act

            if (
                args.dataset.startswith("parkour") and args.evaluate
            ):  # return predictions
                for cam_act, pose_act, pose_2d_act, force_act, seq_name in zip(
                    cameras_act, poses_act, poses_2d_act, forces_act, seq_names
                ):
                    gen = UnchunkedGenerator(
                        [cam_act],
                        [pose_act],
                        [pose_2d_act],
                        pad=pad,
                        causal_shift=causal_shift,
                        augment=args.test_time_augmentation,
                        kps_left=kps_left,
                        kps_right=kps_right,
                        joints_left=joints_left,
                        joints_right=joints_right,
                        forces=[force_act],
                    )

                    pred_grf = evaluate(
                        gen, action_key, return_predictions=True
                    ).squeeze()

                    # Parkour data to save and eval externally
                    force_preds = (
                        pred_grf.reshape(pred_grf.shape[0], 2, 3) * 74.6
                    )  # multiply by est. mass
                    force_preds = np.concatenate(
                        (force_preds, np.zeros_like(force_preds)), axis=-1
                    )  # Add dummy moment forces
                    force_preds = np.concatenate(
                        (force_preds, np.zeros_like(force_preds)), axis=-2
                    )  # Add left-right hand forces

                    out_dir = os.path.join(
                        "data//Parkour-dataset/predictions", args.exp_name
                    )
                    os.makedirs(out_dir, exist_ok=True)
                    np.save(os.path.join(out_dir, seq_name), force_preds)

                    losses = {cam_act["id"]: 1.0}
                    # title_groups = {}

            else:
                gen = UnchunkedGenerator(
                    cameras_act,
                    poses_act,
                    poses_2d_act,
                    pad=pad,
                    causal_shift=causal_shift,
                    augment=args.test_time_augmentation,
                    kps_left=kps_left,
                    kps_right=kps_right,
                    joints_left=joints_left,
                    joints_right=joints_right,
                    forces=forces_act,
                )

                eval_losses = evaluate(gen, action_key)
                losses = eval_losses["cam_losses"]
                losses_groups = eval_losses["cam_group_losses"]
                cam_stats_pred = eval_losses["cam_stats_pred"]
                cam_stats_gt = eval_losses["cam_stats_gt"]

            seq = actions[action_key][0][0] + "_" + action_key
            for cam, v in losses.items():
                if cam not in cam_losses.keys():
                    cam_losses[cam] = []
                    cam_groups_losses[cam] = {}

                cam_losses[cam].append(v)

                cam_groups_losses[cam][seq] = {}
                for group in title_groups.keys():
                    cam_groups_losses[cam][seq][group] = losses_groups[cam][group]

            # Force stats across cameras
            cam_stats_pred = np.stack(list(cam_stats_pred.values()))
            cam_stats_gt = np.stack(list(cam_stats_gt.values()))
            diffs = abs(
                cam_stats_pred - cam_stats_gt
            )  # shape: num_cams x axes x k x stat_dim

            # average out all k peaks and valleys
            # Multi-threshold. k=1, k=3, k=5
            diffs_1 = diffs[:, :, 0]
            diffs_3 = np.mean(diffs[:, :, :3], axis=2)
            diffs_5 = np.mean(diffs[:, :, :5], axis=2)

            # avg peaks, valleys, and idxs across sequences
            vals_1 = np.mean(diffs_1, axis=(0, 1))
            vals_3 = np.mean(diffs_3, axis=(0, 1))
            vals_5 = np.mean(diffs_5, axis=(0, 1))

            seq_force_stats[1].append(vals_1)
            seq_force_stats[3].append(vals_3)
            seq_force_stats[5].append(vals_5)

        mass = 1.0  # average mass of training set - already scaled earlier
        print("\n")
        avg_cam_errs = []
        cam_errs_groups = {}
        cam_columns = ""
        cam_out_line = ""
        for cam, loss in cam_losses.items():
            avg_seq_err = np.mean(loss)
            print(f"{cam} avg camera loss: {avg_seq_err * mass:.3f}")
            cam_columns += cam + ","
            cam_out_line += ",".join((f"{avg_seq_err.item() * mass:.3f}", ""))
            avg_cam_errs.append(avg_seq_err)

            for group in title_groups.keys():
                avg_seq_err = torch.mean(
                    torch.tensor(
                        [
                            cam_groups_losses[cam][seq][group]["rmse"]
                            for seq in cam_groups_losses[cam].keys()
                        ]
                    )
                )
                avg_seq_nerr = torch.mean(
                    torch.tensor(
                        [
                            cam_groups_losses[cam][seq][group]["nrmse"]
                            for seq in cam_groups_losses[cam].keys()
                        ]
                    )
                )

                print(f"{cam} {group} loss: {avg_seq_err * mass:.3f}")
                if group not in cam_errs_groups:
                    cam_errs_groups[group] = {"rmse": [], "nrmse": []}

                cam_errs_groups[group]["rmse"].append(avg_seq_err)
                cam_errs_groups[group]["nrmse"].append(avg_seq_nerr)

                cam_columns += ",".join((group, ""))
                cam_out_line += ",".join((f"{avg_seq_nerr.item() * 100:.3f}%", ""))

            print("--" * 30)

        avg_err = np.mean(avg_cam_errs)
        print(f"Avg sequence loss: {avg_err * mass}")

        avg_seq_force_stats = {}
        for k_peak in seq_force_stats.keys():
            avg_seq_force_stats[k_peak] = np.mean(
                np.stack(seq_force_stats[k_peak]), axis=0
            )
            peak_dist = np.sqrt(
                avg_seq_force_stats[k_peak][0] ** 2
                + avg_seq_force_stats[k_peak][2] ** 2
            )
            vall_dist = np.sqrt(
                avg_seq_force_stats[k_peak][1] ** 2
                + avg_seq_force_stats[k_peak][3] ** 2
            )
            avg_seq_force_stats[k_peak] = np.concatenate(
                (avg_seq_force_stats[k_peak], np.array([peak_dist, vall_dist]))
            )
            print(
                f"Avg distance k: {k_peak}, peak dist: {avg_seq_force_stats[k_peak][4]:.3f}, valley dist:{avg_seq_force_stats[k_peak][5]:.3f},"
            )

        columns = "exp_name,"
        out_line = ""
        for group in title_groups.keys():
            avg_grp_err = torch.mean(torch.tensor(cam_errs_groups[group]["rmse"]))
            avg_grp_nerr = (
                torch.mean(torch.tensor(cam_errs_groups[group]["nrmse"])) * 100
            )  # in terms of percentage
            print(
                f"Avg sequence {group} RMSE: {avg_grp_err * mass:.3f}, nRMSE: {avg_grp_nerr:.3f}%"
            )

            columns += ",".join((group, "%", ""))
            out_line += ",".join(
                (
                    f"{avg_grp_err.item() * mass:.3f}",
                    f"{avg_grp_nerr.item():.3f}%",
                    "",
                )
            )

        filename = "errors_along_axes.txt"
        columns += cam_columns
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                f.write(columns + "\n")

        with open(filename, "a") as f:
            f.write(str(args.exp_name) + "," + out_line + cam_out_line + "\n")

        columns = "exp_name,"
        out_line = ""
        for k_peak, avg_stats in avg_seq_force_stats.items():
            columns += "k,peak,valley,peak_idx,valley_idx,peak_dist,valley_dist,"
            out_line += f"{k_peak}, {avg_stats[0]:.3f}, {avg_stats[1]:.3f}, {avg_stats[2]:.3f}, {avg_stats[3]:.3f}, {avg_stats[4]:.3f}, {avg_stats[5]:.3f},"

        filename = "peak_distance_avg.txt"
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                f.write(columns + "\n")

        with open(filename, "a") as f:
            f.write(str(args.exp_name) + "," + out_line + "\n")

    if not args.by_subject:
        run_evaluation(all_actions, action_filter)
    else:
        for subject in all_actions_by_subject.keys():
            print("Evaluating on subject", subject)
            run_evaluation(all_actions_by_subject[subject], action_filter)
            print("")
