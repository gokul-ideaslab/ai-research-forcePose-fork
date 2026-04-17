from common.generators import UnchunkedGenerator
from common.model_poseformer import PoseTransformer
from common.camera import normalize_screen_coordinates

import os
import numpy as np
import torch
import torch.nn as nn

in_chans = 2
receptive_field = 81
pad = (receptive_field - 1) // 2

kps_left = [1, 3, 5, 7, 9, 11, 13, 15]
kps_right = [2, 4, 6, 8, 10, 12, 14, 16]

W, H = 1920, 1080

checkpoint_name = "2d_81frames_t2.bin"
checkpoint_filename = os.path.join("checkpoint/coco_2d/", checkpoint_name)
checkpoint = torch.load(checkpoint_filename, map_location=lambda storage, loc: storage, weights_only=False)

model_pos = PoseTransformer(
    num_frame=81,
    num_joints=17,
    in_chans=in_chans,
    embed_dim_ratio=32,
    depth=4,
    num_heads=8,
    mlp_ratio=2.0,
    qkv_bias=True,
    qk_scale=None,
    drop_path_rate=0,
    pred_force=True,
)

if torch.cuda.is_available():
    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()

model_pos.load_state_dict(checkpoint["model_pos"], strict=False)


def normalize_coordinates(X):
    assert X.shape[-1] == 2

    offset = torch.tensor([1, H/W], device=X.device, dtype=X.dtype)

    return X/W*2 - offset

def eval_data_prepare(receptive_field, inputs_2d):
    inputs_2d_p = torch.squeeze(inputs_2d)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    for i in range(out_num):
        eval_input_2d[i, :, :, :] = inputs_2d_p[i : i + receptive_field, :, :]
    return eval_input_2d

def evaluate(test_generator):
    with torch.no_grad():
        model_pos.eval()
        N = 0
        for _, _, batch_2d, _ in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype("float32"))

            print("Input: ", inputs_2d.shape)

            ##### apply test-time-augmentation (following Videopose3d)
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip[:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

            inputs_2d = eval_data_prepare(receptive_field, inputs_2d)
            inputs_2d_flip = eval_data_prepare(receptive_field, inputs_2d_flip)

            inputs_2d = normalize_coordinates(inputs_2d)
            inputs_2d_flip = normalize_coordinates(inputs_2d_flip)

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_2d_flip = inputs_2d_flip.cuda()

            predicted_grf = model_pos(inputs_2d)
            predicted_grf_flip = model_pos(inputs_2d_flip)

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

            return predicted_grf.squeeze(0).cpu().numpy()


if __name__ == '__main__':
    kps_2d = np.load("yolo_kps/keypoints_general_pose_1002.npy")

    gen = UnchunkedGenerator(
        None,
        [None],
        [kps_2d],
        pad=pad,
        kps_left=kps_left,
        kps_right=kps_right,
        joints_left=kps_left,
        joints_right=kps_right
    )

    pred_force = evaluate(gen)
    np.save("yolo_kps/pred_force_gen.npy", pred_force)

    print(pred_force.shape)
