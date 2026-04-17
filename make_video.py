import os
import subprocess
import tempfile

import numpy as np
import matplotlib.pyplot as plt

def save_force_video(keypoints, forces, original_forces=None, out_path='out_files/force_video.mp4', fps=30, scale=5, figsize=(12,6), dpi=100):
    keypoints = np.asarray(keypoints)
    forces = np.asarray(forces)
    if original_forces is not None:
        original_forces = np.asarray(original_forces)
        assert original_forces.shape[1:] == forces.shape[1:], "original_forces must match forces shape except for frame length"

    frame_count = min(
        keypoints.shape[0],
        forces.shape[0],
        original_forces.shape[0] if original_forces is not None else forces.shape[0],
    )
    if frame_count < keypoints.shape[0] or frame_count < forces.shape[0] or (original_forces is not None and frame_count < original_forces.shape[0]):
        print(f'Warning: truncating to {frame_count} frames to match all arrays')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x_min = np.nanmin(keypoints[:frame_count,:,0]) - 20
    x_max = np.nanmax(keypoints[:frame_count,:,0]) + 20
    y_min = np.nanmin(keypoints[:frame_count,:,1]) - 20
    y_max = np.nanmax(keypoints[:frame_count,:,1]) + 20

    skeleton = [
        (5,7),(7,9),
        (6,8),(8,10),
        (5,6),
        (5,11),(6,12),
        (11,12),
        (11,13),(13,15),
        (12,14),(14,16)
    ]

    def draw_frame(ax, k, f, title):
        ax.scatter(k[:,0], k[:,1], c='red')
        for i, j in skeleton:
            ax.plot([k[i,0], k[j,0]], [k[i,1], k[j,1]], 'blue')

        if not np.any(np.isnan(k[16])):
            x, y = k[16]
            fx, fy, fz = f[0], f[1], f[2]
            ax.arrow(x, y, fx * scale, -fy * scale, color='purple', head_width=5)
            ax.text(0.02, 0.95, f'F16=[{fx:.1f},{fy:.1f},{fz:.1f}]', transform=ax.transAxes, color='purple', fontsize=8, va='top')

        if not np.any(np.isnan(k[15])):
            x, y = k[15]
            fx, fy, fz = f[3], f[4], f[5]
            ax.arrow(x, y, fx * scale, -fy * scale, color='purple', head_width=5)
            ax.text(0.02, 0.88, f'F15=[{fx:.1f},{fy:.1f},{fz:.1f}]', transform=ax.transAxes, color='purple', fontsize=8, va='top')

        ax.set_title(title)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
        ax.set_aspect('equal')
        ax.axis('off')

    with tempfile.TemporaryDirectory() as tmpdir:
        ncols = 2 if original_forces is not None else 1
        for frame_idx in range(frame_count):
            k = keypoints[frame_idx]
            f = forces[frame_idx]

            fig, axes = plt.subplots(1, ncols, figsize=figsize, dpi=dpi)
            axes = np.atleast_1d(axes)

            draw_frame(axes[0], k, f, 'Predicted force')
            if original_forces is not None:
                draw_frame(axes[1], k, original_forces[frame_idx], 'Original force')

            fig.suptitle(f'Frame {frame_idx}', fontsize=12)
            filename = os.path.join(tmpdir, f'frame_{frame_idx:04d}.png')
            fig.savefig(filename, dpi=dpi)
            plt.close(fig)

        cmd = [
            'ffmpeg',
            '-y',
            '-framerate', str(fps),
            '-i', os.path.join(tmpdir, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            out_path,
        ]
        print('Running ffmpeg...')
        subprocess.run(cmd, check=True)

    print(f'Video saved to {out_path}')


if __name__ == "__main__":
    out_path = 'out_files/force_video_compare.mp4'
    
    kps = np.load("yolo_kps/keypoints_golf.npy")

    pred = np.load("yolo_kps/pred_force_golf.npy")
    forces = pred.squeeze(axis=1)

    original = None
    gt_path = "forceplate_arr/fp_041220.npy"
    if os.path.exists(gt_path):
        original = np.load(gt_path)
    else:
        print(f'Original force file not found: {gt_path}. Running with predictions only.')

    print('kps', kps.shape)
    print('pred forces', forces.shape)
    if original is not None:
        print('original forces', original.shape)

    original = original[:, [2,1,0, 5,4,3]]

    output_video = 'out_files/general_pose.mp4'
    save_force_video(kps, forces, original_forces=original, out_path=output_video, fps=30, scale=10)
    print('Saved video to', output_video)