# Merge two videos side by side using ffmpeg
import subprocess

def merge_videos(video1_path, video2_path, output_path):
    # Command to merge videos side by side
    filter_string = (
        "[0:v]drawtext=text='ForcePlate':fontcolor=red:fontsize=24:x=10:y=10[v1]; "
        "[1:v]drawtext=text='ForcePose':fontcolor=red:fontsize=24:x=10:y=10[v2]; "
        "[v1][v2]hstack=inputs=2"
    )
    
    command = [
        'ffmpeg',
        '-i', video1_path,
        '-i', video2_path,
        '-filter_complex', filter_string,
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'veryslow',
        output_path
    ]

    # Run the command
    subprocess.run(command, check=True)

if __name__ == "__main__":
    video1 = 'out_files/untitled folder/gen_pose_forceplate.mp4'  # Path to the first video
    video2 = 'out_files/untitled folder/gen_pose_pred.mp4'  # Path to the second video
    output_video = 'out_files/untitled folder/merged_geny.mp4'  # Path for the output merged video

    merge_videos(video1, video2, output_video)
    print(f'Merged video saved as {output_video}')