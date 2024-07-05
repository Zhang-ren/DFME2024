import os
import subprocess
print('start')
def run_temporal_on_videos(base_dir, output_base_dir, exp_name="o3f_hmhm2_bg_qnoise_mix4_nl_n_t_ds3", amplification_factor=10, fl=0.04, fh=0.4, fs=30, n_tap=2, filter_type="differenceOfIIR"):
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            video_dir = os.path.join(root, dir_name)
            # for video in os.listdir(video_dirs):
            #     video_dir = os.path.join(video_dirs, video)
            if os.path.isdir(video_dir): 
                video_output_dir = os.path.join(output_base_dir, dir_name) # , video
                print(video_output_dir)
                print(video_dir)
                # if not os.path.exists(video_output_dir):
                #     os.makedirs(video_output_dir)

                cmd = [
                    'bash', 'run_temporal_on_test_videos.sh', exp_name,
                    video_dir, str(amplification_factor), str(fl),
                    str(fh), str(fs), str(n_tap), filter_type, video_output_dir
                ]
                print(f"Running command: {' '.join(cmd)}")
                subprocess.run(cmd)

if __name__ == "__main__":
# base_dir = "/home/data2/MEGC-2019/DATASET/test_data_A"
# output_base_dir = "/home/data2/MEGC//DFME2024_test_5"  # 修改为你的输出目录
    base_dir = "/home/data2/MEGC-2019/DATASET/test_data_B/"
    output_base_dir = "/home/data2/MEGC/DFME_B_10"
    run_temporal_on_videos(base_dir, output_base_dir)
