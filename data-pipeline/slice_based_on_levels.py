import os
import pandas as pd
import numpy as np
import glob

base_dir = "/datasets/youtube_aicenter_dataset/"
input_dir = "/datasets/youtube_aicenter_dataset/%d/"
scores_dir = "../scores/vol0%d/group/levels"
out_dir = "../engagement-slices/%d"
out_file = "../engagement-slices/%d/%s/%s.mp4"
video_name = "{}.mp4"
csv_name = "list.csv"

engagement_levels = ["-2", "-1", "0", "1", "2"]


def sec_convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
      
    return "%02d:%02d:%02d" % (hour, minutes, seconds)


if __name__ == "__main__":
    total_cnt = 0
    for i in range(3, 9):
        video_list = []
        for vi in range(1, len(glob.glob1(input_dir % i,"*.mp4")) + 1):
            vid_file = os.path.join(input_dir % i, "%d.mp4" % vi)
            ci = vi - 1
            csv_file = os.path.join(scores_dir % i, "v%d.csv" % ci)
            print(vid_file, csv_file)
            df = pd.read_csv(csv_file, sep=',')
            df.head()

            start_time = 0
            slices = []

            for curr_level in df['group']:
                #print(start_time, curr_level)

                start_time = sec_convert(i)
                end_time = sec_convert(i+10)

                get_ipython().system('ffmpeg -ss {start_time} -to {end_time} -i {vid_file} -c copy {out_file % (i, str(curr_level), str(total_cnt))} -hide_banner -loglevel error')
                video_list.append(["%d.mp4" % vi, curr_level])
                total_cnt += 1

        df = pd.DataFrame(video_list, columns = ['video_name', 'tag'])
        df['tag'] = df['tag'].map({-2: 'HD', -1:'D', 0:'M', 1: 'E', 2: 'HE'})
        df.to_csv(os.path.join(out_dir % i, csv_name), index=False)

