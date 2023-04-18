# %%
import os
import pandas as pd
from glob import glob, glob1

# %%
base_dir = "/datasets/youtube_aicenter_dataset/"
#base_dir = "C:\\Users\\ASABUNCUOGLU13\\Documents\\data"
input_dir = "/datasets/youtube_aicenter_dataset/%d/face-center"
#input_dir = "vol0%d\\face-center"
scores_dir = "../scores/vol0%d/group/levels"
out_dir = "../face-levels"
video_name = "{}.mp4"
csv_name = "list.csv"

engagement_levels = ["-2", "-1", "0", "1", "2"]
engagement_level_names = ["HD", "D", "M", "E", "HE"]

# %%
"""
get_ipython().system('mkdir {out_dir}')
for i in range(4, 9):
    get_ipython().system('mkdir {os.path.join(out_dir, "%d" % i)}')
    for j in range(1, 6):
        get_ipython().system(
            'mkdir {os.path.join(out_dir, "%d" % i, "%d" % j)}')
        for e in engagement_level_names:
            get_ipython().system(
                'mkdir {os.path.join(out_dir, "%d" % i, "%d" % j, e)}')
"""
# %%
anonimityDict = [{}, {}, {
    "B": "berk",
    "C": "cansu",
    "D": "yagmur"},
    {
    "0": "bugra",
    "1": "furkan",
    "2": "ecem",
    "3": "seyda"},
    {
    "1": "yavuz",
    "2": "canberk",
    "3": "deniz",
    "4": "oya",
    "5": "ata"},
    {
    "1": "utku",
    "2": "doga",
    "3": "begum",
    "4": "aslı",
    "5": "yigithan"},
    {
    "1": "irmak",
    "2": "metehan",
    "3": "berkecan",
    "4": "damla",
    "5": "eren"},
    {
    "1": "ece",
    "2": "doga",
    "3": "furkan",
    "4": "umutcan",
    "5": "arda"},
    {
    "1": "beyza",
    "2": "bora",
    "3": "kerem",
    "4": "emre",
    "5": "sema"}]

# %%
for i in range(7, 9):
    person_list = glob(os.path.join(base_dir, input_dir % i, "*"))
    for p in range(len(person_list)):
        print(person_list[p])
        frame_list = glob(os.path.join(person_list[p], "*"))
        for f in range(len(frame_list)):
            print(frame_list[f])
            csv_file = os.path.join(scores_dir % i, "v%d.csv" % f)
            df = pd.read_csv(csv_file, sep=',')
            face_list = glob1(frame_list[f], "*.png")
            face_list.sort(key=lambda f: int(os.path.splitext(f)[0]))
            person_name = anonimityDict[i][str(p + 1)]
            for face in range(len(face_list)):
                print(face_list[face])
                if(face < len(df[person_name])):
                    if df[person_name][face] == -2:
                        get_ipython().system('cp {os.path.join(frame_list[f], face_list[face])} {os.path.join(out_dir, str(i), str(p + 1), "HD")}')
                    elif df[person_name][face] == -1:
                        get_ipython().system('cp {os.path.join(frame_list[f], face_list[face])} {os.path.join(out_dir, str(i), str(p + 1), "D")}')
                    elif df[person_name][face] == 0:
                        get_ipython().system('cp {os.path.join(frame_list[f], face_list[face])} {os.path.join(out_dir, str(i), str(p + 1), "M")}')
                    elif df[person_name][face] == 1:
                        get_ipython().system('cp {os.path.join(frame_list[f], face_list[face])} {os.path.join(out_dir, str(i), str(p + 1), "E")}')
                    elif df[person_name][face] == 2:
                        get_ipython().system('cp {os.path.join(frame_list[f], face_list[face])} {os.path.join(out_dir, str(i), str(p + 1), "HE")}')
                    else:
                        print("Level is incorrect.")
                        

# %%
