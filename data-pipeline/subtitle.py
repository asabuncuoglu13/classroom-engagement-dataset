# %%
import pysrt
import os
import pandas as pd
import numpy as np
from scipy.stats import mode

# %%
file_path = "C:\\Users\\ASABUNCUOGLU13\\Documents\\data\\srt"
file_name = ["SE2-1%s", "SE2-2%s", "SE2-3%s", "SE2-4%s"]
level_file = ["../scores/vol02/group/levels/0.csv",
              "../scores/vol02/group/levels/1.csv",
              "../scores/vol02/group/levels/2.csv",
              "../scores/vol02/group/levels/3.csv"]

# %%
i = 0
for f in file_name:
    subs = pysrt.open(os.path.join(file_path, f % ".srt"), encoding='utf8')

    df_level = pd.read_csv(level_file[i])
    df_cats = df_level.columns

    d = []
    for sub in subs:
        s = sub.start.minutes * 60 + sub.start.seconds
        e = sub.end.minutes * 60 + sub.end.seconds
        if(s == e):
            e = e + 1
        d.append({
            'text': ' '.join(sub.text.split()),
            df_cats[0]:  mode(df_level[df_cats[0]][s: e])[0][0],
            df_cats[1]:  mode(df_level[df_cats[1]][s: e])[0][0],
            df_cats[2]:  mode(df_level[df_cats[2]][s: e])[0][0],
            df_cats[3]:  mode(df_level[df_cats[3]][s: e])[0][0]
            #df_cats[4]:  mode(df_level[df_cats[4]][s: e])[0][0]
        })

    df_sub_with_eng = pd.DataFrame(d)

    df_sub_with_eng.to_csv(os.path.join("../transcript", f %
                           "-subtitle.csv"), index=False)

    i = i + 1

# %%
