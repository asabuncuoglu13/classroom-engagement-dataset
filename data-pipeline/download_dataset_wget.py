# %%
#!pip install youtube_dl

# %%
import os
from pathlib import Path
import pandas as pd

# %%
df = pd.read_csv("youtube_links_public.csv")

# %%
df.head()

# %%
for f in df['Session'].unique():
    !mkdir {f}
    !mkdir {"%s/frames" % f}

# %%
for i in range(len(df)):
    !youtube-dl {df['Group Video Link'][i]} -o {"./%s/%s.mp4"  % (df['Session'][i], df['Video'][i])}

# %% [markdown]
# # Create Frames

# %%
video_name = "v%d.mp4"
output_name = "f%d.png"
in_folder = "./%d"
output_folder = "./%d/frames/%d"

for i in [2,3,5,6]:
    vidfiles = Path(in_folder % i).glob('*.mp4')
    ind = 0
    for f in vidfiles:
        !mkdir {output_folder % (i, ind)}
        !ffmpeg -i {f} -vf fps=1 {os.path.join(output_folder % (i, ind), output_name)}
        ind = ind + 1

# %% [markdown]
# # Download Centered Face Frames

# %%
download_gdrive = "wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={}\" -O {} && rm -rf /tmp/cookies.txt"

# %%
download_gdrive.format("1QWhBsqkg4-bahUlsk6UOyqZo6chWZFM5", "1QWhBsqkg4-bahUlsk6UOyqZo6chWZFM5", "./2/engagement-slices.zip")

# %%
!{download_gdrive.format("1QWhBsqkg4-bahUlsk6UOyqZo6chWZFM5", "1QWhBsqkg4-bahUlsk6UOyqZo6chWZFM5", "./2/face-center.zip")}
!{download_gdrive.format("1vVNOuP59g4fy3ISS-cN-c44TsO-6cD85", "1vVNOuP59g4fy3ISS-cN-c44TsO-6cD85", "./3/face-center.zip")}
## 4 is deleted due to preserving participant privacy
!{download_gdrive.format("1OddNJwPjS_ST1lmqxqdWA9ETONfBUttA", "1OddNJwPjS_ST1lmqxqdWA9ETONfBUttA", "./5/face-center.zip")}
!{download_gdrive.format("1cCg9TZDw5dmA_x0oTjyN2IJVTZyfBckJ", "1cCg9TZDw5dmA_x0oTjyN2IJVTZyfBckJ", "./6/face-center.zip")}
## 7 is deleted due to preserving participant privacy
## 8 is deleted due to preserving participant privacy

# %%
!unzip ./2/face-center.zip -d ./2
!rm ./2/face-center.zip

!unzip ./3/face-center.zip -d ./3
!rm ./3/face-center.zip

!unzip ./5/face-center.zip -d ./5
!rm ./5/face-center.zip

!unzip ./6/face-center.zip -d ./6
!rm ./6/face-center.zip

# %% [markdown]
# # Download Engagement Slices

# %%
!{download_gdrive.format("1QWhBsqkg4-bahUlsk6UOyqZo6chWZFM5", "1QWhBsqkg4-bahUlsk6UOyqZo6chWZFM5", "./2/engagement-slices.zip")}
!{download_gdrive.format("1X-8ZFJgtUFEsu8Je1YZ8nKa6cWnmk0gw", "1X-8ZFJgtUFEsu8Je1YZ8nKa6cWnmk0gw", "./3/engagement-slices.zip")}
## 4 is deleted due to preserving participant privacy
!{download_gdrive.format("11mQaeyYqEy5py8PAvir7e8D6-lFqf2KW", "11mQaeyYqEy5py8PAvir7e8D6-lFqf2KW", "./5/engagement-slices.zip")}
!{download_gdrive.format("1g9r-3WnS_cND8vB9HJI0bz-AE40363wr", "1g9r-3WnS_cND8vB9HJI0bz-AE40363wr", "./6/engagement-slices.zip")}
## 7 is deleted due to preserving participant privacy
## 8 is deleted due to preserving participant privacy

# %%
!unzip ./2/engagement-slices.zip -d ./2
!rm ./2/engagement-slices.zip

!unzip ./3/engagement-slices.zip -d ./3
!rm ./3/engagement-slices.zip

!unzip ./5/engagement-slices.zip -d ./5
!rm ./5/engagement-slices.zip

!unzip ./6/engagement-slicesr.zip -d ./6
!rm ./6/engagement-slices.zip


