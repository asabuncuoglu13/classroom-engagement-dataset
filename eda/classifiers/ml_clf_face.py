# %%
import os
import numpy as np
import pandas as pd

# %%
eng_levels = [-2, -1, 0, 1, 2, 3]

# %% [markdown]
# ## Face Features

# %%
person_name = "cansu"
base_dir = "../face/features/%s" % person_name

# %%
labels = pd.read_csv(os.path.join(base_dir, 'levels.tsv'), delimiter='\t', header=None)
scores = pd.read_csv(os.path.join(base_dir, 'scores.tsv'), delimiter='\t', header=None)

# %%
# Load data
df = pd.read_csv(os.path.join(base_dir, 'features.csv'))
# Remove empty spaces in column names.
df.columns = [col.replace(" ", "") for col in df.columns]
# Print few values of data.
df.head()

# %%
print(len(df), len(labels))

# %%
df.describe()

# %%
high_conf_ind = ~np.logical_or(df['confidence'] < 0.5, df['success'] == 0)

df = df.loc[high_conf_ind]
labels = labels.loc[high_conf_ind]
scores = scores.loc[high_conf_ind]

# %%
print(len(df), len(labels))

# %%
# Define Feature Series Ranges
r_au_intensities = range(df.columns.get_loc("AU01_r"), df.columns.get_loc("AU45_r"))
r_au_class = range(df.columns.get_loc("AU01_c"), df.columns.get_loc("AU45_c"))
r_3d_eye_landmarks = range(df.columns.get_loc("eye_lmk_X_0"), df.columns.get_loc("eye_lmk_Z_55"))
r_gaze_directions = range(df.columns.get_loc("gaze_0_x"), df.columns.get_loc("gaze_angle_y"))
r_pose = range(df.columns.get_loc("pose_Tx"), df.columns.get_loc("pose_Rz"))
r_3d_face_landmarks = range(df.columns.get_loc("X_0"), df.columns.get_loc("Z_67"))

# %%
df_au_intensities = df.iloc[:, r_au_intensities]
df_au_class = df.iloc[:, r_au_class]
df_3d_eye_landmarks = df.iloc[:, r_3d_eye_landmarks]
df_gaze_directions = df.iloc[:, r_gaze_directions]
df_pose = df.iloc[:, r_pose]
df_3d_face_landmarks = df.iloc[:, r_3d_face_landmarks]

# %%
df_au_intensities['label'] = labels.values
df_au_class['label'] = labels.values
df_3d_eye_landmarks['label'] = labels.values
df_gaze_directions['label'] = labels.values
df_pose['label'] = labels.values
df_3d_face_landmarks['label'] = labels.values

df_au_intensities['score'] = scores.values
df_au_class['score'] = scores.values
df_3d_eye_landmarks['score'] = scores.values
df_gaze_directions['score'] = scores.values
df_pose['score'] = scores.values
df_3d_face_landmarks['score'] = scores.values


# %%
df_face_and_pose = pd.concat([df_3d_face_landmarks.iloc[:, :-2],
df_pose],axis=1)

df_all = pd.concat([df_3d_eye_landmarks.iloc[:, :-2], 
df_au_intensities.iloc[:, :-2],
df_gaze_directions.iloc[:, :-2],
df_3d_face_landmarks.iloc[:, :-2],
df_pose],axis=1)


feature_sets = {
    "AU Intensity": df_au_intensities,
    "3D Eye Landmark": df_3d_eye_landmarks,
    "3D Face Landmark": df_3d_face_landmarks,
    "Gaze Directions": df_gaze_directions,
    "Head Pose": df_pose,
    "3D Face and Head Pose": df_face_and_pose,
    "All OpenFace Fts": df_all
}

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score


# %%

classifier_names = ['LR', 'knn', 'rbf svm', 'random forest', 'boosted trees']
classifiers = [LogisticRegression(random_state=42, solver="liblinear"),
                KNeighborsClassifier(n_neighbors=6),
                SVC(gamma=2, C=1),
                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                GradientBoostingClassifier(n_estimators=10, learning_rate=1, max_depth=5)]

results = pd.DataFrame(columns= ['LR', 'knn', 'rbf svm', 'random forest', 'boosted trees', 'title'])

for title in feature_sets:
    s = [0, 0, 0, 0, 0, "title"]
    dfc = feature_sets[title]
    not_zero_ind = ~(dfc == 0).all(axis=1)

    dfc = dfc.loc[not_zero_ind]
    labels = dfc['label'].loc[not_zero_ind]


    scaler = StandardScaler()
    scaled_samples = scaler.fit_transform(dfc.iloc[:,:-2])

    X_train, X_test, y_train, y_test = train_test_split(scaled_samples, labels, test_size=0.2, random_state=42, stratify=labels)

    i = 0
    for model in classifiers:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        res = f1_score(y_test, y_pred, average='weighted')
        s[i] = res
        i +=1
    s[i] = title
    results.loc[len(results.index)] = s
    #results.head()

# %%
results.to_csv('reports/f1_scores_%s.csv' % person_name)

# %%

