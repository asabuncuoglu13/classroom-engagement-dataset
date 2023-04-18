
# %%
import os
from pathlib import Path
from PIL import Image
import cv2
import mediapipe as mp

# %%
base_dir = "C:\\Users\\ASABUNCUOGLU13\\Documents\\data\\vol05\\face"
output_folder = os.path.join("C:\\Users\\ASABUNCUOGLU13\\Documents\\data\\vol05\\face-center\\%d")

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

#!mkdir {os.path.join(input_folder, "face-center")}
#!mkdir {output_folder % 4}
!mkdir {output_folder % 3}

# %%
for i in range(12):

  input_folder = os.path.join(base_dir, "frames-%d" % i)
  images = Path(input_folder).glob('*.png')

  !mkdir {os.path.join(output_folder % 3 , str(i))}
  #!mkdir {os.path.join(output_folder % 5 , str(i))}

  with mp_face_detection.FaceDetection(
      model_selection=1, min_detection_confidence=0.5) as face_detection:
    for idx, file in enumerate(images):
      image = cv2.imread(str(file))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
      results = face_detection.process(image)
      h, w, c = image.shape
      # Draw face detections of each face.
      if not results.detections:
        continue
      #annotated_image = image.copy()
      for detection in results.detections:
        data = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP)
        x= data.x*w 

        """
        if(x <= 720 and x > 200):
          n = 4
        if(x < 1380 and x > 720):
          n = 5
        """
        n = 3

        x = int(x) - 160
        y = data.y*h 
        y = int(y) - 160
        face = Image.fromarray(image).crop([x, y, x+320, y+320])
        
        if (x < 1170 and x > 810):
          face.save(output_folder % n + "\\" + str(i) + "\\" + str(idx) + ".png")

# %%



