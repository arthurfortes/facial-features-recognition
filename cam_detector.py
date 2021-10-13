import silence_tensorflow.auto
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as img_keras
from collections import deque
import mediapipe as mp
import numpy as np
import pyshine as ps
import math
import cv2


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

Q_race = deque(maxlen=30)
Q_age = deque(maxlen=30)
Q_gender = deque(maxlen=30)
writer = None
padding = True

# parameters for loading data and images
race_label = ['Black', 'East Asian', 'Indian', 
'Latino_Hispanic', 'Middle Eastern', 
'Southeast Asian', 'White']

gender_label = ['Female', 'Male']

age_label = ['0-2','10-19', '20-29', '3-9', '30-39',
'40-49', '50-59', '60-69', 'more than 70']

age_weights = [4.45811712,  1.00083209,  0.37950401,  0.80876815,  0.52459874, 
                0.95672924,  1.62408858,  3.86007702, 12.68776371]

model_path = 'models/last_model_mnv2_tf24.h5'
out_video_path = 'output/video_cam.avi'

# loading models
face_model = load_model(model_path, compile=False)

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        no_ann_img = image.copy()

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

                h, w, c = no_ann_img.shape
                cx_min=  w
                cy_min = h
                cx_max= cy_max= 0

                for id, lm in enumerate(face_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if cx < cx_min:
                        cx_min = cx if cx >= 0 else 0
                    if cy < cy_min:
                        cy_min = cy if cy >= 0 else 0
                    if cx > cx_max:
                        cx_max = cx if cx >= 0 else 0
                    if cy > cy_max:
                        cy_max = cy if cy >= 0 else 0
                
                # crop detected face      
                detected_face = no_ann_img[int(cy_min):int(cy_max), int(cx_min):int(cx_max)]
                
                # inference
                face = cv2.flip(detected_face, 1)
                face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_CUBIC)
                face = np.expand_dims(face, axis=0)
                face = face/ 255.

                prediction = face_model.predict(face)

                race_value = prediction[0]
                gender_value = prediction[1]
                age_value = prediction[2]*age_weights

                Q_race.append(race_value)
                Q_gender.append(gender_value)
                Q_age.append(age_value)
                
                # # perform prediction averaging over the current history of previous predictions
                results_race = np.array(Q_race).mean(axis=0)
                idx_race = np.argmax(results_race)

                results_gender = np.array(Q_gender).mean(axis=0)
                idx_gender = np.argmax(results_gender)

                results_age = np.array(Q_age).mean(axis=0)
                idx_age = np.argmax(results_age)

                # write text above rectangle       
                cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (36, 255, 12), 2)

                try:
                    ps.putBText(image, f"Age: {age_label[idx_age]}", text_offset_x=cx_max+10, text_offset_y=cy_min+15, vspace=8, hspace=5, font_scale=.8, background_RGB=(0,0,0), text_RGB=(255,255,255), thickness=1)
                    ps.putBText(image, f"Gender: {gender_label[idx_gender]}", text_offset_x=cx_max+10, text_offset_y=cy_min+55, vspace=8, hspace=5, font_scale=.8, background_RGB=(0,0,0), text_RGB=(255,255,255), thickness=1)
                    ps.putBText(image, f"Race: {race_label[idx_race]}", text_offset_x=cx_max+10, text_offset_y=cy_min+95, vspace=8, hspace=5, font_scale=.5, background_RGB=(0,0,0), text_RGB=(255,255,255), thickness=1)
                except ValueError:
                    pass

                out = image.copy()
                shapes = np.zeros_like(image, np.uint8)
                alpha = 0.5
                mask = shapes.astype(bool)
                out[mask] = cv2.addWeighted(image, alpha, shapes, 1 - alpha, 0)[mask]
                
                
        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            h, w, c = image.shape
            fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            writer = cv2.VideoWriter(out_video_path, fourcc, 20, (w, h), True)

        # write the output frame to disk
        writer.write(image)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh', image)
        if cv2.waitKey(5) & 0xFF == 27:
          break
  
cap.release()