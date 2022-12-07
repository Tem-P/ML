import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
dimension = [1280, 720]

sit_flag = False
pick_flag = False
elbow_angle = False
start_flag = False
# Calculate Angle


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[1])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("C:/Users/Amar Suryawanshi/Downloads/WL.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, dimension[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dimension[1])

# setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
    while cap.isOpened():
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, dimension[0])
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dimension[1])
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1540, 800))
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = pose.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
            left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].z]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
            right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].z]

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate Angle
            left_hand_angle = calculate_angle(
                left_shoulder, left_elbow, left_wrist)

            right_hand_angle = calculate_angle(
                right_shoulder, right_elbow, right_wrist)

            left_leg_angle = calculate_angle(
                left_hip, left_knee, left_ankle)

            right_leg_angle = calculate_angle(
                right_hip, right_knee, right_ankle)

            left_hip_angle = calculate_angle(
                left_shoulder, left_hip, left_knee)

            right_hip_angle = calculate_angle(
                right_shoulder, right_hip, right_knee)

            if sit_flag == False:
                if left_leg_angle <= 75 and right_leg_angle <= 75 and left_hip_angle <= 75 and right_hip_angle <= 75:
                    sit_flag = True
                    print("Sit Flag True")
                    # time.sleep(1)

            if sit_flag == True and pick_flag == False:
                if left_leg_angle >= 120 and right_leg_angle >= 120 and left_hip_angle >= 90 and right_hip_angle >= 90:
                    pick_flag = True
                    print("Pick Flag True")
                    # time.sleep(1)

            if pick_flag == True and start_flag == False:
                if left_hand_angle >= 90 and right_hand_angle >= 90 and right_leg_angle >= 90 and left_leg_angle >= 90 and right_hip_angle >= 90 and left_hip_angle >= 90 and right_eye[1] < right_wrist[1] and left_eye[1] < left_wrist[1]:
                    start_flag = True
                    start_time = time.time()
                    print("Start Flag True".format(start_time))
                    # time.sleep(1)
            # Visualize angle
            # cv2.putText(image, str(left_hand_angle),
             #           tuple(np.multiply(left_elbow, dimension).astype(int)),
              #          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # cv2.putText(image, str(right_hand_angle),
             #           tuple(np.multiply(right_elbow, dimension).astype(int)),
              #          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # cv2.putText(image, str(left_leg_angle),
             #           tuple(np.multiply(left_knee, dimension).astype(int)),
              #          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # cv2.putText(image, str(right_leg_angle),
             #           tuple(np.multiply(right_knee, dimension).astype(int)),
              #          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

           # print(landmarks)
        except:
            pass

        # Render Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(
                                      color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('MediaPipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
