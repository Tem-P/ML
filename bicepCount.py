import cv2
import mediapipe as mp
import numpy as np


# Calculating the angle between the three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = (np.arctan2(c[1] - b[1], c[0] - b[0]) -
               np.arctan2(a[1] - b[1], a[0] - b[0]))
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return round(angle, 2)


def bicepCount(fname=0, config=None):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    dimension = [1500, 800]
    vid_writer = None
    left_count = 0
    right_count = 0

    # For webcam input:
    cap = cv2.VideoCapture(fname)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, dimension[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dimension[1])

    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break

            frame = cv2.resize(frame, (1540, 800))

            if vid_writer == None:
                vid_writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(
                    *'mp4v'), 10, (frame.shape[1], frame.shape[0]))

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

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]

                # Calculate angle

                left_hand_angle = calculate_angle(
                    left_shoulder, left_elbow, left_wrist)

                right_hand_angle = calculate_angle(
                    right_shoulder, right_elbow, right_wrist)

                # Visualize angle
                cv2.putText(image, str(left_hand_angle),
                            tuple(np.multiply(left_elbow, [
                                  1500, 800]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.putText(image, str(right_hand_angle),
                            tuple(np.multiply(right_elbow, [
                                  1500, 800]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # Counting the biceps
                if left_hand_angle > 160:
                    left_stage = "down"
                if left_hand_angle < 30 and left_stage == "down":
                    left_stage = "up"
                    left_count = left_count+1

                if right_hand_angle > 160:
                    right_stage = "down"
                if right_hand_angle < 30 and right_stage == "down":
                    right_stage = "up"
                    right_count = right_count+1
            except:
                pass

            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
            # Render left count
            cv2.putText(image, 'Left Count: ' + str(left_count), (95, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Render right count
            cv2.putText(image, 'Right Count: ' + str(right_count), (95, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            cv2.imshow('MediaPipe Feed', image)
            vid_writer.write(image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    vid_writer.release()
    cap.release()
    cv2.destroyAllWindows()


bicepCount()
