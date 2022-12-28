import cv2
import mediapipe as mp
import numpy as np


# Calculate Angle

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


# Function for Sit Flag

def sit_Check(left_leg_angle, right_leg_angle):
    if left_leg_angle <= 90 and left_leg_angle >= 30 and right_leg_angle <= 90 and right_leg_angle >= 30:
        return True
    else:
        return False


# Function for Pick Flag

def pick_Check(left_leg_angle, right_leg_angle, left_hip_angle, right_hip_angle, right_wrist, left_wrist, right_hip, left_hip):
    if left_leg_angle <= 120 and right_leg_angle <= 120 and left_hip_angle <= 120 and right_hip_angle <= 120 and left_leg_angle >= 90 and right_leg_angle >= 90 and left_hip_angle >= 90 and right_hip_angle >= 90 and right_hip[1] > right_wrist[1] and left_hip[1] > left_wrist[1]:
        return True
    else:
        return False


# Function for Start Flag

def start_Check(left_hand_angle, right_hand_angle, left_leg_angle, right_leg_angle, left_hip_angle, right_hip_angle, nose, right_wrist, left_wrist):
    if left_hand_angle >= 160 and right_hand_angle >= 160 and right_leg_angle >= 160 and left_leg_angle >= 160 and right_hip_angle >= 160 and left_hip_angle >= 160 and nose[1] > right_wrist[1] and nose[1] > left_wrist[1]:
        return True
    else:
        return False


def weight_lifting(fname=None, foutname='output.mp4', configs=None):

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    dimension = [1080, 720]

    vid_writer = None
    sit_flag = False
    pick_flag = False
    lift_flag = False
    end_flag = True
    frame_Counter = 35
    current_frame = 0

    cap = cv2.VideoCapture(fname)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, dimension[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dimension[1])
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            #cap.set(cv2.CAP_PROP_FRAME_WIDTH, dimension[0])
            #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dimension[1])
            ret, frame = cap.read()

            if ret == False:
                break

            frame = cv2.resize(frame, (1540, 800))
            current_frame = current_frame + 1
            if vid_writer == None:
                vid_writer = cv2.VideoWriter(foutname, cv2.VideoWriter_fourcc(
                    *'mp4v'), 23, (frame.shape[1], frame.shape[0]))

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

                # Left Arm
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Left Leg
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Right Arm
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Right Leg
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # Nose
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                        landmarks[mp_pose.PoseLandmark.NOSE.value].y]

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

                #  Sit Flag
                if sit_flag == False:
                    sit_flag = sit_Check(left_leg_angle, right_leg_angle)

                #  Pick Flag
                if sit_flag == True and pick_flag == False:
                    pick_flag = pick_Check(
                        left_leg_angle, right_leg_angle, left_hip_angle, right_hip_angle, right_wrist, left_wrist, right_hip, left_hip)

                # Lift Flag
                if pick_flag == True and lift_flag == False:
                    lift_flag = start_Check(left_hand_angle, right_hand_angle, left_leg_angle,
                                            right_leg_angle, left_hip_angle, right_hip_angle, nose, right_wrist, left_wrist)

                # Final Stage
                if lift_flag == True and frame_Counter > 0:
                    frame_Counter = frame_Counter - 1
                    if left_leg_angle <= 160 or right_leg_angle <= 160 or nose[1] < right_wrist[1] or nose[1] < left_wrist[1]:
                        end_flag = False

                    if left_hip_angle <= 150 or right_hip_angle <= 150 or nose[1] < right_wrist[1] or nose[1] < left_wrist[1]:
                        end_flag = False

                    if left_hand_angle <= 160 or right_hand_angle <= 160 or nose[1] < right_wrist[1] or nose[1] < left_wrist[1]:
                        end_flag = False

                # Visualize angle
                cv2.putText(image, str(left_hand_angle),
                            tuple(np.multiply(left_elbow, [
                                  1500, 800]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.putText(image, str(right_hand_angle),
                            tuple(np.multiply(right_elbow, [
                                  1500, 800]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.putText(image, str(left_leg_angle),
                            tuple(np.multiply(left_knee, [
                                  1500, 800]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,
                                                            255, 0), 2, cv2.LINE_AA
                            )

                cv2.putText(image, str(right_leg_angle),
                            tuple(np.multiply(right_knee, [
                                  1500, 800]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,
                                                            255, 0), 2, cv2.LINE_AA
                            )

                cv2.putText(image, str(left_hip_angle),
                            tuple(np.multiply(
                                left_hip, [1500, 800]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.putText(image, str(right_hip_angle),
                            tuple(np.multiply(right_hip, [
                                  1500, 800]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            except:
                pass

            # Setup Status Box
            cv2.rectangle(image, (0, 0), (235, 130), (123, 255, 255), -1)

            # Fill the Box with Flags
            cv2.putText(image, "Sit Posture : {}".format(
                sit_flag), (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(image, "Pick Posture : {}".format(
                pick_flag), (5, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(image, "Lift Posture : {}".format(
                lift_flag), (5, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            if (end_flag == True and lift_flag == True and frame_Counter == 0):
                cv2.putText(image, "Result : SUCCESS :)", (5, 118),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            elif ((end_flag == False and frame_Counter == 0) or ((sit_flag == False or pick_flag == False or lift_flag == False) and length - current_frame < 40)):
                cv2.putText(image, "Result : FAIL !!! ", (5, 118),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Result : Wait....", (5, 118),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            # Render Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            #cv2.imshow('MediaPipe Feed', image)
            vid_writer.write(image)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

    # Return Success or Fail
    if end_flag == True and lift_flag == True:
        ans = True  # "Success"
    else:
        ans = False  # "Fail"
    vid_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    return ans


if __name__ == "__main__":
    print(weight_lifting("Fail3.mp4"))
