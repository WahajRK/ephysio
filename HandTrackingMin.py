import cv2
import mediapipe as mp
import time
import threading
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Curl Counter
stage = None
holdup = 0
holddown = 0
counter = 0
global seconds
timer = 0

e_physio = int(input('Enter Exercise number'))
match e_physio:
    case 1:
    # VIDEO FEED
        cap = cv2.VideoCapture(0)

        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()

        # Color Image to RBG
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
        # Make Detection Here
                results = pose.process(image)
        # Recolor Image to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extracting LandMarks T-T
                try:
                    landmarks = results.pose_landmarks.landmark

                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    angle = calculate_angle(shoulder, elbow, wrist)

                    def timer(seconds):
                        for i in range(seconds):
                            seconds = (seconds - i)
                            print(seconds)

            # Visualize
                    cv2.putText(image, str(angle),
                                tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                    print(landmarks)

            # Curl counting
                    if angle > 160:
                        stage = 'Down'
                    if angle < 66 and stage == 'Down':
                        stage = 'Hold Up'
                        counter += 1
                        print(counter)
                    if counter > 1:
                        x = threading.Thread(target=time(seconds), args=())
                        x.start()
                    if angle >= 55 and angle <= 65:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(000, 240, 0), thickness=1, circle_radius=2),
                                                  mp_drawing.DrawingSpec(color=(100, 250, 100), thickness=2, circle_radius=1)
                                                  )
                    else:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(239, 240, 239), thickness=1, circle_radius=2),
                                                  mp_drawing.DrawingSpec(color=(255, 230, 000), thickness=2, circle_radius=1)
                                                  )

            # def countdown():
            #     global timer
            #     timer = 10
            #     for x in range(10):






                except:
                    pass

        # Rectangular Box at top corner with parameters
                cv2.rectangle(image, (0, 0), (255, 73), (245, 117, 16), -1)
                cv2.circle(image, (560, 400), 50, (0, 0, 0), -1)
                cv2.putText(image, seconds, (554, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Rendering counts on cv2 :)
                cv2.putText(image, 'REPS: ', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Rendering STAGE on cv2 :)
                cv2.putText(image, 'STAGE: ', (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, (120, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)


        # Rendering To Estimate Pose
        # if angle >= 55 and angle <= 65:
        #     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                               mp_drawing.DrawingSpec(color=(000, 240, 0), thickness=1, circle_radius=2),
        #                               mp_drawing.DrawingSpec(color=(255, 230, 000), thickness=2, circle_radius=1)
        #                               )
        # else:
        #     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                               mp_drawing.DrawingSpec(color=(239, 240, 239), thickness=1, circle_radius=2),
        #                               mp_drawing.DrawingSpec(color=(255, 230, 000), thickness=2, circle_radius=1)
        #                               )

        # landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
        # landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        # landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

                def calculate_angle(a, b, c):
                    a = np.array(a)
                    b = np.array(b)
                    c = np.array(c)
                    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    angle = np.abs(radians * 180.0 / np.pi)
                    if angle > 180.0:
                        angle = 360 - angle
                    return angle


                cv2.imshow('E-Physiotherapist', image)

                if cv2.waitKey(10) & 0XFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
    case 2:
        # VIDEO FEED
        cap = cv2.VideoCapture(0)

        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()

                # Color Image to RBG
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                # Make Detection Here
                results = pose.process(image)
                # Recolor Image to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extracting LandMarks T-T
                try:
                    landmarks = results.pose_landmarks.landmark

                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    angle = calculate_angle(hip, shoulder, elbow)



                    # Visualize
                    cv2.putText(image, str(angle),
                                tuple(np.multiply(shoulder, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                    print(landmarks)

                    # Curl counting
                    if angle > 175:
                        stage = 'Hold Up'
                    if angle < 40 and stage == 'Hold Up':
                        stage = 'Down'
                        counter += 1
                        print(counter)
                    if angle >=40  and angle <= 175:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(000, 240, 0), thickness=1,
                                                                         circle_radius=2),
                                                  mp_drawing.DrawingSpec(color=(100, 250, 100), thickness=2,
                                                                         circle_radius=1)
                                                  )
                    else:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(239, 240, 239), thickness=1,
                                                                         circle_radius=2),
                                                  mp_drawing.DrawingSpec(color=(255, 230, 000), thickness=2,
                                                                         circle_radius=1)
                                                  )


                except:
                    pass

                # Rectangular Box at top corner with parameters
                cv2.rectangle(image, (0, 0), (255, 73), (245, 117, 16), -1)


                # Rendering counts on cv2 :)
                cv2.putText(image, 'REPS: ', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                            cv2.LINE_AA)

                # Rendering STAGE on cv2 :)
                cv2.putText(image, 'STAGE: ', (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, (120, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

                def calculate_angle(a, b, c):
                    a = np.array(a)
                    b = np.array(b)
                    c = np.array(c)
                    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    angle = np.abs(radians * 180.0 / np.pi)
                    if angle > 180.0:
                        angle = 360 - angle
                    return angle


                cv2.imshow('E-Physiotherapist', image)

                if cv2.waitKey(10) & 0XFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
    case 3:
        # VIDEO FEED
        cap = cv2.VideoCapture(0)

        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()

                # Color Image to RBG
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                # Make Detection Here
                results = pose.process(image)
                # Recolor Image to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extracting LandMarks T-T
                try:
                    landmarks = results.pose_landmarks.landmark

                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    angle = calculate_angle(shoulder, hip, knee)

                    # Visualize
                    cv2.putText(image, str(angle),
                                tuple(np.multiply(hip, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                    print(landmarks)

                    # Curl counting
                    if angle > 175:
                        stage = 'Hold Up'
                    if angle < 40 and stage == 'Hold Up':
                        stage = 'Down'
                        counter += 1
                        print(counter)
                    if angle >= 40 and angle <= 120:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(000, 240, 0), thickness=1,
                                                                         circle_radius=2),
                                                  mp_drawing.DrawingSpec(color=(100, 250, 100), thickness=2,
                                                                         circle_radius=1)
                                                  )
                    else:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(239, 240, 239), thickness=1,
                                                                         circle_radius=2),
                                                  mp_drawing.DrawingSpec(color=(255, 230, 000), thickness=2,
                                                                         circle_radius=1)
                                                  )


                except:
                    pass

                # Rectangular Box at top corner with parameters
                cv2.rectangle(image, (0, 0), (255, 73), (245, 117, 16), -1)

                # Rendering counts on cv2 :)
                cv2.putText(image, 'REPS: ', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                            cv2.LINE_AA)

                # Rendering STAGE on cv2 :)
                cv2.putText(image, 'STAGE: ', (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, (120, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)


                def calculate_angle(a, b, c):
                    a = np.array(a)
                    b = np.array(b)
                    c = np.array(c)
                    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    angle = np.abs(radians * 180.0 / np.pi)
                    if angle > 180.0:
                        angle = 360 - angle
                    return angle


                cv2.imshow('E-Physiotherapist', image)

                if cv2.waitKey(10) & 0XFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    case 4:
        # VIDEO FEED
        cap = cv2.VideoCapture(0)

        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()

                # Color Image to RBG
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                # Make Detection Here
                results = pose.process(image)
                # Recolor Image to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extracting LandMarks T-T
                try:
                    landmarks = results.pose_landmarks.landmark

                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    angle = calculate_angle(hip, knee, ankle)

                    # Visualize
                    cv2.putText(image, str(angle),
                                tuple(np.multiply(knee, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                    print(landmarks)

                    # Curl counting
                    if angle > 175:
                        stage = 'Hold Up'
                    if angle < 40 and stage == 'Hold Up':
                        stage = 'Down'
                        counter += 1
                        print(counter)
                    if angle >= 40 and angle <= 175:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(000, 240, 0), thickness=1,
                                                                         circle_radius=2),
                                                  mp_drawing.DrawingSpec(color=(100, 250, 100), thickness=2,
                                                                         circle_radius=1)
                                                  )
                    else:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(239, 240, 239), thickness=1,
                                                                         circle_radius=2),
                                                  mp_drawing.DrawingSpec(color=(255, 230, 000), thickness=2,
                                                                         circle_radius=1)
                                                  )


                except:
                    pass

                # Rectangular Box at top corner with parameters
                cv2.rectangle(image, (0, 0), (255, 73), (245, 117, 16), -1)

                # Rendering counts on cv2 :)
                cv2.putText(image, 'REPS: ', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                            cv2.LINE_AA)

                # Rendering STAGE on cv2 :)
                cv2.putText(image, 'STAGE: ', (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, (120, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)


                def calculate_angle(a, b, c):
                    a = np.array(a)
                    b = np.array(b)
                    c = np.array(c)
                    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    angle = np.abs(radians * 180.0 / np.pi)
                    if angle > 180.0:
                        angle = 360 - angle
                    return angle


                cv2.imshow('E-Physiotherapist', image)

                if cv2.waitKey(10) & 0XFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

