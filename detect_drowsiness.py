import numpy as np
import cv2
import imutils
import argparse
import dlib
import time
from pygame import mixer
from scipy.spatial import distance as dist
from imutils import face_utils
from threading import Thread


# Utility function to compute Eye aspect ratio
def eye_aspect_ratio(eye):
    # Euclidean distance between two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Euclidean distance between horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # Eye Aspect Ration(EAR)
    EAR = (A + B) / (2.0 * C)

    # Return the eye aspect ratio
    return EAR


# Utility function to play alarm sound
def sound_alarm(path):
    mixer.init()
    mixer.music.load(path)
    mixer.music.play()


if __name__ == '__main__':
    # Construct argument parser and parse the input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--shape-predictor', required=True,
                    help='Path to facial landmark predictor')
    ap.add_argument('-a', '--alarm', type=str, default='',
                    help='Path alarm .WAV file')
    args = vars(ap.parse_args())

    # Initialize dlib's face detector (HOG - based) and then create facial landmark predictor
    print('[INFO] Loading facial landmark predictor')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args['shape_predictor'])

    # Threshold EAR to inticate eye blink
    EAR_THRESH = 0.2

    # Number of consecutive frames for which EAR must be below threshold to set off alarm
    EAR_CONSEC_FRAMES = 20

    # Frame counter
    COUNTER = 0

    # Boolean to indicate alarm status
    ALARM_ON = False

    # Extract start and end index for landmarks for both eyes
    (left_eye_st, left_eye_en) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (right_eye_st, right_eye_en) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

    print("[INFO] Starting video stream thread")
    # Start video capture using OpenCV
    cap = cv2.VideoCapture(0)
    time.sleep(1.0)

    while True:
        # Read frame from threaded video stream
        ret, frame = cap.read()

        # Resize frame
        frame = imutils.resize(frame, width=450)

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        ROI = detector(gray, 0)

        # Loop over each region of interest (ROI)
        for rect in ROI:
            # Determine 68 facial landmarks for the face region
            face = predictor(gray, rect)

            # Convert shape to NumPy array
            face = face_utils.shape_to_np(face)

            # Extract left and right eye coordinates from face
            left_eye = face[left_eye_st: left_eye_en]
            right_eye = face[right_eye_st: right_eye_en]

            # Compute EAR for both eyes
            left_eye_ear = eye_aspect_ratio(left_eye)
            right_eye_ear = eye_aspect_ratio(right_eye)

            # Compute mean EAR
            ear = (left_eye_ear + right_eye_ear) / 2.0

            # Constuct convex hull for both eyes
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

            # Check if EAR is below threshold
            if ear < EAR_THRESH:
                COUNTER += 1

                # If eyes were closed for sufficient time
                if COUNTER >= EAR_CONSEC_FRAMES:
                    # Turn on alarm, if not already
                    if not ALARM_ON:
                        print('[Info] ALERT!')
                        ALARM_ON = True

                        # Check if alarm file was provided in input, if so sound alarm in background thread
                        if args['alarm'] != '':
                            t = Thread(target=sound_alarm,
                                       args=(args["alarm"],))
                            t.deamon = True
                            t.start()

                    # Draw an alert on the frame
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Otherwise, the eye aspect ratio is not below the blink threshold, so reset the counter and alarm
            else:
                COUNTER = 0
                ALARM_ON = False

            # Draw the computed eye aspect ratio on the frame to help
            # with debugging and setting the correct eye aspect ratio
            # thresholds and frame counters
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        # If the `q` key was pressed, break from the loop
        if key == ord('q'):
            break

    # Cleanup
    cv2.destroyAllWindows()
    cap.release()
