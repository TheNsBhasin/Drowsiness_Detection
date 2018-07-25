# Drowsiness Detection
A Python script to detect driver drowsiness in a real-time and sound an alarm if driver appears to be drowsy.

## Setup

### Dependencies
```Linux
pip install -r requirements.txt
```

### Execution
```Linux
python detect_drowsiness.py \
--shape-predictor shape_predictor_68_face_landmarks.dat \
--alarm alarm.wav
```

## Description
A computer vision system that uses a real-time algorithm to detect eye blinks in a video sequence from a standard camera and sound an alarm if the driver appears to be drowsy.

### Algorithm
Given an input image (and normally an ROI that specifies the object of interest), a shape predictor attempts to localize key points of interest along the shape.
Detecting facial landmarks is therefore a two step process:

1. Localize the face in the image.
2. Detect the key facial structures on the face ROI.

![alt text](https://github.com/TheNsBhasin/Drowsiness_Detection/blob/master/facial_landmarks_68markup-768x619.jpg "Facial landmarks")

Now, each eye is represented by 6 (x, y)-coordinates, starting at the left-corner of the eye (as if you were looking at the person), and then working clockwise around the eye:

![alt text](https://github.com/TheNsBhasin/Drowsiness_Detection/blob/master/eye_landmark.jpg "Eye Landmarks")

### Condition
For every video frame, the eye landmarks are detected. The eye aspect ratio (EAR) between height and width of the eye is computed. If the eye aspect ratio value is less than a threshold value for sufficient time, an alarm is set to wake up the driver.

![alt text](https://github.com/TheNsBhasin/Drowsiness_Detection/blob/master/EAR.png "EAR")

### Plots
![alt text](https://github.com/TheNsBhasin/Drowsiness_Detection/blob/master/blink_detection_plot.jpg "Blink detection plot")


## References
[Drowsiness detection with OpenCV](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)

[Real-Time Eye Blink Detection using Facial Landmarks](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
