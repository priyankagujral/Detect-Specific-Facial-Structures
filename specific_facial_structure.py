from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
from collections import OrderedDict

facial_landmark_indexes = OrderedDict([
        ("mouth", (48, 68)),
        ("right_eyebrow", (17, 22)),
    	("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 35)),
        ("jaw", (0, 17))
        ])

def shape_to_numpy(shape, dtype="int"):
    np_array = np.zeros((68,2), dtype = dtype)
    for i in range(0, 68):
        np_array[i] = (shape.part(i).x, shape.part(i).y)
    return np_array

def rect_boundingBox(rectangle):
    left = rectangle.left()
    top = rectangle.top()
    right = rectangle.right()-left
    bottom = rectangle.bottom()-top
    return (left, top, right, bottom)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

input_image = cv2.imread("man_face.jpeg")
input_image = imutils.resize(input_image, width=750)
gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
faces = detector(gray, 1)

for (i, face) in enumerate(faces):
    mouth = predictor(gray, face)
    mouth_numpy = shape_to_numpy(mouth)
    (left, top, right, bottom) = rect_boundingBox(face)
    cv2.rectangle(input_image, (left, top), (left+right, top+bottom), (0, 255, 0), 2)
    
    for (counter, name) in enumerate(facial_landmark_indexes.keys()):
        (j, k) = facial_landmark_indexes[name]
        pts = mouth_numpy[j:k]
        clone = input_image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(clone, name, (10, 30), font, 1, (0, 255, 0), 2)
        for (i, (x, y)) in enumerate(pts):
            cv2.circle(clone, (x,y), 3, (0, 255, 0), -1)
        cv2.putText(input_image, "Face".format(i+1), (left-10, top-10), font, 0.5, (0, 255, 0), 2)
        cv2.imshow("image", clone)
        cv2.waitKey(0)
    

    



