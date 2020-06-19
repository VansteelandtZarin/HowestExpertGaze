import pandas as pd
from sympy import Point3D, Line3D, Plane
import numpy as np
import cv2 as cv

# net = cv.dnn.readNet("TinyWeightsZarinV7.weights", "TinyConfigZarin.cfg")
# classes = ["Jam", "Knife", "Bread", "Choco"]

# layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# colors = np.random.uniform(0, 255, size=(len(classes), 3))


def map_value(value, left_min, left_max, right_min, right_max):
    left_span = left_max - left_min
    right_span = right_max - right_min
    value_scaled = (value - left_min) / left_span  # convert the left range into a 0-1 range
    return np.nan_to_num(right_min + (value_scaled * right_span))  # convert the 0-1 range into a value in the right range


def calculate_coord(plane, Class_unity, Class_frame, norm_eye_3d_x, norm_eye_3d_y, norm_eye_3d_z):
    unity_width_half = Class_unity.width / 2
    unity_height_half = Class_unity.height / 2
    line = Line3D(Point3D(0, 0, 0), Point3D(norm_eye_3d_x, norm_eye_3d_y, norm_eye_3d_z))
    intersection = plane.intersection(line)[0]
    px_eye_2d_x = map_value(intersection.x, -unity_width_half, unity_width_half, 0, Class_frame.width)
    px_eye_2d_y = map_value(intersection.y, -unity_height_half, unity_height_half, Class_frame.height, 0)
    return px_eye_2d_x, px_eye_2d_y

def coord_3d_to_2d(Class_unity, Class_frame, data):
    unity_width_half = Class_unity.width/2
    unity_height_half = Class_unity.height/2
    unity_distance = Class_unity.distance
    plane = Plane(Point3D(-unity_width_half, unity_height_half, -unity_distance),
                  Point3D(-unity_width_half, -unity_height_half, -unity_distance),
                  Point3D(unity_width_half, unity_height_half, -unity_distance))
    px_eye_2d_x, px_eye_2d_y = calculate_coord(plane, Class_unity, Class_frame, data[1], data[2], data[3])
    data.append(px_eye_2d_x)
    data.append(px_eye_2d_y)
    return data


def process_frames(framedata):
#     global net, classes, layer_names, output_layers, colors
#     print(framedata)
    data = framedata[0]
    img = framedata[1]
    print(int(data["frame_number"]))
    if int(data['Fixation']) == 1: #wel een fixation
        data, img = labelimg(img, data)
        cv.circle(img, (int(data["px_eye_2d_x"]), int(data["px_eye_2d_y"])), 30, (255, 255, 255), 3)
    if int(data['Fixation']) == 0: #geen fixation
        cv.circle(img, (int(data["px_eye_2d_x"]), int(data["px_eye_2d_y"])), 30, (0, 0, 255), 3)
    return [data, img]

def labelimg(img, data):
    height, width, channels = img.shape
    
    net = cv.dnn.readNet("TinyWeightsZarinV7.weights", "TinyConfigZarin.cfg")
    classes = ["Jam", "Knife", "Bread", "Choco"]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    # Detecting objects
    blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids, confidences, boxes = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            print(confidence)
            if confidence > 0.3:
                # Object detected
                print("%s detected!"%class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.putText(img, label, (x, y + 30), font, 2, color, 2)

            if int(data["px_eye_2d_x"]) in range(x, x + w) and int(data["px_eye_2d_y"]) in range(y, y + h):
#                 print("looking at: %s        " % (label), end="\r")
                data["objectId"] = class_ids[i]
    
    return data, img



class Class_device:
    path = ''
    #H264
    input_path_h264 = ''
    input_file_h264 = ''
    #MP4
    output_path_mp4 = ''
    output_file_mp4 = ''
    #JSON
    output_path_json = ''
    output_file_json = ''
    #PKL
    output_path_pkl = ''
    output_file_pkl = ''


class Class_unity:
    width = 0.0
    height = 0
    distance = 0.0
    
class Class_frame:
    width = 0
    height = 0
    rate = 0.0
    total = 0
    number = 0
#     timestamp = 0.0 # seconds
    
class Class_video:
    window_name = ''
    input_file = ''
    output_file = ''
    nr_of_frames = 0
