# -------------- Imports --------------

import os

import cv2 as cv
import numpy as np


# -------------- Prepare object detection --------------
classes = ["Jam", "Knife", "Bread", "Choco"]
net = cv.dnn.readNet("Weights/TinyWeightsZarinV4.weights", "Configs/TinyConfigZarin.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# -------------- Shared definitions --------------

# This definition uses the input file and the preprocessed data to create a mp4 file
# Frame by frame, the video is processed
# If there is a fixation in the frame, a red circle will be drawn and object will be identified
# Else, a white circle will be drawn

def create_video(data, video, frame):
    if os.path.exists(video.output_file):
        os.remove(video.output_file)

    data["objectId"] = np.nan
    data["object"] = np.nan

    # Open the video input file
    cap = cv.VideoCapture(video.input_file)

    # Creates a writer that will write frame by frame to a mp4 file
    codec = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv.VideoWriter(video.output_file, codec, int(frame.rate), (int(frame.width), int(frame.height)))  # MP4

    current_frame = 0

    try:
        while cap.isOpened():

            # Tries to read a frame from the video and move the offset 1 forward
            ret, img = cap.read()

            # Checks if there is successfully extracted a frame
            if ret is True:
                # Checks if this frame contains a fixation

                # If true, draw a red circle on its position
                # AND run the algorithm to define what object is fixated on
                if data['fixation'][current_frame] == 1:
                    img, serie = labelimg(img, data.iloc[[current_frame]])
                    data.iloc[[current_frame]] = serie
                    cv.circle(img, (data["px_eye_2d_x"][current_frame], data["px_eye_2d_y"][current_frame]), 30,
                              (255, 255, 255), 3)

                # If false, draw a white circle on its position
                if data['fixation'][current_frame] == 0:  # geen fixation
                    cv.circle(img, (data["px_eye_2d_x"][current_frame], data["px_eye_2d_y"][current_frame]), 30,
                              (0, 0, 255), 3)

                # The frame is added to the mp4 file
                out.write(img)
                current_frame = current_frame + 1

            # If no frame was extracted, break the loop
            else:
                break

        # Releases all the opencv objects
        cap.release()
        out.release()
        cv.destroyAllWindows()

    except Exception as e:
        print('error')
        print(e)
        cap.release()
        out.release()
        cv.destroyAllWindows()

    # Finally the name of the identifiedobject is added to the dataframe
    data['object'] = data.apply(lambda x: name_object(x), axis=1)
    return data

# This definition is the algorithm that tries to identify the object in the frame with fixations

def labelimg(img, data):

    height, width, channels = img.shape

    # Detecting objects
    blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids, confidences, boxes = [], [], []
    # confidences = []
    # boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                #                 print(class_id)
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
    #     print(indexes)
    font = cv.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.putText(img, label, (x, y + 30), font, 2, color, 2)

            if int(data["px_eye_2d_x"]) in range(x, x + w) and int(data["px_eye_2d_y"]) in range(y, y + h):
                print("looking at: %s        " % (label), end="\r")
                data["objectId"] = class_ids[i]
                data["object"] = classes[int(class_ids[i])]

    return img, data

# This definition checks if there is wrong datapoint by looking at previous and future datapoints
# It corrects this mistakes

def interpolate(data, max_pointRange):
    classes = ["Jam", "Knife", "Bread", "Choco"]

    previous_point = {"index": 0, "object": None}

    for index, row in data.iterrows():
        if previous_point['object'] == None and row['object'] != None:
            previous_point['object'] = row['object']
            previous_point['index'] = row['frame_number']

        if row['object'] != None:
            if previous_point['object'] == row['object']:
                if int(row['frame_number'] - previous_point['index']) in range(2, max_pointRange+1):
                    for frame in range(int(previous_point['index'] +1) , int(row['frame_number'])):
                        data.at[frame, "object"] = row['object']
                        data.at[frame, "objectId"] = classes.index(row['object'])

                previous_point['object'] = row['object']
                previous_point['index'] = row['frame_number']
    return data

# This definition adds the right name the the identified object (in the dataframe)

def name_object(row):
    classes = ["Jam", "Knife", "Bread", "Choco"]

    if row['objectId'] in range(0, len(classes)):
        return classes[int(row['objectId'])]