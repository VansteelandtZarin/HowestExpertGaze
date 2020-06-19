import datetime

import cv2
import numpy as np
import glob
import random
import time

# Eerste model inladen en klaarzetten voor gebruik
net = cv2.dnn.readNet("weights/TinyWeightsV4.weights", "configs/TinyConfig.cfg")  # weight en configuration file ophalen
classes = ["Jam", "Knife", "Bread", "Choco"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Tweede model inladen en klaarzetten voor gebruik
net2 = cv2.dnn.readNet("weights/TinyWeightsV1.weights", "configs/TinyConfig.cfg")  # weight en configuration file ophalen
layer_names2 = net2.getLayerNames()
output_layers2 = [layer_names2[i[0] - 1] for i in net2.getUnconnectedOutLayers()]

# Webcam properties instellen
cap = cv2.VideoCapture(0)  # 0 is je standaard webcam
cap.set(3, 640)  # x breedte van webcam instellen
cap.set(4, 480)  # y hoogte van webcam instellen
# cap.set(10, 150) #brightness van webcam instellen


def drawboxes(img, model, layer):
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    model.setInput(blob)
    outs = model.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # boxes tekenen
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[int(class_ids[i])])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    return img


# functie om images naast elkaar weer te geven
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


while True:
    # Loading image
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)  # webcam spiegeling uitschakelen

    # copy maken van image en elke met een ander model object detection uitvoeren
    img2 = img
    img = drawboxes(img, net, output_layers)
    img2 = drawboxes(img2, net2, output_layers2)

    # alle twee de images naast elkaar plakken
    concat = stackImages(1, [img, img2])

    # image weergeven
    cv2.imshow("frame", concat)
    if img.all() != img2.all():
        print('Different')

    # zorgen dat de image zichtbaar blijft
    # druk op  de 'q' knop om de script te stoppen
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
