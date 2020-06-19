# Opmerkingen:
# Dit script kan enkel werken als je webcam nog niet in gebruik is door een andere applicatie


import cv2
import numpy as np

# Yolo model inladen en klaarzetten voor gebruik
net = cv2.dnn.readNet("weights/TinyWeightsV4.weights", "configs/TinyConfig.cfg")  # weight en configuration file ophalen
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# Namen van de classes definiëren (volgorde is zeer belangrijk)
classes = ["Jam", "Knife", "Bread", "Choco"]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Webcam properties instellen
cap = cv2.VideoCapture(0)  # 0 is je standaard webcam
cap.set(3, 640)  # x breedte van webcam instellen
cap.set(4, 480)  # y hoogte van webcam instellen
# cap.set(10, 150) #brightness van webcam instellen


def drawboxes(img):
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

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


while True:
    # Image van webcam inlezen
    success, img = cap.read()
    if not success:
        break

    # object detection uitvoeren
    img = drawboxes(img)

    # image weergeven
    cv2.imshow("frame", img)

    # zorgen dat de image zichtbaar blijft
    # druk op  de 'q' knop om de script te stoppen
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
