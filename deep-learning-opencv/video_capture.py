import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import time
import imutils

rows = open('synset_words.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('bvlc_googlenet.prototxt', 'bvlc_googlenet.caffemodel')

print("[INFO] starting video stream...")

cap=cv2.VideoCapture(0)
print(cap.isOpened())
time.sleep(2.0)
fps = FPS().start()

while cap.isOpened():
    print("[INFO] inside loop")
    ret, frame=cap.read()
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (224, 224), 127.5)
    net.setInput(blob)
    start = time.time()
    preds = net.forward()
    end = time.time()
    print("[INFO] classification took {:.5} seconds".format(end - start))

    idxs = np.argsort(preds[0])[::-1][:5]

    for (i, idx) in enumerate(idxs):
        # draw the top prediction on the input image
        if i == 0:
            text = "Label: {}, {:.2f}%".format(classes[idx],
                                               preds[0][idx] * 100)
            cv2.putText(frame, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
                                                                classes[idx], preds[0][idx]))

    if ret==True:
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
        cv2.waitKey(0)
        fps.update()
    else:
        break
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()