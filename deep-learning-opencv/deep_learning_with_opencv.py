from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import time
import cv2
import imutils

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
print("[INFO] after video stream...")
time.sleep(2.0)
fps = FPS().start()

rows = open('synset_words.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('bvlc_googlenet.prototxt', 'bvlc_googlenet.caffemodel')

print("[INFO] before loop...")
while True:
	print("[INFO] inside loop...")
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	print("[INFO] 1...")

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)
	print("[INFO] 2...")
	net.setInput(blob)
	start = time.time()
	preds = net.forward()
	end = time.time()
	print("[INFO] classification took {:.5} seconds".format(end - start))

	idxs = np.argsort(preds[0])[::-1][:5]

	for (i, idx) in enumerate(idxs):
		if i == 0:
			text = "Label: {}, {:.2f}%".format(classes[idx],
				preds[0][idx] * 100)
			cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 255), 2)

		print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
			classes[idx], preds[0][idx]))

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	fps.update()
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()