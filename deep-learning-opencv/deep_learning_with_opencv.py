# USAGE
# python deep_learning_with_opencv.py --image images/jemma.png --prototxt bvlc_googlenet.prototxt --model bvlc_googlenet.caffemodel --labels synset_words.txt

# import the necessary packages
import numpy as np
import time
import cv2

image = cv2.imread('images/eagle.png')

rows = open('synset_words.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('bvlc_googlenet.prototxt', 'bvlc_googlenet.caffemodel')

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

cv2.imshow("Image", image)
cv2.waitKey(0)