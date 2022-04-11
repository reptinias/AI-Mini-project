from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import cv2
import os
from imutils import paths
import numpy as np

# resize the image to a fixed size to 32 x 32 pixels, then flatten the image into
# a list of raw pixel intensities
# output feature vector will be 32 x 32 x 3 = 3072
def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	# normalizing the histogram
	cv2.normalize(hist, hist)
	# return the flattened histogram as the feature vector that is 8 x 8 x 8 = 512
	return hist.flatten()

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images("dataset"))

# initialize the raw pixel intensities matrix, the features matrix and labels list
rawImages = []
features = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
	# extract raw pixel intensity "features", followed by a color
	# histogram to characterize the color distribution of the pixels in the image
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
	# update the raw images, features, and labels matricies respectively
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)
	# show an update every 10 images
	if i > 0 and i % 10 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

# partition the data into training and testing splits, using 75% of the data for training and the remaining 25% for testing
(trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)


# train and evaluate a k-NN classifer on the raw pixel intensities
print("[INFO] evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=5,
	n_jobs=-1)

model.fit(trainRI, trainRL)
# findes the mean accuracy of the test data and label
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

image = np.array(cv2.imread("input.jpg"))
pixel = image_to_feature_vector(image)
histo = extract_color_histogram(image)

print(model.predict(pixel))

# train and evaluate a k-NN classifer on the histogram
# representations
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=5,
	n_jobs=-1)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

print(model.predict(histo))