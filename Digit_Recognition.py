import warnings
warnings.filterwarnings(action="ignore")

import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()
print("digits : ", digits.keys())
print("digits.target---- : ", digits.target)
images_and_labels = list(zip(digits.images, digits.target))
print("len(images_and_labels) ",len(images_and_labels))

for index, [image,label] in enumerate(images_and_labels[:5]):
    print("index : ", index, "image : \n", image, "  label : ", label)
    plt.subplot(2, 5, index+1)
    plt.axis('on')
    plt.imshow(image , cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i ' % label)

plt.show()

n_samples = len(digits.images)
print("n_samples : ", n_samples)
imageData = digits.images.reshape((n_samples, -1))
print("After Reshaped: len(imageData[0]) : ", len(imageData[0]))

classifier = svm.SVC(gamma=0.001)
classifier.fit(imageData[: n_samples//2],digits.target[:n_samples//2])
expected = digits.target[n_samples // 2:]

predicted = classifier.predict(imageData[n_samples//2:])
images_and_predictions = list(zip(digits.images[n_samples//2:], predicted))
for index, [image, prediction] in enumerate(images_and_predictions[:5]):
    plt.subplot(2, 5, index + 6)
    plt.axis('on')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

print("Original Values: ", digits.target[n_samples//2:(n_samples//2)+5])
print("Accuracy: ", accuracy_score(expected, predicted)*100)


from scipy.misc import imread,imresize, bytescale
img = imread("Eight.jpeg")
img = imresize(img,(8,8))

classifier = svm.SVC(gamma=0.001)
classifier.fit(imageData[:], digits.target[:])

img = img.astype(digits.images.dtype)
img = bytescale(img, high=16.0, low=0)

print(" img.shape : ", img.shape)
print("\n", img)

x_testData = []

for c in img:
    for r in c:
        x_testData.append(sum(r)/3.0)

print("x_testData : \n", x_testData)

print("len(x_testData) :", len(x_testData))

x_testData = [x_testData]
print("len(x_testData) : ", len(x_testData))

print("Machine Output = ",classifier.predict(x_testData))

print("Accuracy: ", accuracy_score(expected, predicted)*100)

plt.show()
