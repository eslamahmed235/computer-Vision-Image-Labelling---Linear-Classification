# Computer Vision characters and Latters -Labelling-Linear-Classification
## Dataset
You will use the Chars74K dataset of characters of the English alphabet
## 1.1 preprocessing 
we scikit-learn image processing tools to read and process the images
 as skimage. io.imread, skimage. color. rgb2gray, and skimage.resize.

## 1.2 Binary Classifiers
Train two logistic regression classifiers to distinguish grayscale images of the letters ’o’ and ’q’, as well as of ’G’ and ’8’

## 1.3 Multiclass Classifier
Train a logistic regression classifier but this time to classify all grayscale images into multiple (4) classes: characters ’o’,’q’,’G’ and digit ’8’

## 1.4 evaluation
print the confusion matrix. Without using any scikit-learn function calculate the accuracy, the average recall, the average precision and the F1 score of your two classifiers based on the confusion matrix of the training data, as well as on the testing data. Print the ROC curves for each. And the accuracy <br/>


![Multiclass ROC curve](https://github.com/eslamahmed235/computer-Vision-Image-Labelling---Linear-Classification/blob/main/Data/Multiclass%20ROC%20curve.png)
