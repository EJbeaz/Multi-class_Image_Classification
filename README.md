# Multi-class_Image_Classification
1.0 Introduction

The purpose of this investigation is to evaluate and compare the efficiency of a multitude of machine learning models in their ability to predict and accurately 
classify different types of images into 3 separate ‘animal’ classes. In the specific case of this report, 5 trained machine learning models will 
be used as multiclass classifiers.The basic assumption of a multi-class classification task is that each data point will belong to only one of the 
‘N’ (This task=3) classes. When training a machine learning model, it is both advantageous and expected that a data set can provide test data large enough to 
provide sufficient examples belonging to each class (This task=900) so that the machine learning model can identify and learn the underlying hidden patterns for
accurate classification. The dataset used in this report, sufficiently met this criterion, and therefore was selected for this particular task. In this report,
90% of the data was used as training data, and the subsequent 10% for test data, of which is true for each of the five different models.

2.0 Dataset summary

The dataset used in this investigation was sourced from Kaggle, and comprised of 3000 instances, or images, numerically distributed equally between 3 feature 
classes that each defined a given animal species. The species under investigation were Cats, Dogs and Pandas. The dataset was 400 megabytes in size, this was 
not just due to the quantity of, but also size of each image, and it should therefore be noted that in the case of each machine learning model used, the images 
were first resized during the data pre-processing stage, in order to decrease the time taken to run each model.

The data can be found as per the reference link below:

https://www.kaggle.com/datasets/ashishsaxena2209/animal-image-datasetdog-cat-and- panda

3.0 Machine learning models

For the purpose of fluidity in reading this report, coding description has been integrated into the jupyter notebook files, as such making it easier to follow 
what is being implemented and at what stage by the reader. In short, all the models followed the same basic structure which commenced with the loading of 
necessary libraries and the datafile, as well as the pre-processing the data. Then upon completion, the hyperparameters were tuned to find the optimal values. 
The different hyperparameters of each model being optimised in this report are defined in table 1. Hyperparameter tuning was utilised in this report alongside a
10-fold cross-validation. By using k-fold cross-validation to evaluate the performance of the model with different sets of hyperparameters, we can find the set
of hyperparameters that leads to the best performance on average across all folds. These optimised hyperparameters were then used alongside the training data to
build an efficient image classification model.In this report, two methods were applied to employ the 10-fold cross validation which was dependent on the particular 
training model of use. ‘KFold’ is one of the methods utilised to split the data into k equally sized folds, where k=10 in this instance. One-fold was used for 
the validation set and the remaining 9 are used as the training set which allows for an unbiased estimate of a model’s performance on new data. 
‘StratifiedKfold’ is variation of Kfold and served as the other method of cross-validation in this report.

Table1: Hyperparameters tuned for each of the image classification models.

Classification Model:                        Hyperparameters: 
Support Vector machine (SVM)                 - C (0.1 or 10),Gamma (scale or auto)
K-nearest Neighbor (K-NN)                    - N_neighbors (3, 5 or 7), weights (uniform or distance)
Decision tree (DT)                           - Max_length (length (1,10))
Convolutional Neural network (CNN)           - Batch size (16, 32 or 64),Epochs (10, 20 or 30), Optimizer ('adam' or 'sgd')
Fully connected neural network (FCNN)        - Batch size (16, 32 or 64),Epochs (10, 20 or 30), Optimizer ('adam' or 'sgd')

Table 2: Scoring metrics used to evaluate and compare the image classification models (Chicco and Jurman, 2020).

Scoring Metrics:    Purpose:
Confusion Matrix    - Summarises the performance of a classification model by visualising the number of correct versus incorrect predictions for each class.
Accuracy            - A measure of the proportion of correct predictions made by a model, expressed as a percentage which is used to evaluate the overall performance of a model.
F1-Score            - A weighted average of the precision and recall, which are two scoring metrics that quantify the quality of a model’s image classification performance.


With reference to Table 2, Three scoring metrics were then computed to evaluate and compare the performance of each of the classification models. 
To further add to the depth of my evaluation, rather than looking simply at the overall accuracy of my model I used the confusion matrix to calculate 
the individual class accuracy of each image classification model (viewable in full report to be uploaded).
       



