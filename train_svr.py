# USAGE
# time python train_svr.py


# No hyperparameter tuning alg

# import the necessary packages
from submodules import config
from sklearn.preprocessing import StandardScaler # standard score or z-score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR # support vector regression model
import pandas as pd # load and use CSV

# load the dataset, separate the features and labels, and perform a
# training and testing split using 85% of the data for training and
# 15% for evaluation
print("[INFO] loading data...")
# loads from disk
dataset = pd.read_csv(config.CSV_PATH, names=config.COLS)
dataX = dataset[dataset.columns[:-1]] # everything but last col (age) to predict
dataY = dataset[dataset.columns[-1]]

# split the train/test data 85% and 15% for eval
(trainX, testX, trainY, testY) = train_test_split(dataX,
	dataY, random_state=3, test_size=0.15)

# standardize the feature values by computing the mean, subtracting
# the mean from the data points, and then dividing by the standard
# deviation
scaler = StandardScaler()
# fit_transform = compute mean and std. dev. on the column, Transform does the z-score calc.
trainX = scaler.fit_transform(trainX)
# transform = keep test separates from training, never call transform on test.
testX = scaler.transform(testX)

# train the model with *no* hyperparameter tuning
print("[INFO] training our support vector regression model")
model = SVR() # no hyperparameter tuning involved
model.fit(trainX, trainY) # fit on the training data

# evaluate our model using R^2-score (0.0 is the worst, 1.0 is the best value)
print("[INFO] evaluating...")
print("R2: {:.2f}".format(model.score(testX, testY)))

# score of .55 in 2.37s runtime to train the model
# can we boost it?