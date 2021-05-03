# USAGE
# time python train_svr_grid.py

# import the necessary packages
from submodules import config
from sklearn.model_selection import RepeatedKFold # k equally size pieces, train on n-1 folds, evaluating the model on different splits
from sklearn.model_selection import GridSearchCV # implementation of gridsearch algorithm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR # support vector regression model
from sklearn.model_selection import train_test_split
import pandas as pd

# load the dataset, separate the features and labels, and perform a
# training and testing split using 85% of the data for training and
# 15% for evaluation
print("[INFO] loading data...")
dataset = pd.read_csv(config.CSV_PATH, names=config.COLS)
dataX = dataset[dataset.columns[:-1]]
dataY = dataset[dataset.columns[-1]]
(trainX, testX, trainY, testY) = train_test_split(dataX,
	dataY, random_state=3, test_size=0.15)

# standardize the feature values by computing the mean, subtracting
# the mean from the data points, and then dividing by the standard
# deviation
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)

# initialize model and define the space of the hyperparameters to
# perform the grid-search over
model = SVR()
# these are hyperparameters
# kernel project into higher dimensions
kernel = ["linear", "rbf", "sigmoid", "poly"]
tolerance = [1e-3, 1e-4, 1e-5, 1e-6]
# strictness, large value the harder the classifer,
# too large may overfit the data
# too low creates a soft classifier
C = [1, 1.5, 2, 2.5, 3]
# examine every combination of hyperparameters
grid = dict(kernel=kernel, tol=tolerance, C=C)

# initialize a cross-validation fold and perform a grid-search to
# tune the hyperparameters
print("[INFO] grid searching over the hyperparameters...")
# 10 splits and repeat 3 times
cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# hyperparameters going to search over, # of jobs is number of cores sckit learn will utilize
# it can run in parallel to speed up based on processor
# -1 = every core on all processors on the machine
gridSearch = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1,
	# smaller the MSE, the better we are at making predictions
	cv=cvFold, scoring="neg_mean_squared_error")
# exhaustively train every single
searchResults = gridSearch.fit(trainX, trainY)

# extract the best model and evaluate it
print("[INFO] evaluating...")
bestModel = searchResults.best_estimator_
print("R2: {:.2f}".format(bestModel.score(testX, testY)))

# score of .56 in 4m3s runtime to train the model
# can we boost this exhaustive search?