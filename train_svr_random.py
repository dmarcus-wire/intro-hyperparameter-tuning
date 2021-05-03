# USAGE
# time python train_svr_random.py

# import the necessary packages
from submodules import config
from sklearn.model_selection import RandomizedSearchCV # sample hyperparameters across a distrubition
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler # used for processing
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split # splitting training and testing
from scipy.stats import loguniform # distribution of tolerance values
import pandas as pd # loading csv from disk

# load the dataset, separate the features and labels, and perform a
# training and testing split using 85% of the data for training and
# 15% for evaluation
print("[INFO] loading data...")
# load from disk
dataset = pd.read_csv(config.CSV_PATH, names=config.COLS)
# grab features
dataX = dataset[dataset.columns[:-1]]
# grab the age (what we want to predict)
dataY = dataset[dataset.columns[-1]]
# split the data
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
kernel = ["linear", "rbf", "sigmoid", "poly"]
tolerance = loguniform(1e-6, 1e-3)
C = [1, 1.5, 2, 2.5, 3]
# dictionary based on the hyperparameters
grid = dict(kernel=kernel, tol=tolerance, C=C)

# initialize a cross-validation fold and perform a grid-search to
# tune the hyperparameters
print("[INFO] grid searching over the hyperparameters...")
# k fold on 10 splits 3 times across all cores on the processor
cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
randomSearch = RandomizedSearchCV(estimator=model, n_jobs=-1,
	cv=cvFold, param_distributions=grid,
	# minimize MSE
	scoring="neg_mean_squared_error")
searchResults = randomSearch.fit(trainX, trainY)

# extract the best model and evaluate it
print("[INFO] evaluating...")
# grab the best model and evualuate on test set
bestModel = searchResults.best_estimator_
print("R2: {:.2f}".format(bestModel.score(testX, testY)))

# score of .56 in 260.91s runtime to train the model
# was this worth the change?