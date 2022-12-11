import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Splitting data into 70:30 train:test ratio
df_X=df.iloc[:,:4]
df_Y=df.iloc[:,4]
X_train,X_test,Y_train,Y_test=train_test_split(df_X,df_Y,test_size=0.3,random_state=33)


# Changing the index of the records to sequential
X_train.index=range(len(X_train))
Y_train.index=range(len(X_train))
X_test.index=range(len(X_test))
Y_test.index=range(len(Y_test))


# Function to return the list of distances of the test records from train records
def distNeighbours(X_train,Y_train,X_test,K):
    distance=[]
    for i in range(len(X_train)):
        eDistance=0
        for j in range(len(X_train.columns)):   
                eDistance+=round(np.sqrt(pow((X_train.iloc[i,j]-X_test[j]),2)),2)
        distance.append((eDistance,i,Y_train.iloc[i]))
        distance=sorted(distance, key=lambda x: x[0])[:K]
    return distance

# Predict the output of the numeric variables based on K nearest neighbours
# Output is the mean of the K nearest neighbours
def predictOutputNumeric(X_train,Y_train,X_test,K):
    neighbours=[]
    responses=[]
    for i in range(len(X_test)):
        neighbours.append(distNeighbours(X_train,Y_train,X_test.iloc[i,:],K))
    for i in neighbours:
        mean=0
        for j in i:
            mean+=j[-1]
        mean=mean/K
        responses.append(mean)
    return responses

# Accuarcy of the numerical predictions
def getAccuracyNumeric(actual,predicted):
    error=0
    for i in range(len(predicted)):
        error+=pow((actual[i]-predicted[i]),2)
    error=error/len(predicted)-1
    return 100-error

model=KNeighborsRegressor(n_neighbors=3,p=2)
model.fit(X_train,Y_train)


print('Accuracy from the model {:^0.2f}'.
      format(metrics.mean_squared_error(Y_test,model.predict(X_test))*100))



     

#Accuracy from the model 99.88


# Check whether both the outputs are same or not
# They are not same - Need to find why?
output==model.predict(X_test)

#######################################################


# Importing a regression dataset from sklearn 

from sklearn.datasets import fetch_california_housing
X, y = fetch_california_housing(return_X_y=True)

#Setting the train and test split percentage 
train_split_percent = 0.7


# Splitting the dataset into test and train datasets

size = X.shape[0]
X_train = X[:int(train_split_percent * size),:]
X_test = X[int(train_split_percent * size):,:]
y_train = y[:int(train_split_percent * size)]
y_test = y[int(train_split_percent * size):]


#Standardizing the X_train and X_test daatsets
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu ) / sigma

#We use the same mean and SD as the one of X_train as we dont know the mean of X_test
X_test = (X_test - mu ) / sigma

#Standardizing the y_train data
mu_y = np.mean(y_train, 0)
sigma_y = np.std(y_train, 0, ddof = 0)

y_train = (y_train - mu_y ) / sigma_y


#Changing the shape of the target varibale for easy computation 

y_train = y_train.reshape(len(y_train),1)
y_test = y_test.reshape(len(y_test),1)
y_pred = np.zeros(y_test.shape)
y_train.shape, y_test.shape,y_pred.shape


#Naive Implementation

n_neigh = 10
for row in range(len(X_test)):
    euclidian_distance = np.sqrt(np.sum((X_train - X_test[row])**2, axis = 1 ))
    y_pred[row] = y_train[np.argsort(euclidian_distance, axis = 0)[:n_neigh]].mean()* sigma_y + mu_y
    
#Finding the root mean squared error 

RMSE = np.sqrt(np.mean((y_test - y_pred)**2))
print(RMSE)


#Vectorised implementation of KNN, using numpy broadcasting

# We are setting a range of K values and calculating the RMSE for each of them. This way we can chose the optimal K value
k_list = [x for x in range(1,50,1)]

# Calculating the distance matrix using numpy broadcasting technique 
distance = np.sqrt(((X_train[:, :, None] - X_test[:, :, None].T) ** 2).sum(1))

#Sorting each data points of the distance matrix to reduce computational effort 
sorted_distance = np.argsort(distance, axis = 0)

#The knn function takes in the sorted distance and returns the RMSE of the 
def knn(X_train,X_test,y_train,y_test,sorted_distance,k):
    y_pred = np.zeros(y_test.shape)
    for row in range(len(X_test)):
        
        #Transforming the y_train values to adjust the scale. 
        y_pred[row] = y_train[sorted_distance[:,row][:k]].mean() * sigma_y + mu_y

    RMSE = np.sqrt(np.mean((y_test - y_pred)**2))
    return RMSE

#Storing the RMSE values in a list for each k value 
rmse_list = []
for i in k_list:
    rmse_list.append(knn(X_train,X_test,y_train,y_test,sorted_distance,i))
    
    
#Finding the optimal K value
min_rmse_k_value = k_list[rmse_list.index(min(rmse_list))]

#Finding the lowest possible RMSE
optimal_RMSE = knn(X_train,X_test,y_train,y_test,sorted_distance,min_rmse_k_value)
optimal_RMSE

###########################################

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y=True)
train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.25)
print(train_data.shape)
print(test_data.shape)
print(train_target.shape)
print(test_target.shape)


#After teaching the machine, we need to get the error of our prediction. For that, I'll use metrics library.

from sklearn import metrics

#Now, we will proceed with the first approach.
#1. Without any external library

#For calculating the Euclidean distance, I'll use external library from scipy. But for teaching the machine, I won't be using.

from scipy.spatial import distance

class kNNRegressor:

    def __init__(self, neighbors):
        self.train_data = []
        self.train_target = []
        self.neighbors = neighbors

    def fit(self, train_data, train_target):
        self.train_data = train_data
        self.train_target = train_target

    def predict(self, test_data):
        test_predicted = []
        for i in range(len(test_data)):
            distances = []
            for j in range(len(self.train_data)):
                distances.append((distance.euclidean(self.train_data[j], test_data[i]), self.train_target[j]))
            distances.sort(key = lambda x: x[0])
            mean = 0
            for j in range(self.neighbors):
                mean += distances[j][1]
            mean /= self.neighbors
            test_predicted.append(mean)
        return test_predicted

#Now, I've the class. I'll proceed with our procedure.

neigh = kNNRegressor(5)
neigh.fit(train_data, train_target)
test_predicted = neigh.predict(test_data)
metrics.mean_absolute_error(test_target, test_predicted)


#Now, I'll do the same thing with sklearn's KNeighborsRegressor.
#2. sklearn's KNeighborsRegressor

from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=5)
neigh.fit(train_data, train_target)
test_predicted = neigh.predict(test_data)
metrics.mean_absolute_error(test_target, test_predicted)
##########
