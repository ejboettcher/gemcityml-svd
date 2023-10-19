import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sunspot_data = pd.read_csv("data/Sunspots.csv")
sunspot_data['time'] = pd.to_datetime(sunspot_data['Date'], format='%Y-%m-%d')
nname = 'Monthly Mean Total Sunspot Number'

# No Features


# Features Window of 30.

window = 30
dataset = np.ones((len(sunspot_data)-window, window))

for ii in range(len(dataset)):
    dataset[ii,:] = sunspot_data.loc[ii:ii+window-1, nname].to_numpy().T

split_num = 3000
A = dataset
A = np.column_stack([np.ones(A.shape[0]), A])
X_train = A[:split_num,:]

y_train = sunspot_data.loc[window:split_num+window-1, nname].to_numpy()

X_test = A[split_num:,:]
y_test = sunspot_data.loc[split_num+window:, nname].to_numpy()

print(X_test.shape, y_test.shape, X_train.shape, y_train.shape)
# calculate the economy SVD for the data matrix A
U,S,Vt = np.linalg.svd(X_train, full_matrices=False)

# solve Ax = b for the best possible approximate solution in terms of least squares
x_hat = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y_train

# perform train and test inference
y_pred = X_train @ x_hat
test_predictions = X_test @ x_hat

train_data = pd.DataFrame({'time':sunspot_data.loc[window:split_num+window-1,'time'],
                           'train':y_train, 
                           'y_pred':y_pred} )
train_data.plot(x='time', y=['train', 'y_pred'])
plt.show()

test_data =  pd.DataFrame({'time':sunspot_data.loc[split_num+window:,'time'],
                           'test':y_test, 
                           'y_pred':test_predictions} )
test_data.plot(x='time', y=['test', 'y_pred'])
plt.show()

# compute train and test MSE
train_mse = np.mean(np.sqrt((y_pred - y_train)**2))
test_mse = np.mean(np.sqrt((test_predictions - y_test)**2))

print("Train Mean Squared Error:", train_mse)
print("Test Mean Squared Error:", test_mse)