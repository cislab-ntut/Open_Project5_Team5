import pandas as pd
import numpy as np

def predict(weight, test):
	pre = list()
	for i in range(len(test)):
		pre.append(dot(test[i], weight))
	return pre
def dot(data, w):
    val = 0;
    for i in range(len(data)):
        val += data[i]*w[i]
    #print(val)
    return val
def normalize(dataset, max_val, min_val):
	lst = list()
	for i in range(len(dataset)):
		val = (dataset[i] - min_val) / (max_val - min_val)
		lst.append(val)
	return lst

def loss(dataset, ans, theta):
	m, p = dataset.shape
	yp = list()
	for i in range(len(dataset)):
		yp.append(dot(dataset[i],theta))
	yp = np.array(yp)
	#print(yp[0])
	err = yp - ans
	#print(err[0])
	gradient = list()
	for i in range(p):
		gradient.append(dot(dataset[:][i], err))
	gradient = np.array(gradient)
	#print(gradient[0])
	return gradient
	

def linear_regression(n_iterations, learning_rate, dataset, ans, theta):
	
	for iteration in range(n_iterations):
		if iteration % 500 == 0:
			print(iteration)
		gradients = loss(dataset, ans, theta)
		theta = theta - learning_rate * gradients
		#print(theta[0])
	return theta

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train_id = train['Id']
result = train['SalePrice']
test_id = test['Id']
train = train.drop("Id",axis=1).drop("SalePrice",axis=1).drop("Alley",axis=1).drop("Fence",axis=1).drop("MiscFeature",axis=1).drop("PoolQC",axis=1)
train = train.fillna(0)

test = test.drop("Id",axis=1).drop("Alley",axis=1).drop("Fence",axis=1).drop("MiscFeature",axis=1).drop("PoolQC",axis=1)
test = test.fillna(0)
index = test.index

deal = pd.concat([train, test],axis=0)

for col in deal:
    if (deal[col].dtypes == 'object'):
        deal[col] = pd.factorize(deal[col], sort= True)[0]
deal = deal.values
max_index = np.argmax(deal,axis=0)
max_val = list()
for i in range(len(max_index)):
	max_val.append(deal[max_index[i]][i])
min_index = np.argmin(deal,axis=0)
min_val = list()
for i in range(len(min_index)):
	min_val.append(deal[min_index[i]][i])
max_val = np.array(max_val)
min_val = np.array(min_val)
deal = np.array(normalize(deal, max_val, min_val))

train = deal[:1460]
test = deal[1460:]

result = result.values

p_max = result.max()
p_min = result.min()
new_result = np.array(normalize(result, p_max, p_min))

#theta = np.random.randn(train.shape[1])

theta = list()
for i in range(train.shape[1]):
	theta.append(0.5)
theta = np.array(theta)

weight = linear_regression(100, 0.001, train, new_result, theta)

pre = predict(weight, test)
for i in range(len(pre)):
	pre[i] = pre[i]*(p_max - p_min) + p_min
pre = pd.DataFrame(pre, columns=['SalePrice'], index=index)
output = pd.concat([test_id,pre], axis=1)
output.to_csv('submission_org.csv', index=0)
#print(pre)