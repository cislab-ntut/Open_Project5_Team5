import pandas as pd
import numpy as np
from phe import paillier
public_key, private_key = paillier.generate_paillier_keypair()

def predict(weight, test):
	pre = list()
	for i in range(len(test)):
		pre.append(dot(test[i], weight))
	return pre
def dot(data, w):
	global public_key
	val = public_key.encrypt(0)
	for i in range(len(data)):
		val = val + (data[i]*w[i])
	#print(val)
	return val
def normalize(dataset, max_val, min_val):
	lst = list()
	for i in range(len(dataset)):
		val = (dataset[i] - min_val) / (max_val - min_val)
		lst.append(val)
	return lst

def loss(dataset, ans, theta):
	global private_key
	p = len(dataset[0])
	yp = list()
	for i in range(len(dataset)):
		yp.append(dot(dataset[i],theta))
	yp = np.array(yp)
	#print(yp[0])
	err = list()
	for i in range(len(yp)):
		err.append(private_key.decrypt(yp[i] - ans[i]))
	#print(err[0])
	gradient = list()
	for i in range(p):
		gradient.append(dot(dataset[:][i], err))
	gradient = np.array(gradient)
	#print(gradient[0])
	return gradient
	

def linear_regression(n_iterations, learning_rate, dataset, ans, theta):
	global public_key, private_key
	for iteration in range(n_iterations):
		if iteration % 500 == 0:
			print(iteration)
		gradients = loss(dataset, ans, theta)
		for i in range(len(theta)):
			theta[i] = private_key.decrypt(public_key.encrypt(theta[i]) - (gradients[i] * learning_rate))
		#print(theta[0])
	return theta

def encode(input_data):
	global public_key
	encrypted = [public_key.encrypt(x) for x in input_data]
	#print(encrypted)
	return encrypted
def decode(enc):
	global private_key
	dec = [private_key.decrypt(x) for x in enc]
	return dec

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

print("start encryption")
#encryption
new_result = encode(new_result)
print("after")
train = [encode(x) for x in train]
print("after")
test = [encode(x) for x in test]
print("after")

weight = linear_regression(100, 0.001, train, new_result, theta)

pre = predict(weight, test)
pre = [private_key.decrypt(x) for x in pre]
for i in range(len(pre)):
	pre[i] = pre[i]*(p_max - p_min) + p_min
pre = pd.DataFrame(pre, columns=['SalePrice'], index=index)
output = pd.concat([test_id,pre], axis=1)
output.to_csv('submission.csv', index=0)
#print(pre)