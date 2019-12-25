import pandas as pd
import numpy as np

test = pd.read_csv("submission.csv")
truth = pd.read_csv("submission_org.csv")

count = 0
test = test['SalePrice'].values
truth = truth['SalePrice'].values
test = [round(x, 6) for x in test]
truth = [round(x,6) for x in truth]

for i in range(len(test)):
	if test[i] != truth[i]:
		count += 1
print(count)
