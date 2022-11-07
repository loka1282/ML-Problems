 

import numpy as np
from sklearn.linear_model import LinearRegression
P=np.array([15,12,8,8,7,7,7,6,5,3]).reshape(-1,1)
H=np.array([10,25,17,11,13,17,20,13,9,15])
model = LinearRegression()
model.fit(P,H)
model = LinearRegression().fit(P,H.reshape(-1,1))
rsq = model.score(P,H)
print(f"coefficient of determination: {rsq}")
print(f"slope: {model.coef_}")
