# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.


## Program:
```python
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: M.Chandru
RegisterNumber:  24900224

```
```python
import pandas as pd
data = pd.read_csv("Salary (1).csv")
print(data.head())
data.info() 
print(data.isnull().sum())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head())
x = data[["Position","Level"]]
y = data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
print(mse)
r2 = metrics.r2_score(y_test,y_pred)
print(r2)
dt.predict([[5,6]])
```
## Output:
![Screenshot 2024-11-28 173406](https://github.com/user-attachments/assets/00166c14-3882-4d93-b0da-090b4e3861af)

![Screenshot 2024-11-28 173412](https://github.com/user-attachments/assets/a704490c-0fef-44a5-b4c0-53c780bd60b5)

![Screenshot 2024-11-28 173419](https://github.com/user-attachments/assets/c959318a-ac08-44b9-81e0-6cba7a729223)

![Screenshot 2024-11-28 173426](https://github.com/user-attachments/assets/9a8a1f37-d68b-4bff-ac14-fcb03a76441a)

![Screenshot 2024-11-28 173429](https://github.com/user-attachments/assets/8372c8e1-820a-4548-b042-7d800657a356)

![Screenshot 2024-11-28 173434](https://github.com/user-attachments/assets/b99dbb62-afd5-4d57-ba63-4bc87f782a2d)

![Screenshot 2024-11-28 173454](https://github.com/user-attachments/assets/862abbb4-e2e0-42aa-bbd0-5b2bad83a8e1)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
