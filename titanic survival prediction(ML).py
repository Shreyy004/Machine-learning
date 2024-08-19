import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


#supervised learning-classification model
#data collection
df=pd.read_csv('heart.csv')
print(df)

#data preprocessing
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
#convert all the string values to numbers
df['Sex']=le.fit_transform(df['Sex'])
df['Embarked']=le.fit_transform(df['ChestPainType']) #based on alphabetical oredr it orders the data
#replacing null
df['Age']=df['Age'].replace(np.nan,0)
df['Embarked']=df['Embarked'].replace(np.nan,0)



#print(df)

x=df.drop(columns=['Survived','Name','PassengerId','Ticket','Cabin']) #x-input -all the features are input except the last col


y=df['Survived'] #y-output only the survived col is output
print("XXXX",x)
print("YYYY",y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=12)

#shape gives the number of rows and cols
print("DF",df.shape)
print("x_train",x_train.shape)
print("x_test",x_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)


#x_train-80% of input data 
#y_train-80% of output data                                                
#x_test-20% of input data 
#y_test-20% of output data




#model training
#choosing the alg
from sklearn.naive_bayes import GaussianNB
NB=GaussianNB()

NB.fit(x_train,y_train)

#model evaluation
y_pred=NB.predict(x_test) #20% input

print("y_pred",y_pred)
print("y_test",y_test)

from sklearn.metrics import accuracy_score #compared the models predicted output with the actual output
print("Accuracy is",accuracy_score(y_test,y_pred))


#model prediction

testPrediction=NB.predict([[19,1,4,120,166,0,1,138,0,0,2]])
if testPrediction==1:
    print("Not survived")

else:
    print("Survived")

    
