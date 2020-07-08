"""
Cancer Detection Using Tensorflow :
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('DATA/cancer_classification.csv')
print(df.head())
print(df.columns)
print(df.info())
print(df.describe().T)
print(df.isnull().sum())
print('\n')

""" 
Exploratory Data Analysis :
"""

print(df.corr()['benign_0__mal_1'].sort_values(ascending=False))
df.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')
plt.show()

sns.heatmap(df.corr())
plt.show()

sns.set_style('darkgrid')
sns.countplot(x='benign_0__mal_1', data=df)
plt.show()

sns.boxplot(x='benign_0__mal_1', y='smoothness error', data=df)
plt.show()

sns.barplot(x='benign_0__mal_1', y='mean fractal dimension', data=df)
plt.show()

""" 
TRAIN AND SPLIT OF DATA :
"""

from sklearn.model_selection import train_test_split

X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=102)

""" 
SCALING OF DATA :
"""
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)

""" 
Creating A Classification Model As A Nerual Network :

we will be adding units same as our features of X_train shape
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

model = Sequential()

model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))


model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')


""" 
Training The Model : 
"""

model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),verbose=1,epochs=450)

mod_loss_1=pd.DataFrame(model.history.history)
mod_loss_1.plot()
plt.show()

""" 
The Above Plot has shown us the that validation data ' s loss has increased,
which we dont want for our model ..so let us try to analyse it by  adding a 
callback function - EarlyStop


EarlyStop Function Basically helps to prevent overfitting of our Data and tries to stop 
running of epochs , if it senses that the validation data loss has started to increase
eventually..

"""


from tensorflow.keras.callbacks import EarlyStopping


model1 = Sequential()

model1.add(Dense(30, activation='relu'))
model1.add(Dense(15, activation='relu'))


model1.add(Dense(1,activation='sigmoid'))


model1.compile(loss='binary_crossentropy',optimizer='adam')


early_stop=EarlyStopping(monitor='val_loss',mode='min',patience=25,verbose=1)


model1.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),epochs=450,
          verbose=1,callbacks=[early_stop])

""" 
What the above line basically does is that it will try to to minimise the loss ,
so it will monitor and keep track of 'val_loss',which is validation loss..

and also..it will try to minimise it so .. the mode parameter has been set to ' min '

and we have put the patience parameter to 25 .. it means that our epochs , even after callback
of early stop will keep running 25 times moore..just to ensure proper fitting of data

verbose has been set to 1 for logging output


"""


mod_loss_2=pd.DataFrame(model1.history.history)
mod_loss_2.plot()
plt.show()


""" 
The latest plot obtained has shown us that our model 's performance has been improved

but still validation data is still somewhat seems like been overfitted..


so for this issue .. we can use dropout layers..


Dropout layers basically just turn off some fraction of neurons randomly to ensure ,
weights and biases are not updated and prevent overfitting of data..

"""

from tensorflow.keras.layers import Dropout



model2 = Sequential()

model2.add(Dense(30, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(15, activation='relu'))
model2.add(Dropout(0.5))


model2.add(Dense(1,activation='sigmoid'))


model2.compile(loss='binary_crossentropy',optimizer='adam')


early_stop=EarlyStopping(monitor='val_loss',mode='min',patience=25,verbose=1)


model2.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),epochs=450,
          verbose=1,callbacks=[early_stop])

""" 
Generally You can keep any value of dropout you want between 0 to 1,comprising a fraction of 
0 to 100 %

usually we keep value between 0.2 to 0.5, which is 20 to 50 % respectively

"""

mod_loss_3=pd.DataFrame(model2.history.history)
mod_loss_3.plot()
plt.show()


""" 
So as you can see the above plot..our model ' s performance has been increased much better 
by now...


Metrics and Evaluation : 
"""

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

predictions=model2.predict_classes(X_test)

print(predictions)

print("y test shape :",y_test.shape)
print("Predictions shape : ",predictions.shape)
predictions=predictions.reshape(171,)

mod_comp_df=pd.DataFrame({'Actual Class ':y_test,'Predicted Class':predictions})
comp=mod_comp_df.head(20)
print(comp)




print('\n')
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))
print('\n')
print("The Accuracy Score : ",round(accuracy_score(y_test,predictions),2))