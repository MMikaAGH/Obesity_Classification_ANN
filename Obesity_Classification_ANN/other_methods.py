import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

data = pd.read_csv('C:\\Users\\Marcin\\Desktop\\ProjektII\\ObesityDataSet_cleaned_and_data_sinthetic.csv')
data = data[['Age', 'Gender',"NCP",'family_history_with_overweight', 'FAVC', 'FCVC', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 
             'MTRANS',"NObeyesdad"]]

#zmieniam kolumne "Label" za pomocą kodowania "gorącojedynkowego" (one-hot-encoding)
data['Gender'] = data['Gender'].map({'male': 1, 'female': 0})
data['family_history_with_overweight'] = data['family_history_with_overweight'].map({'yes': 1, 'no': 0})
data['FAVC'] = data['FAVC'].map({'yes': 1, 'no': 0})
data['FCVC'] = data['FCVC'].map({'sometimes': 1, 'always': 2, 'never': 3})
data['CAEC'] = data['CAEC'].map({'no':0,'sometimes': 1, 'frequently': 2, 'always': 3})
data['SMOKE'] = data['SMOKE'].map({'no': 0, 'yes': 1})
data['CH2O'] = data['CH2O'].map({'less than a liter': 0, 'between 1 and 2 l': 1, 'more than 2 l': 2})
data['SCC'] = data['SCC'].map({'no': 0, 'yes': 1})
data['FAF'] = data['FAF'].map({'0': 0, '1 to 2': 1, '2 to 4': 2, '4 to 5': 3})
data['TUE'] = data['TUE'].map({'0 to 2': 0, '3 to 5': 1, '>5': 2})
data['CALC'] = data['CALC'].map({'sometimes': 1, 'no': 0, 'frequently': 2, 'always': 3})
data['MTRANS'] = data['MTRANS'].map({'public_transportation': 0, 'automobile': 1, 'walking': 2, 'motorbike': 3, 'bike': 4})
data_ohe = pd.get_dummies(data, columns=["NObeyesdad"])
data_ohe.head()

# Dziele caly zbiór na treningowy (w tym walidacyjny) i testowy
X = data_ohe.iloc[:,:14]
# Standaryzacja wybranych kolumn
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X))
y = data_ohe.iloc[:,14:]
y.columns = ["insufficient_w","normal_w","obesity_t1","obesity_t2", "obesity_t3","overweight_l1","overweight_l2"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

model_SNN = tf.keras.models.load_model('C:\\Users\\Marcin\\Desktop\\ProjektII\Model\\Obesity.keras')
y_pred_SNN = model_SNN.predict(X_test)
mse1 = mean_squared_error(y_test, y_pred_SNN)
loss_SNN, accuracy_SNN = model_SNN.evaluate(X_test, y_test)

# regresja liniowa
X_b = np.c_[np.ones((len(X), 1)), X]
theta_best = np.linalg.pinv(X_b).dot(y)
y_pred = X_b.dot(theta_best)
mse_all = (np.sum((y - y_pred) ** 2, axis = 0)) / len(y)
mse2 = np.sum(mse_all.values) / 7
print("MSE SNN: ", mse1)
print("MSE Regresja liniowa: ", mse2 )

#regresja wielomianowa 2 stopnia
X_poly = np.column_stack((X, X ** 2))
X_poly_b = np.column_stack((np.ones(X_poly.shape[0]), X_poly))
theta_best_poly = np.linalg.pinv(X_poly_b).dot(y)
y_pred_poly = X_poly_b.dot(theta_best_poly)
mse_3 = np.mean((y_pred_poly - y) ** 2)
print("MSE Regresja wielowymiarowa: ", mse_3)

#musze sprawić aby y miał jednowymiarową formę
y_1d = y_train.idxmax(axis=1)
y_1d2 = y_test.idxmax(axis=1)  
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_1d)
y_test_enc = label_encoder.fit_transform(y_1d2)

#regresja logistyczna
log_reg = LogisticRegression(max_iter=10000, multi_class='multinomial', solver='saga')
log_reg.fit(X_train, y_train_enc)
y_pred_rl = log_reg.predict(X_test)
accuracy_rl = accuracy_score(y_test_enc, y_pred_rl)
print("Dokładność SNN: ", accuracy_SNN)
print("Dokładność regresji logistycznej: ", accuracy_rl)

#Maszyny Wektorów Nośnych
svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
svm_model.fit(X_train, y_train_enc)
y_pred_SVM = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test_enc, y_pred_SVM)
print("Dokładność modelu SVM:", accuracy_svm )