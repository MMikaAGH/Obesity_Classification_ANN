import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)
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

#Model bazowy
model_base = keras.Sequential()
model_base.add(keras.layers.InputLayer(input_shape=[14])) 
model_base.add(keras.layers.Dense(32, activation="relu"))
model_base.add(keras.layers.Dense(32, activation="relu")) 
model_base.add(keras.layers.Dense(7, activation="softmax")) 

model_base.summary()
model_base.compile(loss="categorical_crossentropy",
 optimizer="sgd",
 metrics=["accuracy"])
#Uczenie
history = model_base.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

#Model na zbiorze testowym
ev_base = model_base.evaluate(X_test, y_test)
accuracy_dic = {}
accuracy_dic['base'] = ev_base[1]

# Teraz sprawdzę prognoze dla 5 obiektów - używam zbioru testowego
X_new = X_test[:5]
y_proba = pd.DataFrame(model_base.predict(X_new))
y_proba.columns = y.columns
print(y_proba.round(2))
y_new = y_test.iloc[:5]
print(y_new)

#Zapis
model_base.save("Obesity.keras")

# ANALIZA ILOSCI WARSTW 
#Model z 1 warstwa ukryta
model_layer1 = keras.Sequential()
model_layer1.add(keras.layers.InputLayer(input_shape=[14])) 
model_layer1.add(keras.layers.Dense(32, activation="relu"))
model_layer1.add(keras.layers.Dense(7, activation="softmax")) 

model_layer1.summary()
model_layer1.compile(loss="categorical_crossentropy",
 optimizer="sgd",
 metrics=["accuracy"])
#Uczenie
history_l1 = model_layer1.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

#Model na zbiorze testowym
ev_l1 = model_layer1.evaluate(X_test, y_test)
accuracy_dic['number of layers: 1'] = ev_l1[1]

# Teraz sprawdzę prognoze dla 5 obiektów - używam zbioru testowego
y_proba = pd.DataFrame(model_layer1.predict(X_new))
y_proba.columns = y.columns
print(y_proba.round(2))
print(y_new)

#Zapis
model_layer1.save("Obesity_l1.keras")

#Model z 3 warstwami ukrytymi
model_layer2 = keras.Sequential()
model_layer2.add(keras.layers.InputLayer(input_shape=[14])) 
model_layer2.add(keras.layers.Dense(32, activation="relu"))
model_layer2.add(keras.layers.Dense(32, activation="relu"))
model_layer2.add(keras.layers.Dense(32, activation="relu"))
model_layer2.add(keras.layers.Dense(7, activation="softmax")) 
model_layer2.summary()
model_layer2.compile(loss="categorical_crossentropy",
 optimizer="sgd",
 metrics=["accuracy"])
#Uczenie
history_l2 = model_layer2.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

#Model na zbiorze testowym
ev_l2 = model_layer2.evaluate(X_test, y_test)
accuracy_dic['number of layers: 3'] = ev_l2[1]

# Teraz sprawdzę prognoze dla 5 obiektów - używam zbioru testowego
y_proba = pd.DataFrame(model_layer2.predict(X_new))
y_proba.columns = y.columns
print(y_proba.round(2))
print(y_new)

#Zapis
model_layer2.save("Obesity_l2.keras")

#Model z 4 warstwami ukrytymi
model_layer3 = keras.Sequential()
model_layer3.add(keras.layers.InputLayer(input_shape=[14]))
model_layer3.add(keras.layers.Dense(32, activation="relu")) 
model_layer3.add(keras.layers.Dense(32, activation="relu"))
model_layer3.add(keras.layers.Dense(32, activation="relu"))
model_layer3.add(keras.layers.Dense(32, activation="relu"))
model_layer3.add(keras.layers.Dense(7, activation="softmax")) 
model_layer3.summary()
model_layer3.compile(loss="categorical_crossentropy",
 optimizer="sgd",
 metrics=["accuracy"])
#Uczenie
history_l3 = model_layer3.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

#Model na zbiorze testowym
ev_l3 = model_layer3.evaluate(X_test, y_test)
accuracy_dic['number of layers: 4'] = ev_l3[1]

# Teraz sprawdzę prognoze dla 5 obiektów - używam zbioru testowego
y_proba = pd.DataFrame(model_layer3.predict(X_new))
y_proba.columns = y.columns
print(y_proba.round(2))
print(y_new)

#Zapis
model_layer3.save("Obesity_l3.keras")

# POROWNANIE I WNIOZKI
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()

pd.DataFrame(history_l1.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()

pd.DataFrame(history_l2.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()

pd.DataFrame(history_l3.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()

print(accuracy_dic)
del accuracy_dic["number of layers: 1"]
del accuracy_dic["number of layers: 3"]
del accuracy_dic["number of layers: 4"]
# ANALIZA ILOSCI NEURONOW W WARSTWIE
#100 neuronów w każdej warstwie
model_neuron1 = keras.Sequential()
model_neuron1.add(keras.layers.InputLayer(input_shape=[14])) 
model_neuron1.add(keras.layers.Dense(100, activation="relu"))
model_neuron1.add(keras.layers.Dense(100, activation="relu")) 
model_neuron1.add(keras.layers.Dense(7, activation="softmax")) 

model_neuron1.summary()
model_neuron1.compile(loss="categorical_crossentropy",
 optimizer="sgd",
 metrics=["accuracy"])
#Uczenie
history_n1 = model_neuron1.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

#Model na zbiorze testowym
ev_n1 = model_neuron1.evaluate(X_test, y_test)
accuracy_dic['number of neurons: 100 and 100'] = ev_n1[1]

# Teraz sprawdzę prognoze dla 5 obiektów - używam zbioru testowego
y_proba = pd.DataFrame(model_neuron1.predict(X_new))
y_proba.columns = y.columns
print(y_proba.round(2))
print(y_new)

#Zapis
model_neuron1.save("Obesity_n1.keras")

# 64 i 32 neurony w warstwach ukrytych
model_neuron2 = keras.Sequential()
model_neuron2.add(keras.layers.InputLayer(input_shape=[14])) 
model_neuron2.add(keras.layers.Dense(200, activation="relu"))
model_neuron2.add(keras.layers.Dense(200, activation="relu")) 
model_neuron2.add(keras.layers.Dense(7, activation="softmax")) 

model_neuron2.summary()
model_neuron2.compile(loss="categorical_crossentropy",
 optimizer="sgd",
 metrics=["accuracy"])
#Uczenie
history_n2 = model_neuron2.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

#Model na zbiorze testowym
ev_n2 = model_neuron2.evaluate(X_test, y_test)
accuracy_dic['number of neurons: 200 and 200'] = ev_n2[1]

# Teraz sprawdzę prognoze dla 5 obiektów - używam zbioru testowego
y_proba = pd.DataFrame(model_neuron2.predict(X_new))
y_proba.columns = y.columns
print(y_proba.round(2))
print(y_new)

#Zapis
model_neuron2.save("Obesity_n2.keras")

#16 i 32 neutrony w warstwach ukryty
model_neuron3 = keras.Sequential()
model_neuron3.add(keras.layers.InputLayer(input_shape=[14])) 
model_neuron3.add(keras.layers.Dense(300, activation="relu"))
model_neuron3.add(keras.layers.Dense(300, activation="relu")) 
model_neuron3.add(keras.layers.Dense(7, activation="softmax")) 

model_neuron3.summary()
model_neuron3.compile(loss="categorical_crossentropy",
 optimizer="sgd",
 metrics=["accuracy"])
#Uczenie
history_n3 = model_neuron3.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

#Model na zbiorze testowym
ev_n3 = model_neuron3.evaluate(X_test, y_test)
accuracy_dic['number of neurons: 300 and 300'] = ev_n3[1]

# Teraz sprawdzę prognoze dla 5 obiektów - używam zbioru testowego
y_proba = pd.DataFrame(model_neuron3.predict(X_new))
y_proba.columns = y.columns
print(y_proba.round(2))
print(y_new)

#Zapis
model_neuron3.save("Obesity_n3.keras")

# POROWNANIE I WNIOZKI
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()

pd.DataFrame(history_n1.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()

pd.DataFrame(history_n2.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()

pd.DataFrame(history_n3.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()
print(accuracy_dic)
del accuracy_dic['number of neurons: 100 and 100']
del accuracy_dic['number of neurons: 200 and 200']
del accuracy_dic['number of neurons: 300 and 300']

#ANALIZA FUNKCJI AKTYWACYJNYCH
# funkcja sigmoid
model_fun1 = keras.Sequential()
model_fun1.add(keras.layers.InputLayer(input_shape=[14])) 
model_fun1.add(keras.layers.Dense(32, activation="sigmoid"))
model_fun1.add(keras.layers.Dense(32, activation="sigmoid")) 
model_fun1.add(keras.layers.Dense(7, activation="softmax")) 

model_fun1.summary()
model_fun1.compile(loss="categorical_crossentropy",
 optimizer="sgd",
 metrics=["accuracy"])
#Uczenie
history_fun1 = model_fun1.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

#Model na zbiorze testowym
ev_f1 = model_fun1.evaluate(X_test, y_test)
accuracy_dic['activaction function: sigmoid'] = ev_f1[1]

# Teraz sprawdzę prognoze dla 5 obiektów - używam zbioru testowego
y_proba = pd.DataFrame(model_fun1.predict(X_new))
y_proba.columns = y.columns
print(y_proba.round(2))
print(y_new)

#Zapis
model_fun1.save("Obesity_f1.keras")

#funkcja tanh
model_fun2 = keras.Sequential()
model_fun2.add(keras.layers.InputLayer(input_shape=[14])) 
model_fun2.add(keras.layers.Dense(32, activation="tanh"))
model_fun2.add(keras.layers.Dense(32, activation="tanh")) 
model_fun2.add(keras.layers.Dense(7, activation="softmax")) 

model_fun2.summary()
model_fun2.compile(loss="categorical_crossentropy",
 optimizer="sgd",
 metrics=["accuracy"])
#Uczenie
history_fun2 = model_fun2.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

#Model na zbiorze testowym
ev_f2 = model_fun2.evaluate(X_test, y_test)

accuracy_dic['activaction function: tanh'] = ev_f2[1]

# Teraz sprawdzę prognoze dla 5 obiektów - używam zbioru testowego
y_proba = pd.DataFrame(model_fun2.predict(X_new))
y_proba.columns = y.columns
print(y_proba.round(2))
print(y_new)

#Zapis
model_fun2.save("Obesity_f2.keras")

#funkcja linear
model_fun3 = keras.Sequential()
model_fun3.add(keras.layers.InputLayer(input_shape=[14])) 
model_fun3.add(keras.layers.Dense(32, activation="linear"))
model_fun3.add(keras.layers.Dense(32, activation="linear")) 
model_fun3.add(keras.layers.Dense(7, activation="softmax")) 

model_fun3.summary()
model_fun3.compile(loss="categorical_crossentropy",
 optimizer="sgd",
 metrics=["accuracy"])
#Uczenie
history_fun3 = model_fun3.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

#Model na zbiorze testowym
ev_f3 = model_fun3.evaluate(X_test, y_test)
accuracy_dic['activaction function: linear'] = ev_f3[1]

# Teraz sprawdzę prognoze dla 5 obiektów - używam zbioru testowego
y_proba = pd.DataFrame(model_fun3.predict(X_new))
y_proba.columns = y.columns
print(y_proba.round(2))
print(y_new)

#Zapis
model_fun3.save("Obesity_f3.keras")

# POROWNANIE I WNIOZKI
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()

pd.DataFrame(history_fun1.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()

pd.DataFrame(history_fun2.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()

pd.DataFrame(history_fun3.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()
print(accuracy_dic)
del accuracy_dic['activaction function: linear']
del accuracy_dic['activaction function: sigmoid']
del accuracy_dic['activaction function: tanh']

# ANALIZA ILOSCI EPOK

#Epoki = 10
model_epocs1 = keras.Sequential()
model_epocs1.add(keras.layers.InputLayer(input_shape=[14])) 
model_epocs1.add(keras.layers.Dense(32, activation="relu"))
model_epocs1.add(keras.layers.Dense(32, activation="relu")) 
model_epocs1.add(keras.layers.Dense(7, activation="softmax")) 

model_epocs1.summary()
model_epocs1.compile(loss="categorical_crossentropy",
 optimizer="sgd",
 metrics=["accuracy"])
#Uczenie
history_epo1 = model_epocs1.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

#Model na zbiorze testowym
ev_epo1 = model_epocs1.evaluate(X_test, y_test)
accuracy_dic['number of epocs: 10'] = ev_epo1[1]

# Teraz sprawdzę prognoze dla 5 obiektów - używam zbioru testowego
y_proba = pd.DataFrame(model_epocs1.predict(X_new))
y_proba.columns = y.columns
print(y_proba.round(2))
print(y_new)

#Zapis
model_epocs1.save("Obesity_epo1.keras")

#Epoki = 200
model_epocs2 = keras.Sequential()
model_epocs2.add(keras.layers.InputLayer(input_shape=[14])) 
model_epocs2.add(keras.layers.Dense(32, activation="relu"))
model_epocs2.add(keras.layers.Dense(32, activation="relu")) 
model_epocs2.add(keras.layers.Dense(7, activation="softmax")) 

model_epocs2.summary()
model_epocs2.compile(loss="categorical_crossentropy",
 optimizer="sgd",
 metrics=["accuracy"])
#Uczenie
history_epo2 = model_epocs2.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val))

#Model na zbiorze testowym
ev_epo2 = model_epocs2.evaluate(X_test, y_test)
accuracy_dic['number of epocs: 200'] = ev_epo2[1]

# Teraz sprawdzę prognoze dla 5 obiektów - używam zbioru testowego
y_proba = pd.DataFrame(model_epocs2.predict(X_new))
y_proba.columns = y.columns
print(y_proba.round(2))
print(y_new)

#Zapis
model_epocs2.save("Obesity_epo2.keras")

#Epoki = 2000
model_epocs3 = keras.Sequential()
model_epocs3.add(keras.layers.InputLayer(input_shape=[14])) 
model_epocs3.add(keras.layers.Dense(32, activation="relu"))
model_epocs3.add(keras.layers.Dense(32, activation="relu")) 
model_epocs3.add(keras.layers.Dense(7, activation="softmax")) 

model_epocs3.summary()
model_epocs3.compile(loss="categorical_crossentropy",
 optimizer="sgd",
 metrics=["accuracy"])
#Uczenie
history_epo3 = model_epocs3.fit(X_train, y_train, epochs=2000, validation_data=(X_val, y_val))

#Model na zbiorze testowym
ev_epo3 = model_epocs3.evaluate(X_test, y_test)
accuracy_dic['number of epocs: 2000'] = ev_epo3[1]

# Teraz sprawdzę prognoze dla 5 obiektów - używam zbioru testowego
y_proba = pd.DataFrame(model_epocs3.predict(X_new))
y_proba.columns = y.columns
print(y_proba.round(2))
print(y_new)

#Zapis
model_epocs3.save("Obesity_epo3.keras")
# POROWNANIE I WNIOZKI
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()

pd.DataFrame(history_epo1.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()

pd.DataFrame(history_epo2.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()

pd.DataFrame(history_epo3.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()
print(accuracy_dic)
del accuracy_dic['number of epocs: 10']
del accuracy_dic['number of epocs: 200']
del accuracy_dic['number of epocs: 2000']

#ANALIZA PODZIALU ZBIORU TRENINGOWEGO, VALIDACYJNEGO I TESTOWEGO
#Treningowy 50%, testowy 25%, walidacyjny 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
model_set1 = keras.Sequential()
model_set1.add(keras.layers.InputLayer(input_shape=[14])) 
model_set1.add(keras.layers.Dense(32, activation="relu"))
model_set1.add(keras.layers.Dense(32, activation="relu")) 
model_set1.add(keras.layers.Dense(7, activation="softmax")) 

model_set1.summary()
model_set1.compile(loss="categorical_crossentropy",
 optimizer="sgd",
 metrics=["accuracy"])
#Uczenie
history_s1 = model_set1.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

#Model na zbiorze testowym
ev_s1 = model_set1.evaluate(X_test, y_test)
accuracy_dic['set: 50% training, 25%val, 25%test'] = ev_s1[1]

# Teraz sprawdzę prognoze dla 5 obiektów - używam zbioru testowego
y_proba = pd.DataFrame(model_set1.predict(X_new))
y_proba.columns = y.columns
print(y_proba.round(2))
print(y_new)

#Zapis
model_set1.save("Obesity_s1.keras")
#Treningowy 50%, testowy 50%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
model_set2 = keras.Sequential()
model_set2.add(keras.layers.InputLayer(input_shape=[14])) 
model_set2.add(keras.layers.Dense(32, activation="relu"))
model_set2.add(keras.layers.Dense(32, activation="relu")) 
model_set2.add(keras.layers.Dense(7, activation="softmax")) 

model_set2.summary()
model_set2.compile(loss="categorical_crossentropy",
 optimizer="sgd",
 metrics=["accuracy"])
#Uczenie
history_s2 = model_set2.fit(X_train, y_train, epochs=100)

#Model na zbiorze testowym
ev_s2 = model_set2.evaluate(X_test, y_test)
accuracy_dic['set: 50% training, 50%test'] = ev_s2[1]

# Teraz sprawdzę prognoze dla 5 obiektów - używam zbioru testowego
y_proba = pd.DataFrame(model_set2.predict(X_new))
y_proba.columns = y.columns
print(y_proba.round(2))
print(y_new)

#Zapis
model_set2.save("Obesity_s2.keras")

#Treningowy 90%, testowy 5%, walidacyjny 5%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
model_set3 = keras.Sequential()
model_set3.add(keras.layers.InputLayer(input_shape=[14])) 
model_set3.add(keras.layers.Dense(32, activation="relu"))
model_set3.add(keras.layers.Dense(32, activation="relu")) 
model_set3.add(keras.layers.Dense(7, activation="softmax")) 

model_set3.summary()
model_set3.compile(loss="categorical_crossentropy",
 optimizer="sgd",
 metrics=["accuracy"])
#Uczenie
history_s3 = model_set3.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

#Model na zbiorze testowym
ev_s3 = model_set3.evaluate(X_test, y_test)
accuracy_dic['set: 90% training, 5%val, 5%test'] = ev_s3[1]

# Teraz sprawdzę prognoze dla 5 obiektów - używam zbioru testowego
y_proba = pd.DataFrame(model_set3.predict(X_new))
y_proba.columns = y.columns
print(y_proba.round(2))
print(y_new)

#Zapis
model_set3.save("Obesity_s3.keras")
# POROWNANIE I WNIOZKI
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()

pd.DataFrame(history_s1.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()

pd.DataFrame(history_s2.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()

pd.DataFrame(history_s3.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Wyznacza zakres osi pionowej od 0 do 1
plt.show()
print(accuracy_dic)