from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models,Sequential 
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

#carregando os dados e dividindo entre treino e teste
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

#Abre imagens aleatórias do conjunto de dados para visualização em um grid de tamanho (side x side)
side = 5
start = np.random.random_integers(low=0, high=x_train.shape[0], size=(1,))[0]
fig, ax = plt.subplots(side, side)
for a in range(side):
    for b in range(side):
        ax[a, b].axes.xaxis.set_visible(False)
        ax[a, b].axes.yaxis.set_visible(False)
        ax[a, b].imshow(x_train[2 * a + b + start, :, :], cmap='gray')
plt.show()

#Normaliza os dados para treinamento e validação. Essa célula deve ser preenchida
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

x_train_norm =x_train/255 # insira aqui seu código
x_test_norm = x_test/255# insira aqui seu código

#Transforma os rótulos das imagens no formato one-hot
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

#modelo da rede
model = Sequential() 
model.add(Conv2D(64, kernel_size=(5,5), activation='relu', input_shape=(28, 28, 1) )) 
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Conv2D(102, kernel_size=(5,5), activation='relu')) 
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Flatten())
model.add(Dense(1000, activation='relu')) 
model.add(Dense(10, activation='softmax')) 
model.summary()

#copila o modelo
model.compile(loss= 'categorical_crossentropy',# insira aqui seu código,
              optimizer= 'adam',# insira aqui seu código,

#treina a rede              
model.fit(x_train_norm, y_train, validation_data=(x_test_norm, y_test), epochs=10)              
              metrics=['accuracy'])
exibe resultados
#
results = model.evaluate(x_test, y_test)
print("test loss, test acc:", results)
