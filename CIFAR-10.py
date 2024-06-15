import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import load_model

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

print('Train Images Shape:      ', X_train.shape)
print('Train Labels Shape:      ', y_train.shape)

print('\nValidation Images Shape: ', X_valid.shape)
print('Validation Labels Shape: ', y_valid.shape)

print('\nTest Images Shape:       ', X_test.shape)
print('Test Labels Shape:       ', y_test.shape)

# CIFAR-10 classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Crie uma nova figura
plt.figure(figsize=(15, 15))

# Faça um loop nas primeiras 25 imagens
for i in range(64):
    # Crie uma subtrama para cada imagem
    plt.subplot(8, 8, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    # Exibir a imagem
    plt.imshow(X_train[i])

    # Defina o rótulo como o título
    plt.title(class_names[y_train[i][0]], fontsize=12)

# Exibir a figura
plt.show()

# Converter tipo de dados de valores de pixel para float32
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')
X_valid = X_valid.astype('float32')

# Calcule a média e o desvio padrão das imagens de treinamento
mean = np.mean(X_train)
std  = np.std(X_train)

# normalizar os dados
# O pequeno valor 1e-7 é adicionado para evitar a divisão por zero
X_train = (X_train-mean)/(std+1e-7)
X_test  = (X_test-mean) /(std+1e-7)
X_valid = (X_valid-mean)/(std+1e-7)

y_train = to_categorical(y_train, 10)
y_valid = to_categorical(y_valid, 10)
y_test  = to_categorical(y_test, 10)

# Aumento de dados
data_generator = ImageDataGenerator(
    # Gire imagens aleatoriamente em até 15 graus
    rotation_range=15,

    # Mude as imagens horizontalmente em até 12% de sua largura
    width_shift_range=0.12,

    # Mude as imagens verticalmente em até 12% de sua altura
    height_shift_range=0.12,

    # Virar imagens aleatoriamente na horizontal
    horizontal_flip=True,

    # Amplie as imagens em até 10%
    zoom_range=0.1,

    # Altere o brilho em até 10%
    brightness_range=[0.9, 1.1],

    # Intensidade de cisalhamento (ângulo de cisalhamento no sentido anti-horário em graus)
    shear_range=10,

    # Intensidade de mudança de canal
    channel_shift_range=0.1,
)

# Inicialize um modelo sequencial
model = Sequential()

# Defina o valor de redução de peso para regularização L2
weight_decay = 0.0001

# Adicione a primeira camada convolucional com 32 filtros de tamanho 3x3
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay),
                 input_shape=X_train.shape[1:]))
# Adicionar camada de normalização em lote
model.add(BatchNormalization())

# Adicione a segunda camada convolucional semelhante à primeira
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())

# Adicione a primeira camada máxima de pooling com tamanho de pool de 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Add dropout layer with 0.2 dropout rate
model.add(Dropout(rate=0.2))

# Adicione a terceira e quarta camadas convolucionais com 64 filtros
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())

# Adicione a segunda camada máxima de pooling e aumente a taxa de abandono para 0,3
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.3))

# Adicione a quinta e a sexta camadas convolucionais com 128 filtros
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())

# Adicione a terceira camada máxima de pooling e aumente a taxa de abandono para 0,4
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.4))

# Adicione a sétima e a oitava camadas convolucionais com 256 filtros
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())

# Adicione a quarta camada máxima de pooling e aumente a taxa de abandono para 0,5
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))

# Achate a saída do tensor da camada anterior
model.add(Flatten())

# Adicione uma camada totalmente conectada com função de ativação softmax para gerar probabilidades de classe
model.add(Dense(10, activation='softmax'))

model.summary()

# Defina o tamanho do lote para o treinamento
batch_size = 64

# Defina o número máximo de épocas para o treinamento
epochs = 300

# Defina o otimizador (Adam)
optimizer = Adam(learning_rate=0.0005)

# Compile o modelo com o otimizador definido, função de perda e métricas
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Adicionar retorno de chamada ReduceLROnPlateau
# Aqui, a taxa de aprendizagem será reduzida pela metade (fator = 0,5) se nenhuma melhoria na perda de validação for observada por 10 épocas
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)

# Adicionar retorno de chamada EarlyStopping
# Aqui, o treinamento será interrompido se nenhuma melhoria na perda de validação for observada por 40 épocas.
# O parâmetro `restore_best_weights` garante que os pesos do modelo sejam redefinidos para os valores da época
# com o melhor valor da quantidade monitorada (neste caso, 'val_loss').
early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=1)

# Ajustar o modelo aos dados de treinamento, usando o tamanho do lote definido e o número de épocas
# Os dados de validação são usados ​​para avaliar o desempenho do modelo durante o treinamento
# Os retornos de chamada implementados são a redução da taxa de aprendizagem quando um platô é atingido na perda de validação e
# parar de treinar cedo se nenhuma melhora for observada
model.fit(data_generator.flow(X_train, y_train, batch_size=batch_size),
          epochs=epochs,
          validation_data=(X_valid, y_valid),
          callbacks=[reduce_lr, early_stopping],
          verbose=2)

plt.figure(figsize=(15,6))

# Traçando a perda de treinamento e validação
plt.subplot(1, 2, 1)
plt.plot(model.history.history['loss'], label='Train Loss', color='#8502d1')
plt.plot(model.history.history['val_loss'], label='Validation Loss', color='darkorange')
plt.legend()
plt.title('Loss Evolution')

# Traçando a precisão do treinamento e validação
plt.subplot(1, 2, 2)
plt.plot(model.history.history['accuracy'], label='Train Accuracy', color='#8502d1')
plt.plot(model.history.history['val_accuracy'], label='Validation Accuracy', color='darkorange')
plt.legend()
plt.title('Accuracy Evolution')

plt.show()

# Use o modelo para fazer previsões e avaliar dados de teste
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

print('\nTest Accuracy:', test_acc)
print('Test Loss:    ', test_loss)

# Obtenha a imagem bruta do GitHub
url = "https://raw.githubusercontent.com/FarzadNekouee/Keras-CIFAR10-CNN-Model/master/truck_sample.png"
resp = urllib.request.urlopen(url)
image = np.asarray(bytearray(resp.read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)

# Converta a imagem de BGR para RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Exibir a imagem
plt.imshow(image)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.show()

# Redimensione para 32x32 pixels
image = cv2.resize(image, (32,32))

# Normalizar a imagem
image = (image-mean)/(std+1e-7)

# Adicione uma dimensão extra porque o modelo espera um lote de imagens
image = image.reshape((1, 32, 32, 3))

prediction = model.predict(image)

predicted_class = prediction.argmax()

print('Predicted class: ', class_names[predicted_class])




