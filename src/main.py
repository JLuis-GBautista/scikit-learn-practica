import numpy as np
from sklearn.model_selection import train_test_split
from numpy.typing import NDArray

comp = np.load('utils/particiones-datos-balanceados.npz')
x: NDArray[np.float64] = np.load('utils/particiones-datos-balanceados.npz')['X']
y: NDArray[np.int8] = np.load('utils/particiones-datos-balanceados.npz')['Y']

print(x, x.shape)
print(y, y.shape)

# La muestra se usara con respecto a las siguientes particiones:
# 60% entrenamiento, 40% resto

x_train: NDArray[np.float64]
x_resto: NDArray[np.float64]
y_train: NDArray[np.int8]
y_resto: NDArray[np.int8]

x_train, x_resto, y_train, y_resto = train_test_split(x, y, test_size=.4, random_state=123)

print('Xtrain: ',x_train)
print('Xresto: ', x_resto)

print('Ytrain: ', y_train)
print('Yresto: ', y_resto)

# La muestra restante se usara con respecto a las siguientes particiones:
# 50% validacion, 50% prueba

x_val: NDArray[np.float64]
x_test: NDArray[np.float64]
y_val: NDArray[np.int8]
y_test: NDArray[np.int8]

x_val, x_test, y_val, y_test = train_test_split(x_resto, y_resto, test_size=.5, random_state=246)


print('Xval: ', x_val)
print('Xtest: ', x_test)

print('Yval: ', y_val)
print('Ytest: ', y_test)

print('Proporciones categorías (0s/1s): ')
print(f'\tDataset original: {np.sum(y==0)/len(y)}/{np.sum(y==1)/len(y)}')
print(f'\tEntrenamiento: {np.sum(y_train==0)/len(y_train)}/{np.sum(y_train==1)/len(y_train)}')
print(f'\tValidación: {np.sum(y_val==0)/len(y_val)}/{np.sum(y_val==1)/len(y_val)}')
print(f'\tPrueba: {np.sum(y_test==0)/len(y_test)}/{np.sum(y_test==1)/len(y_test)}')