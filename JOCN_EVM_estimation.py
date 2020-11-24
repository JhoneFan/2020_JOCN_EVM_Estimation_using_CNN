# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import imageio
import keras
import numpy as np

import scipy.io as sio

from keras.layers import Dense
from sklearn import svm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

from keras.layers import Input, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras import optimizers, metrics
from keras.utils import plot_model, np_utils
from keras.optimizers import Adam
import os

import matplotlib.pyplot as plt

import tensorflow as tf

seed = 7  
np.random.seed(seed)  
folder = 'D:/DataSet'

scenarios = []
scenarios.append('4QAM_OSNR12dB')
scenarios.append('4QAM_OSNR14dB')
scenarios.append('4QAM_OSNR16dB')
scenarios.append('4QAM_OSNR18dB')
scenarios.append('4QAM_OSNR20dB')
scenarios.append('4QAM_OSNR22dB')
scenarios.append('4QAM_OSNR24dB')
scenarios.append('4QAM_OSNR26dB')
scenarios.append('4QAM_OSNR28dB')
scenarios.append('4QAM_OSNR30dB')

scenarios.append('16QAM_OSNR20dB')
scenarios.append('16QAM_OSNR22dB')
scenarios.append('16QAM_OSNR24dB')
scenarios.append('16QAM_OSNR26dB')
scenarios.append('16QAM_OSNR28dB')
scenarios.append('16QAM_OSNR30dB')
scenarios.append('16QAM_OSNR32dB')
scenarios.append('16QAM_OSNR34dB')
scenarios.append('16QAM_OSNR36dB')
scenarios.append('16QAM_OSNR38dB')

scenarios.append('64QAM_OSNR26dB')
scenarios.append('64QAM_OSNR28dB')
scenarios.append('64QAM_OSNR30dB')
scenarios.append('64QAM_OSNR32dB')
scenarios.append('64QAM_OSNR34dB')
scenarios.append('64QAM_OSNR36dB')
scenarios.append('64QAM_OSNR38dB')
scenarios.append('64QAM_OSNR40dB')
scenarios.append('64QAM_OSNR42dB')
scenarios.append('64QAM_OSNR44dB')


evm_error_metrics = {}
evm_error_metrics['mean_absolute_error'] = {}
evm_error_metrics['mean_squared_error'] = {}
evm_error_metrics['mean_squared_logarithmic_error'] = {}
evm_truth = {}


ber_error_metrics = {}
ber_error_metrics['mean_absolute_error'] = {}
ber_error_metrics['mean_squared_error'] = {}
ber_error_metrics['mean_squared_logarithmic_error'] = {}
ber_truth = {}

number_figures = {}
for scenario in scenarios:
#    evm_error_metrics['diff'][scenario] = {}
    evm_error_metrics['mean_absolute_error'][scenario] = {}
    evm_error_metrics['mean_squared_error'][scenario] = {}
    evm_error_metrics['mean_squared_logarithmic_error'][scenario] = {}
    
    ber_error_metrics['mean_absolute_error'][scenario] = {}
    ber_error_metrics['mean_squared_error'][scenario] = {}
    ber_error_metrics['mean_squared_logarithmic_error'][scenario] = {}
    
    # loading the truth value
    mat_contents = sio.loadmat(folder + 'fullTrace/' + scenario + '/OutputParams_' + scenario + '_FullTrace.mat')
    evm_truth[scenario] = mat_contents['output']['EVM_mu'][0][0]
    ber_truth[scenario] = mat_contents['output']['BERe_mu'][0][0]
    del mat_contents
    
    for file_mat in sorted(os.listdir(folder + scenario)):
        if file_mat.endswith('.mat'):
            names = file_mat.split('_')
            modulation = names[1]
            osnr = names[2]
            points_per_symbol = int(names[4].replace('.mat', ''))
            print(points_per_symbol)
            
            # import matlab
            mat_contents = sio.loadmat(folder + scenario + '/' + file_mat)
            number_figures[scenario] = 100 #len(evm_figures)
            
            evm_truth_vector = np.tile(evm_truth[scenario], number_figures[scenario])
            ber_truth_vector = np.tile(ber_truth[scenario], number_figures[scenario])
            del mat_contents 
#############################################
dx = 40
dx1 = 60
dy = 50

cut_x = 68+dx
cut_x1 = 68+dx1
cut_y = 168+dy

X = np.zeros((len(scenarios) * 100, 520-dx-dx1, 520-2*dy))
Y = np.zeros((len(scenarios) * 100))

X_train = np.zeros((len(scenarios) * 50,  520-dx-dx1, 520-2*dy))
Y_train = np.zeros((len(scenarios) * 50))

X_validation = np.zeros((len(scenarios) * 25,  520-dx-dx1, 520-2*dy))
Y_validation = np.zeros((len(scenarios) * 25))

X_test = np.zeros((len(scenarios) * 25,  520-dx-dx1, 520-2*dy))
Y_test = np.zeros((len(scenarios) * 25))

################################################################
for id_scenario, scenario in enumerate(scenarios):
    for id_figure in range(number_figures[scenario]):
        image = imageio.imread(folder + scenario + '/Fig_'+ scenario + '_PointsPerSymb_' + str(points_per_symbol) + '_constN_' + str(id_figure + 1) + '.png.png')
        shape = image.shape
        
        r, g, b = image[cut_x:shape[0]-cut_x1, cut_y+20:shape[1] - cut_y + 1, 0], image[cut_x:shape[0]-cut_x1, cut_y+20:shape[1] - cut_y + 1, 1], image[cut_x:shape[0]-cut_x1, cut_y+20:shape[1] - cut_y + 1, 2]
     
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        index = id_scenario * 100 + id_figure
        X[index, :, :] = gray / 255.

    index_full = id_scenario * 100
    
    clazz = 0
    if '64QAM' in scenario:
        clazz = 3
    elif '16QAM' in scenario:
        clazz = 2
    elif '4QAM' in scenario:
        clazz = 1
    else:
        print('error')
    
    index_train = id_scenario * 50
    X_train[index_train:index_train + 50, :, :] = X[index_full:index_full + 50, :, :]
    Y_train[index_train:index_train + 50] = evm_truth[scenario]
    
    index_validation = id_scenario * 25
    print(index_validation, ':', index_validation + 25)
    X_validation[index_validation:index_validation + 25, :, :] = X[index_full + 50:index_full + 50 + 25, :, :]
    Y_validation[index_validation:index_validation + 25] = evm_truth[scenario]

    
    index_test = id_scenario * 25
    X_test[index_test:index_test + 25, :, :] = X[index_full + 50 + 25:index_full + 50 + 25 + 25, :, :]
    Y_test[index_test:index_test + 25] = evm_truth[scenario]


print('done')

model_folder = 'conv_classifier_regressor'
results_folder = folder + 'results/' + model_folder
if not os.path.isdir(results_folder):
    os.makedirs(results_folder)
    print('created folder', results_folder)

#####################CNN regression model#####################################

input_0 = Input(shape=(X.shape[1], X.shape[2], 1))
conv_0 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_0)
pool_0 = MaxPooling2D(pool_size=(2, 2))(conv_0)

conv_1 = Conv2D(16, kernel_size=(3, 3), activation='relu')(pool_0)
pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)

conv_2 = Conv2D(16, kernel_size=(3, 3), activation='relu')(pool_1)
pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)

conv_3 = Conv2D(8, kernel_size=(3, 3), activation='relu')(pool_2)
pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)

flat_0 = Flatten()(pool_3)
dense_0 = Dense(500, activation='relu')(flat_0)
dense_1_0 = Dense(100, activation='relu')(dense_0 )

out_0 = Dense(1, name='evm')(dense_1_0)
model = Model(inputs=input_0, outputs=out_0)
model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='msle')
model.summary()

#######################################################################################################################
shape = X_train.shape
print(shape)
X_train = np.reshape(X_train, newshape=(shape[0], shape[1], shape[2], 1))
shape = X_validation.shape
print(shape)
X_validation = np.reshape(X_validation, newshape=(shape[0], shape[1], shape[2], 1))

print(Y_train.shape)
history = model.fit(X_train, Y_train, batch_size=16, epochs=200, verbose=1,validation_data=(X_validation, Y_validation),shuffle=True)#,shuffle=True


model.save('my_model.h5')
X_test = np.reshape(X_test, newshape=(shape[0], shape[1], shape[2], 1))
evaluation = model.evaluate(X_test, Y_test,batch_size=20)
preds = model.predict(X_test)
print('points',points_per_symbol)
print('MAX diff', max(abs(Y_test.reshape((len(Y_test),1))-preds)))

plt.figure()
plt.plot(Y_test.reshape((len(Y_test),1))-preds)
plt.show()
print('MAE :', mean_absolute_error(Y_test.reshape((len(Y_test),1)), preds))
print('MSE :', mean_squared_error(Y_test.reshape((len(Y_test),1)), preds))
print('MSLE :', mean_squared_log_error(Y_test.reshape((len(Y_test),1)), preds))

###################################################
for key in history.history.keys():
    if 'val' not in key:
        plt.figure()
        if key == 'acc':
            plt.plot(history.history[key], label='train', ls='--')
            plt.plot(history.history['val_'+key], label='validation')
        else:
            plt.semilogy(history.history[key], label='train', ls='--')
            plt.semilogy(history.history['val_'+key], label='validation')
        if key != 'loss':
            for scenario in scenarios:
                plt.semilogy(np.tile(evm_error_metrics[key][scenario][points_per_symbol], len(history.history[key])), label='Est. ' + scenario, ls=':')
            plt.title('Points per symbol: {} | Diff: {}'.format(points_per_symbol, evm_error_metrics[key][scenario][points_per_symbol] - history.history[key][-1]))
        plt.xlabel('Epoch')
        plt.ylabel(key)
        plt.legend()
        plt.tight_layout()
        plt.show()


print(*preds.flatten(), sep=', ')
#print(*history.history['loss'], sep=', ')
#print(*history.history['val_loss'], sep=', ')
