# Utils -----------------------
import numpy as np
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, zipfile, io, re
from PIL import Image, ImageOps
import random
import pickle
import datetime
import gc
from tqdm import tqdm
import warnings
import seaborn as sns
# Machine Learning ---------------
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from math import sqrt
import optuna
from optuna import integration
# Keras, TensorFlow ---------------
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, BatchNormalization, Convolution2D, Input
from keras import optimizers
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings('ignore')
gpus = 1
# IO Functions ------------------------------
def pkl_saver(object, pkl_filename):
    with open(pkl_filename, 'wb') as web:
        pickle.dump(object , web)


def pkl_loader(pkl_filename):
    with open(pkl_filename, 'rb') as web:
        data = pickle.load(web)
    return data


# Dir generator ----------------------------
def dir_generator():
    timename = '{0:%Y%m%d%H%M}'.format(datetime.datetime.now())
    os.mkdir(os.path.join(os.getcwd(), timename))
    os.chdir(os.path.join(os.getcwd(), timename))
    result_img_dir = './result_img/'
    if os.path.exists(result_img_dir) == False:
        os.mkdir(result_img_dir)
    model_dir = './model/'
    if os.path.exists(model_dir) == False:
        os.mkdir(model_dir)
    weights_dir = './weights/'
    if os.path.exists(weights_dir) == False:
        os.mkdir(weights_dir)
    logging_dir = './logs/'
    if os.path.exists(logging_dir) == False:
        os.mkdir(logging_dir)
    return timename

# Data Loader ----------------------------------
def crown_DataLoader(zip_name):
    z = zipfile.ZipFile(zip_name)
    imgfiles = [x for x in z.namelist()]
    #imgfiles = [x for x in z.namelist() if re.search(r'^' + zip_name.split('.')[0] + '.tif$', x)]
    filenames = []
    X=[]
    Y=[]
    point = []
    max_light = 0
    print('NTL_processing...')
    ext = ('.tif')
    for imgfile in tqdm(imgfiles):
        if imgfile.endswith(ext):
            print(imgfile)
            image = Image.open(io.BytesIO(z.read(imgfile)))
            data = np.asarray(image).reshape(image_size,image_size,-1)
            file = os.path.basename(imgfile)
            file_split = [i for i in file.split('_')]
            y = float(os.path.splitext(file_split[3])[0])
            filenames.append(file)
            X.append(data)
            Y.append(y)
            point.append([float(file_split[1]), float(file_split[2])])
    z.close()
    filenames = np.array(filenames)
    X = np.array(X).astype('float32')
    Y = np.array(Y).astype('float32')
    print(X.shape, Y.shape)
    return filenames, X, Y, point


def region_visualizer(df):
    points = df[3]
    lon, lat = [], []
    for point in points:
        lon.append(point[0])
        lat.append(point[1])
    lon = np.array(lon).astype(float).reshape(-1,1)
    lat = np.array(lat).astype(float).reshape(-1,1)
    region = df[4]
    region = np.array(region).astype(int).reshape(-1,1)
    df = pd.DataFrame(np.concatenate([lon, lat, region], axis=1))
    df.columns = ['longitude', 'latitude', 'region_class']
    pivotted = df.pivot('longitude', 'latitude', 'region_class')
    for i in range(pivotted.shape[0]):
        pivotted.iloc[i] = pd.to_numeric(pivotted.iloc[i])
    pivotted.columns = pd.to_numeric(pivotted.columns)
    pivotted.index = pd.to_numeric(pivotted.index)
    pivotted = pivotted.fillna(-1)
    pivotted = pivotted.astype(float).T
    cmap = sns.color_palette("deep", cvs + 1)
    cmap[0] = (0,0,0)
    plt = sns.heatmap(pivotted, cmap = cmap)
    plt.invert_yaxis()
    colorbar = plt.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + 0.5 * r / (cvs + 1) + r * i / (cvs + 1) for i in range(cvs + 1)])
    colorbar.set_ticklabels(['background']+list(range(cvs)))
    plt.figure.savefig("./result_img/region_map.jpg")
    del(plt)


def data_splitter_cv(X, Y, point, cv, region):
    #test_index = np.where(region==cv)
    test_index = np.arange(cv, X.shape[0], cvs)
    #train_index = np.where(region!=cv)
    train_index = np.setdiff1d(np.arange(0, X.shape[0], 1), test_index)
    X_test = X[test_index]
    y_test = Y[test_index]
    X_train = X[train_index]
    y_train = Y[train_index]
    return X_train, X_test, y_train, y_test


# Loss Definition ----------------------------------
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis = -1))


def create_model(image_shape, num_layer, padding, dense_num, num_filters, size_filters, dropout_rate_in, dropout_rate_out):
    inputs = Input(image_shape)
    with tf.device('/gpu:0'):
        x = Dropout(dropout_rate_in)(inputs)
        x = Convolution2D(filters = 2**num_filters[0], kernel_size = (size_filters[0],size_filters[0]), padding = 'same', activation = 'relu')(x)
        for i in range(1, num_layer):
            x = Convolution2D(filters = 2**num_filters[i],
                              kernel_size = (size_filters[i], size_filters[i]),
                              padding = padding,
                              activation = 'relu')(x)
            x = MaxPooling2D()(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(units = 2**dense_num)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate_out)(x)
        x = Dense(units = 1)(x)
        model = Model(inputs = inputs, outputs = x)
    return model


def opt_cnn(trial):
    # Opt params -----------------------
    # Categorical parameter
    num_layer = trial.suggest_int('num_layer', 1, 2)
    dense_num = trial.suggest_int('dense_num', 2, 4)
    num_filters = [int(trial.suggest_discrete_uniform('num_filter_' + str(i), 2, 4, 1)) for i in range(num_layer)]
    size_filters = [int(trial.suggest_discrete_uniform('size_filter_' + str(i), 3, 5, 2)) for i in range(num_layer)]
    batch_size = trial.suggest_int('batch_size', 1, 4)
    # Model Compiler -----------------------
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    decay = trial.suggest_loguniform('decay', 1e-6, 1e-3)
    # Discrete-uniform parameter
    dropout_rate_in = trial.suggest_discrete_uniform('dropout_rate_in', 0.0, 0.5, 0.1)
    dropout_rate_out = trial.suggest_discrete_uniform('dropout_rate_out', 0.0, 0.5, 0.1)
    momentum = trial.suggest_discrete_uniform('momentum', 0.0, 1.0, 0.1)
    # categorical parameter
#    optimizer = trial.suggest_categorical("optimizer", ["sgd", "momentum", "rmsprop", "adam"])
    padding = trial.suggest_categorical('padding', ['same', 'valid'])
    # compile model-------------------
    model = create_model(image_shape, num_layer, padding, dense_num, num_filters, size_filters, dropout_rate_in, dropout_rate_out)
    sgd = optimizers.SGD(lr = lr, decay = decay, momentum = momentum, nesterov = True)
#    sgd = optimizers.SGD(lr = lr, decay = decay, momentum = momentum, nesterov = True, clipvalue = 1.0)
    # For CPU run ------------------
    model.compile(optimizer = sgd, loss = root_mean_squared_error)
    # Train Model ----------------------------------
    es_cb = EarlyStopping(monitor = 'val_loss', patience = early_stopping, verbose = 1)
    pr_cb = integration.TFKerasPruningCallback(trial, 'val_loss')
    cbs = [es_cb, pr_cb]
    rmse_list = []
    for inner_cv in tqdm(range(0, cvs)):
        X_val_train, X_val, y_val_train, y_val = data_splitter_cv(X_train, y_train, point, inner_cv, region)
        hist = model.fit(
            train_datagen.flow(X_val_train, y_val_train, batch_size = (2**batch_size) * gpus),
            epochs = train_epochs,
            validation_data = (X_val, y_val),
            callbacks = cbs,
            shuffle = True,
            verbose = 1,
            use_multiprocessing = False)
        rmse_list += [model.evaluate(X_val, y_val)]
    del model
    keras.backend.clear_session()
    gc.collect()
    eval = np.mean(rmse_list)
    return eval


def mean_params_calc(param_names):
    dict = {}
    categoricals = ['padding']
    for param_name in param_names:
        data_num = 0
        if param_name not in categoricals:
            for data in best_params:
                try:
                    try:
                        dict[param_name] += data[param_name]
                    except:
                        dict[param_name] = data[param_name]
                    data_num = data_num + 1
                except:
                    pass
            dict[param_name] = dict[param_name]/data_num
        else:
            categorical_list = []
            for data in best_params:
                try:
                    categorical_list = categorical_list + [data[param_name]]
                except:
                    pass
            dict[param_name] = stats.mode(categorical_list)[0][0]
    return dict


def cv_result_imgs_generator(model, history):
    # Visualize Loss Results ----------------------------
    plt.figure(figsize=(18,6))
    plt.plot(history.history["loss"], label="loss", marker="o")
    plt.plot(history.history["val_loss"], label="val_loss", marker="o")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("")
    plt.legend(loc="best")
    plt.grid(color='gray', alpha=0.2)
    plt.savefig('./result_img/' + str(cv) + '_loss.jpg')
    plt.close()
    # Train data -----------------------
    plt.figure()
    y_train_preds = model.predict(X_train)
    plt.scatter(y_train, y_train_preds, s=3, alpha=0.5)
    plt.xlim(min([np.min(y_train_preds), np.min(y_train)]), max([np.max(y_train_preds), np.max(y_train)]))
    plt.xlabel("obs")
    plt.ylabel("pred")
    x = np.linspace(min([np.min(y_train_preds), np.min(y_train)]), max([np.max(y_train_preds), np.max(y_train)]), 100)
    y = x
    plt.plot(x, y, 'r-')
    plt.savefig('./result_img/' + str(cv) + '_scatter_train.jpg')
    plt.close()
    # Evaluate test data -----------------------
    plt.figure()
    y_preds = model.predict(X_val)
    plt.scatter(y_val, y_preds, s=3, alpha=0.5)
    plt.xlim(min([np.min(y_val),np.min(y_preds)]), max([np.max(y_val), np.max(y_preds)]))
    plt.ylim(min([np.min(y_val),np.min(y_preds)]), max([np.max(y_val), np.max(y_preds)]))
    plt.xlabel("obs")
    plt.ylabel("pred")
    x = np.linspace(min([np.min(y_val),np.min(y_preds)]), max([np.max(y_val), np.max(y_preds)]), 100)
    y = x
    plt.plot(x, y, "r-")
    plt.savefig('./result_img/' + str(cv) + '_scatter_test.jpg')
    plt.close()


def generalization_result_imgs_generator(name, y_val_pred, y_val_all):
    # Evaluate test data -----------------------
    plt.figure()
    plt.scatter(y_val_all, y_val_pred, s=3, alpha=0.5)
    plt.xlim(min([np.min(y_val_all), np.min(y_val_pred)]), max([np.max(y_val_all),np.max(y_val_pred)]))
    plt.xlabel("obs")
    plt.ylabel("pred")
    x = np.linspace(min([np.min(y_val_all), np.min(y_val_pred)]), max([np.max(y_val_all),np.max(y_val_pred)]),100)
    y = x
    plt.plot(x, y, "r-")
    plt.savefig('./result_img/' + name + '_scatter_test.jpg')
    plt.close()


# Data Loader ------------------------------
train_zip_name = 'train_crowns.zip'
test_zip_name = 'test_crowns.zip'
image_size = 30
cvs = 10
train_epochs = 32
ntrials = 32
early_stopping = 32
best_epochs = 32
# Train_DataFrame_Generator ----------------------
filenames, X, Y, point = crown_DataLoader(train_zip_name)
region = KMeans(n_clusters = cvs, random_state=0).fit(point).labels_
filenames_test, X_test, y_test, point_test = crown_DataLoader(test_zip_name)
X_mean, Y_mean = X.mean(), Y.mean()
X_std, Y_std = X.std(), Y.std()
X = (X - X_mean)/X_std
Y = (Y - Y_mean)/Y_std
X_test = (X_test - X_mean)/X_std
y_test = (y_test - Y_mean)/Y_std
df = [filenames, X, Y, point, region]
filenames, X, Y, point, region = df[0], df[1], df[2], df[3], df[4]
image_shape = (X.shape[1], X.shape[2], X.shape[3])
standardization_params = [X_mean, Y_mean, X_std, Y_std]
# Training Settings --------------------------------------
# Data Augmentation --------------------------------
timename = dir_generator()
region_visualizer(df)


# Train Model ----------------------------------
# CV start ------------------------------------------------------------
train_results, y_val_pred, best_params = [], [], []
for cv in tqdm(range(cvs)):
    print('cv_' + str(cv) + '_processing....')
    # Data Loader-------------------------------------
    X_train, X_val, y_train, y_val = data_splitter_cv(X, Y, point, cv, region)
    train_datagen = ImageDataGenerator(
        rotation_range = 360,
        horizontal_flip = True,
        vertical_flip = True)
    test_datagen = ImageDataGenerator()
    # Bayesian optimization -------------------------------------
    study = optuna.create_study()
    study.optimize(opt_cnn, n_trials = ntrials)
    best_params.append(study.best_params)
    num_filters = [int(study.best_params['num_filter_' + str(i)]) for i in range(int(study.best_params['num_layer']))]
    size_filters = [int(study.best_params['size_filter_' + str(i)]) for i in range(int(study.best_params['num_layer']))]
    model = create_model(image_shape, int(study.best_params['num_layer']), study.best_params['padding'], int(study.best_params['dense_num']), num_filters, size_filters, study.best_params['dropout_rate_in'], study.best_params['dropout_rate_out'])
    sgd = optimizers.SGD(lr = study.best_params['learning_rate'], decay = study.best_params['decay'], momentum = study.best_params['momentum'], nesterov = True, clipvalue = 1.0)
    model.compile(optimizer = sgd, loss = root_mean_squared_error)
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size = 2**int(study.best_params['batch_size']) * gpus),
        epochs = train_epochs,
        validation_data = (X_val, y_val),
        shuffle = True,
        verbose = 1,
        use_multiprocessing = False)
    train_results.append(model.evaluate(X_val, y_val))
    try:
        y_val_pred = np.concatenate((y_val_pred, model.predict(X_val)))
    except:
        y_val_pred = model.predict(X_val)
    try:
        y_val_all = np.concatenate((y_val_all, y_val))
    except:
        y_val_all = y_val
    cv_result_imgs_generator(model, history)
    del model
    keras.backend.clear_session()
    gc.collect()


# Save CV_Result -------------------------------------------------
generalization_result_imgs_generator('val', y_val_pred, y_val_all)
pkl_saver(train_results, 'train_results.binaryfile')
pkl_saver(standardization_params, 'standardization_params.binaryfile')
param_names = best_params[list(map(len, best_params)).index(max(list(map(len, best_params))))].keys()
best_params_dict = mean_params_calc(param_names)
pkl_saver(best_params, 'best_params_list.binaryfile')
pkl_saver(best_params_dict, 'best_params.binaryfile')
best_params_dict = pkl_loader('best_params.binaryfile')

# Best Model Training -----------------------------------------------
# Int parameter
num_layer = int(best_params_dict['num_layer'])
num_filters = [int(best_params_dict['num_filter_' + str(i)]) for i in range(num_layer)]
size_filters = [int(best_params_dict['size_filter_' + str(i)]) for i in range(num_layer)]
dense_num = int(best_params_dict['dense_num'])
batch_size = int(best_params_dict['batch_size'])
# Uniform parameter
# Loguniform parameter
lr = best_params_dict['learning_rate']
decay = best_params_dict['decay']
# Discrete-uniform parameter
dropout_rate_in = best_params_dict['dropout_rate_in']
dropout_rate_out = best_params_dict['dropout_rate_out']
momentum = best_params_dict['momentum']
# Categorical parameter
padding = best_params_dict['padding']


# Model Checkpoint ------------------
cp_cb = ModelCheckpoint(
    './weights/best_weights.hdf5',
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True,
    save_weights_only = True,
    mode = 'auto')
# Logging ----------------------------------------
log_dir = os.path.join('./logs/')
tb_cb = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)
es_cb = EarlyStopping(monitor = 'val_loss', patience = int(best_epochs/10), verbose = 1)

cbs = [cp_cb, tb_cb, es_cb]


# Train Best_Model ----------------------------------
# For CPU run ------------------
best_model = create_model(image_shape, num_layer, padding, dense_num, num_filters, size_filters, dropout_rate_in, dropout_rate_out)
sgd = optimizers.SGD(lr = lr, decay = decay, momentum = momentum, nesterov = True, clipvalue = 1.0)


best_model.compile(optimizer = sgd, loss = root_mean_squared_error)
hist = best_model.fit(
    train_datagen.flow(X, Y, batch_size = (2**batch_size) * gpus),
    epochs = best_epochs,
    callbacks = cbs,
    shuffle = True,
    verbose = 1,
    initial_epoch = 0,
    use_multiprocessing = False)

# Save Model -----------------------------------
best_model.save('./model/best_model.hdf5')

# Unknown Predictor ------------------------------------------------
# Load Pre-trained Model -------------------------------------
best_model = load_model('./model/best_model.hdf5', custom_objects={'root_mean_squared_error': root_mean_squared_error})

# Data Loader ------------------------------
y_test_pred = best_model.predict(X_test)
y_test_pred = y_test_pred*Y_std + Y_mean
y_test = y_test*Y_std + Y_mean
generalization_result_imgs_generator('test', y_test_pred, y_test)
np.savetxt('y_test_pred.txt', y_test_pred)
with open("best_model_summary.txt", "w") as fp:
    best_model.summary(print_fn=lambda x: fp.write(x + "\r\n"))

from shutil import copyfile
copyfile('../crowns_v005.py', './crowns_v005.py')
sum_y_pred = np.sum(y_test_pred*Y_std+Y_mean)
sum_y_test = np.sum(y_test*Y_std+Y_mean)

print('finished...')
