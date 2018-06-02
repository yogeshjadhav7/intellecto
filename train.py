
# coding: utf-8

# In[1]:


from intellecto import Intellecto
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib

N_GAMES_PER_EPISODE = 100
N_EPISODES = 1000
VALIDATION_REFRESH_RATE = 50
N_VALIDATION_GAMES = N_GAMES_PER_EPISODE * VALIDATION_REFRESH_RATE
I = Intellecto()
batch_size = N_GAMES_PER_EPISODE * (I.n_bubbles + I.queue_size)

n_input = (I.n_bubbles + 1) * (I.n_bubbles + 2) 
n_output = I.n_bubbles

PCA_MODEL_NAME = "pca.model"


# In[2]:


def get_training_data(n_games=N_GAMES_PER_EPISODE):
    f, l = I.play_episode(n_games=n_games)
    f = ipca.transform(f)
    return f, l


# In[3]:


def ordering_loss(ground_truth, predictions):
    y = np.array(ground_truth)
    y_ = np.array(predictions)
    y = np.argsort(y, axis=1)
    y_ = np.argsort(y_, axis=1)
    diff_y = np.abs(np.subtract(y, y_))
    loss_array = np.sum(diff_y, axis=1)
    return np.median(loss_array) / 12.0


# In[4]:


ipca = joblib.load(PCA_MODEL_NAME)
print("\n\nGetting initial validation data...")
x_val, y_val = get_training_data(n_games=N_VALIDATION_GAMES)


# In[5]:


# MLP Classifier
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.models import load_model
from keras.metrics import mean_absolute_error

MODEL_NAME = "intellecto.hdf5"
batch_size = 16
num_classes = I.n_bubbles
epochs = 30
input_size = ipca.n_components
TRAIN_MODEL = True

droprate = 0.7
activation = 'elu'
try:
    model = load_model(MODEL_NAME)
    print("Loaded saved model: " + MODEL_NAME)
except:
    print("Creating new model: " + MODEL_NAME)
    
    model = Sequential()
    model.add(Dense(units=1024, activation=activation, input_shape=(input_size, )))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))
    
    model.add(Dense(units=512, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))

    model.add(Dense(units=512, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 1.5))

    model.add(Dense(units=256, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 1.5))
    
    model.add(Dense(units=256, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 1.5))
    
    model.add(Dense(units=256, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 1.5))

    model.add(Dense(units=128, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 2))
    
    model.add(Dense(units=128, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 2))
    
    model.add(Dense(units=128, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 2))

    model.add(Dense(units=64, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 2))
    
    model.add(Dense(units=64, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 2.5))
    
    model.add(Dense(units=32, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 3))

    model.add(Dense(units=32, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 3))
    
    model.add(Dense(units=16, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 3.5))

    #model.add(Dense(num_classes, activation='sigmoid'))
    model.add(Dense(num_classes, activation=None))
    
    model.summary()
    
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=[mean_absolute_error])

def do_on_epoch_end(epoch, _):
    if (epoch + 1) % 10 == 0:
        saved_model = load_model(MODEL_NAME)
        y_val_ = saved_model.predict(x_val)
        print("Ordering loss on val data ", ordering_loss(y_val, y_val_))
        
callbacks_supported = [
    ModelCheckpoint(MODEL_NAME, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1),
    LambdaCallback(on_epoch_end=do_on_epoch_end)
]

callbacks = callbacks_supported
        
if TRAIN_MODEL:
    for n_episodes in range(N_EPISODES):
        print("\nTraining model on episode #" + str(n_episodes) + " ...")
        x, y = get_training_data()
        model.fit(
            x, 
            y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(x_val, y_val),
            callbacks = callbacks
        )
        
        if (n_episodes + 1) % VALIDATION_REFRESH_RATE == 0:
            print("\n\nGetting new validation data...")
            x_val, y_val = get_training_data(n_games=N_VALIDATION_GAMES)

