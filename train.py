
# coding: utf-8

# In[14]:


from intellecto import Intellecto
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib

N_GAMES_PER_EPISODE = 250
N_EPISODES = 1000
N_VALIDATION_GAMES = N_GAMES_PER_EPISODE * 100
I = Intellecto()
batch_size = N_GAMES_PER_EPISODE * (I.n_bubbles + I.queue_size)

n_input = (I.n_bubbles + 1) * (I.n_bubbles + 2) 
n_output = I.n_bubbles

PCA_MODEL_NAME = "pca.model"


# In[15]:


def get_training_data(n_games=N_GAMES_PER_EPISODE):
    f, l = I.play_episode(n_games=n_games)
    f = ipca.transform(f)
    return f, l


# In[16]:


def ordering_loss(ground_truth, predictions):
    y = np.array(ground_truth)
    y_ = np.array(predictions)
    y = np.argsort(y, axis=1)
    y_ = np.argsort(y_, axis=1)
    diff_y = np.abs(np.subtract(y, y_))
    loss_array = np.sum(diff_y, axis=1)
    return np.median(loss_array) / 12.0


# In[18]:


ipca = joblib.load(PCA_MODEL_NAME)
print("\n\nGetting validation data...")
x_val, y_val = get_training_data(n_games=N_VALIDATION_GAMES)


# In[ ]:


# MLP Classifier
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.models import load_model

MODEL_NAME = "intellecto.hdf5"
batch_size = 16
num_classes = I.n_bubbles
epochs = 30
input_size = ipca.n_components
TRAIN_MODEL = True
droprate = 0.7

try:
    model = load_model(MODEL_NAME)
    print("Loaded saved model: " + MODEL_NAME)
except:
    print("Creating new model: " + MODEL_NAME)
    
    model = Sequential()
    model.add(Dense(units=1024, activation='relu', input_shape=(input_size, )))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))

    model.add(Dense(units=512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))
    
    model.add(Dense(units=512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 1.5))

    model.add(Dense(units=256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 1.5))
    
    model.add(Dense(units=256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 1.5))
    
    model.add(Dense(units=256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 1.5))

    model.add(Dense(units=128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 2))
    
    model.add(Dense(units=128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 2))
    
    model.add(Dense(units=128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 2))

    model.add(Dense(units=64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 2))
    
    model.add(Dense(units=64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 2.5))
    
    model.add(Dense(units=32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 3))
    
    model.add(Dense(units=32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 3))
    
    model.add(Dense(units=16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate / 3.5))

    model.add(Dense(num_classes, activation='sigmoid'))
    
    model.summary()
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

def do_on_epoch_end(epoch, _):
    if (epoch + 1) % 5 == 0:
        score = model.evaluate(x_val, y_val, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        y_val_ = model.predict(x_val)
        print("Ordering loss on val data ", ordering_loss(y_val, y_val_))
    

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
            validation_data=(x, y),
            callbacks = [
                ModelCheckpoint(MODEL_NAME, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1),
                LambdaCallback(on_epoch_end=do_on_epoch_end)
            ]
        )

