
# coding: utf-8

# In[1]:


from intellecto import Intellecto
import numpy as np
from sklearn.externals import joblib
from challengesimulator import ChallengeSimulator
import gc
from keras import backend as K

N_GAMES_PER_EPISODE = 2500
N_EPISODES = 30
EPISODE_CHECKPOINT_FREQ = 5

I = Intellecto()
simulator = ChallengeSimulator()
batch_size = N_GAMES_PER_EPISODE * (I.n_bubbles + I.queue_size)

n_input = (I.n_bubbles + 1) * (I.n_bubbles + 2) 
n_output = I.n_bubbles

PCA_MODEL_NAME = "pca.model"
simulation_records = []


# In[2]:


def get_training_data(n_games=N_GAMES_PER_EPISODE, one_hot=True):
    f, l, boards, queues, raw_moves_score = I.play_episode(n_games=n_games)
    if one_hot: l = np.argmax(l, axis=1)
    return f, l


# In[3]:


def ordering_loss(ground_truth, predictions):
    y = np.array(ground_truth)
    y_ = np.array(predictions)
    y = np.argsort(y, axis=1)
    y_ = np.argsort(y_, axis=1)
    diff_y = np.equal(y, y_)
    loss_array = np.sum(diff_y, axis=1)
    return np.mean(loss_array)


# In[4]:


ipca = None #joblib.load(PCA_MODEL_NAME)


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
from keras.metrics import mean_absolute_error, categorical_crossentropy

MODEL_NAME = "intellecto.hdf5"
batch_size = 16 #I.n_bubbles
num_classes = I.n_bubbles
epochs = 10
input_size = n_input #ipca.n_components
TRAIN_MODEL = False

droprate = 0.6
activation = 'elu'

def getmodel():
    try:
        model = load_model(MODEL_NAME)
        #print("Loaded saved model: " + MODEL_NAME)
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

        model.add(Dense(units=64, activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(droprate / 3))

        model.add(Dense(units=32, activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(droprate / 3))

        model.add(Dense(units=16, activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(droprate / 3.5))

        model.add(Dense(num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    
    return model


def validation_model(games_per_difficulty=100):
        x_val, y_val = get_training_data(n_games=100, one_hot=False)
        saved_model = load_model(MODEL_NAME)
        orderingloss = ordering_loss(ground_truth=y_val, predictions=saved_model.predict(x_val))
        win_ratio_mean, win_ratio_per_difficulties = simulator.simulate_challenge_games(model=saved_model, ipca=ipca, 
                                                                                        games_per_difficulty=games_per_difficulty)
        
        return win_ratio_mean, win_ratio_per_difficulties, orderingloss
        
        
def do_on_epoch_end(epoch, _):
    if (epoch + 1) == epochs:
        win_ratio_mean, win_ratio_per_difficulties, orderingloss = validation_model()
        print("\n\nWin ratio per difficulties", win_ratio_per_difficulties)
        print("Win ratio mean", win_ratio_mean)
        print("Ordering loss", orderingloss)
        simulation_records.append(win_ratio_mean)
        
        
if TRAIN_MODEL:
    for n_episodes in range(N_EPISODES):
        x, y = get_training_data(one_hot=False)
        model = getmodel()
        print("\n\n\nTraining on episode #" + str(n_episodes + 1))
        model.fit(
            x, 
            y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(x, y),
            callbacks = [
                ModelCheckpoint(MODEL_NAME, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1),
                LambdaCallback(on_epoch_end=do_on_epoch_end)
            ]
        )
        
        if (n_episodes + 1) % EPISODE_CHECKPOINT_FREQ == 0:
            print("\nCurrent mean win ratio overall", np.mean(simulation_records))
            I.plot(y_data=simulation_records, y_label="win_ratio_mean", window=EPISODE_CHECKPOINT_FREQ)
            
        model = None
        gc.collect()
        K.clear_session()
            
    print("\nFinal mean win ratio overall", np.mean(simulation_records))
    
else:
    
    for n_val in range(10):
        print("\n\nValidation Run #" + str(n_val + 1))
        win_ratio_mean, win_ratio_per_difficulties, orderingloss = validation_model()
        print("Win ratio per difficulties", win_ratio_per_difficulties)
        print("Win ratio mean", win_ratio_mean)
        print("Ordering loss", orderingloss)
        

