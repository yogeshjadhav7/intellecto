
# coding: utf-8

# In[4]:


from intellecto import Intellecto
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib

N_GAMES_PER_EPISODE = 50
N_EPISODES = 4000
I = Intellecto()
batch_size = N_GAMES_PER_EPISODE * (I.n_bubbles + I.queue_size)

n_input = (I.n_bubbles + 1) * (I.n_bubbles + 2)
n_output = I.n_bubbles

PCA_DIMENSION = 12
PCA_MODEL_NAME = "pca.model"


# In[6]:


def do_pca():
    try:
        ipca = joblib.load(PCA_MODEL_NAME)
        print("Loaded saved pca model. Incrementally training the model..")
    except:
        print("Could not load the pca model. Creating a new one...")
        ipca = IncrementalPCA(n_components=PCA_DIMENSION, batch_size=batch_size)
    
    full_to_empty_games_ratio = np.int(1 + (I.queue_size / I.n_bubbles))
    for n_episode in range(N_EPISODES):
        print("Fitting data of episode " + str(n_episode), " Progress: " + str(n_episode * 100.0 / N_EPISODES))
        x,_ = I.play_episode(n_games=N_GAMES_PER_EPISODE)
        ipca.partial_fit(x)
        
        for n_empty_queue_game in range(full_to_empty_games_ratio):
            x,_ = I.play_episode(n_games=N_GAMES_PER_EPISODE, queue_size=0)
            ipca.partial_fit(x)

        if n_episode > 0 and n_episode % 100 == 0: joblib.dump(ipca, PCA_MODEL_NAME, protocol=2)
    
    print("Saving the pca model...")
    joblib.dump(ipca, PCA_MODEL_NAME, protocol=2)
    return ipca


# In[7]:

def start_pca():
    ipca = do_pca()
    variances_ = ipca.explained_variance_ratio_.cumsum()
    print(variances_)

