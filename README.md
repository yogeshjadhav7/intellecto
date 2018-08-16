# intellecto

### Description
----
This repository houses the business logic of everything that is related to the Machine Learning model of each user of my android game **Intellecto** (https://play.google.com/store/apps/details?id=com.jadhav.yogesh.intellecto). This project plugs into my Django based Machine Learning microservice **ml-services** (https://github.com/yogeshjadhav7/ml-services) as a dependency.

##### Short description of the data flow
---
1. For each user, responses to game states are captured while they are playing **Challenge games** on their Android devices ( See **Intellecto FAQs** here: https://docs.google.com/document/d/1K4IS9hxqMkF1duPPZOr4CRmZoYZOYpmCZapQgljARxY ) and are stored to my AWS MySQL database periodically.

2. From the database, these game-state-responses pairs (training data) of each user are pushed (on-demand) to **ml-services** by backend Java REST service **restful-api-service-backend** (https://github.com/yogeshjadhav7/restful-api-service-backend). (on-demand) represents the logic in restful-api-service-backend to decide whether to spawn a process of training an ML model for that particular user. This logic consumes factors like the count of new game-state-response pairs for the user post previous training and the age of currently existing model for the user.

3. **ml-services** is responsible for creating, validating and scaling up the trained model of each user. It is also responsible for prediction of responses for unseen game-states, while simulating each user in **Challenge robot games** (Refer FAQs). **ml-services** uses this project to carry out all those ML tasks.


### Prerequisites (Optional but recommended)
---
> Install and setup pyenv. (Ubuntu 14.04)
```sh

# Install essentials
sudo apt-get install git python-pip make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev
sudo pip install virtualenvwrapper

# Clone the pyenv repository
git clone https://github.com/yyuu/pyenv.git ~/.pyenv
git clone https://github.com/yyuu/pyenv-virtualenvwrapper.git ~/.pyenv/plugins/pyenv-virtualenvwrapper

# Update ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'pyenv virtualenvwrapper' >> ~/.bashrc
source ~/.bashrc

# Install python 3.6.4 in pyenv
pyenv install 3.6.4

```

> Install and setup virtualenv using pyenv (python 3.6.4)
```sh

sudo pip install virtualenv

# mlenv is the name of virtualenv and python-envs is the directory that houses virtualenv on your system. You can change the names with corresponding changes below.
mkdir -p ~/python-envs/mlenv

# Create virtualenv
virtualenv -p ~/.pyenv/versions/3.6.4/bin/python3.6 ~/python-envs/mlenv

```

> Create an alias for activating virtualenv everytime easily.
```sh

# append this to your ~/.bashrc
alias mlenv="source ~/python-envs/mlenv/bin/activate"

# restart the bash
source ~/.bashrc

```

### Installation
----
> Clone the repository.
``` sh
git clone https://github.com/yogeshjadhav7/intellecto.git
```

> Go to the root directory of the project.
``` sh
cd intellecto/
```

> (Optional but recommended) Start the virtualenv by typing the alias name. Refer Prerequisites.
```sh
=> mlenv
```

> Install the requirements.
```sh
pip install -r requirements.txt
```

> Package and install the project in your python environment.
``` sh
python setup.py install
```

> Test the installation.
``` sh
 => python
>>> from intellecto import Intellecto
```

### Project Walkthrough
---
First, the reader should know few details of the problem that I am trying to solve. Here, the goal is to create a Machine Learning model which will predict the move (response) of a user to unprecendented game-states.


#### Use Case:
- Generic game state : [ B0, B1, B2, B3, B4, N ], where Bi and N represent the integer value from the set: {None, 0, [1 to 9], [-1 to -9]}

- (Bi)s are the bubbles in the game play area. A user may pop any one of the i bubbles which has no None value and gets the points equal to B(i) multiply B(i - 1) multiply B(i + 1) added to their score. If B(i - 1) and/or B(i + 1) has None value then they are replaced by 1 in the product equation.

- N is the next bubble. It is the head of the NEXT QUEUE (Refer FAQs) which comes in place of Bi on popping ith bubble in the game play area. And the whole queue (not shown here) to the right of N gets shifted to the left and game gets a new N (if queue is not empty).

- Generic response : i # the bubble which user had popped when encountered the above state while playing Challenge games.

####  Problem Statement:

- **I am trying to generalize the response for game-states by creating ML models for each user using their respective game-state-responses data.**

- So as to simulate each users gameplay and users can play Challenge games against their own / friends robot (who mimics their characterists in making decisions to given game states). Reminds me of good old days of 90s where beating CPU in video games was an achievement! This, with a twist, the twist of ML. Sounds cool, doesnt it ?!
---

### **intellecto/intellecto.py**
- The heart and soul of the project, this file houses the subroutines to carry out tasks ranging from simulations of the game play of Intellecto to training and validating Machine Learning models & carrying out predictions. Currently, Intellecto uses **sklearns RandomForest** algorithm to build and train the models.

    > **There could be many reasons why user chose a particular response on a game-state. These unexplained reasons form characteristics of that user. Thus, as responses of users sometimes can never have any logically pattern or explanation, I felt ensemble technique is the way to go and hence RandomForest**.

    **NOTE:** I am trying to solve this problem using deep learning as well. That is still in BETA state and can replace RandomForest on the production environment in near future. **Boolean flag one_hot is used throughout this project to distinguish between Sklearns RandomForest approach and Tensorflow-Keras driven deep learning approach**.

- **def train**(self, behaviourList, one_hot=True, verbose=0, auto_correct_ratio=0.1, auto_correction_degree=0.5):
This routine is called from **ml-services** to train the RandomForest model for a particular user using behaviourList of that user.

- **def cross_validate_and_fit**(self, x, y, verbose=0):
This subroutine of def train: helps in cross validating and finally returning the best fitted model (in terms of validation score) for the training data passed (x, y). As RandomForest is prone to overfitting, cross validating helps in generalizing better to unprecented game-states.

-  **def _get_cv_parameters_range**(self, n_trees, prev_best_parameters=None, prev_parameters_config=None):
This interesting subroutine helps in narrowing down the parameters of **GridSearchCV** using prev_best_parameters. This subroutine gets called till cross_validate_and_fit gets one best value for each parameter.

- **def predict**(self, behaviourMap, saved_model, one_hot=True):
This routine is called from **ml-services** to fetch predictions for the game-states in behaviourMap object passed.

- **def play_episode**(self, n_games=25, queue_size=None, difficulty=None):
This subroutine is used for simulating game play of Intellecto using **difficulty** as a means of choosing a response from the options of the responses for a given game-state.

    > This routine is used by **challengesimulator.py** to guage the quality of ML model for a user.

    For example, difficulty = 0 enforces simulator to choose the best response, difficulty = 1 enforces simulator to choose 2nd best response, and so on. More on this while describing challengesimulator.py
---
### **intellecto/challengesimulator.py**
- **def simulate_challenge_games**(self, model, ipca, predict_method=None, games_per_difficulty=100, queue_size=None, verbose=False):
This routine simulates **games_per_difficulty** number of games for each difficulty. It appends the performance of the **model** per difficulty in **win_ratio_per_difficulties** list. It also returns the total average performance of the **model** accross all difficulties in **win_ratio_mean**.

    > Through challengesimulator.py, we have now a **win_ratio_mean** (performance metric) to judge how good a ML model is in playing Challenge games. And as model is trained from its user itself, we can sense how better or worse a particular user has been playing Challenge games lately by comaparing previous **win_ratio_mean** data with the latest one. There is an impending feature update in Intellecto app which will use challegesimulator.py to notify the user of their performance fluctuations in Challenge games by pitching performance graph of their model against performance graphs of their friends. This will help to bring some competitiveness amoung users making them to play more.

- **def simulate_challenge_game**(self, model, ipca, queue_size, predict_method=None, difficulty=None, verbose=False):
This subroutine simulates the game play of Challenge robot game for specified difficulty. The scores of players and outcome of the game are returned to the caller routine **simulate_challenge_games**;

> **intellecto/{other files}**
These files are related to the Tensorflow-Keras driven deep learning approach to build model for each user. Once this approach become production ready, I will update this README.






