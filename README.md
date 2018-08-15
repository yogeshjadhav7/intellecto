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
mkdir -p ~/env/python-envs/mlenv

# Create virtualenv
virtualenv -p ~/.pyenv/versions/3.6.4/bin/python3.6 ~/python-envs/mlenv

```

> Create an alias for activating virtualenv easy.
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

>> Test the installation.
``` sh
 => python
>>> from intellecto import Intellecto
```