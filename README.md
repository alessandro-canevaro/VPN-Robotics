# VPN_Robotics

# Folders structure

├── README.md          <- The top-level README for developers using this project.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── log                <- Training logs for tensorboard
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
├── setup.py           <- Make this project pip installable
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── environment    <- Environment related files
│   │   ├── env_register.py
│   │   └── gridworld.py
│   │
│   └── models         <- Scripts to train models and then use trained models to make
│       │                 predictions
│       ├── VPN_basic.py
│       ├── env_test_keyboard.py
│       ├── config.py
│       ├── callbacks.py
│       ├── train.py
│       └── test.py
│   
├── test               <- Unit tests
│   └── test_gridworld.py
│ 
└── commands.txt       <- Useful terminal commands.

## Project checklist

* [ ] Create a git repository
* [ ] Make sure that all team members have write access to the github repository
* [ ] Create a dedicated environment for you project to keep track of your packages
* [ ] Create the initial file structure using cookiecutter
* [ ] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project
* [ ] Do a bit of code typing and remember to document essential parts of your code
* [ ] Write one or multiple configurations files for your experiments
* [ ] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [ ] Write unit tests related to the data part of your code
* [ ] Write unit tests related to model construction and or model training
* [ ] Calculate the coverage.
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training