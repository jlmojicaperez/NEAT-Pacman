# NEAT-Pacman
Neuroevolution of Augmented Topologies (NEAT) applied to Pacman 
AI for Artificial worlds project by:
 - [Jose Mojica Perez](https://github.com/J-Mojica)
 - [Oscar Tejax](https://github.com/OskarGhost999)

Read [our blog post](https://neatpacman.wordpress.com/2023/05/03/neat-pacman/) for an explanation of the project.

# How to run our code

The code was developed and tested with Python 3.11.3 on Ubuntu 22.04.2 LTS for WSL 2.
We recommend a Python version 3.9 or higher, although any Python version that's 3.6 or higher will most likely work.
As for the OS, the code should run on any recent Windows, Linux or MacOS version

## Cloning the repository and installing the requirements
```
git clone https://github.com/J-Mojica/NEAT-Pacman.git
cd NEAT-Pacman
pip install -r requirements.txt
```

## Run a game with the best agent

The best agent is saved as `bestPacmanNEAT.pkl`. Please note that running training a new population 
or continuing to train from a checkpoint file will overwrite
this `pkl` file once it reaches the maximum number of generations, there's complete extinction or if the fitness criterion
defined in the configureation file is met.
```
python pacmanNEAT.py test [configuration file]
```

### Example: 
```
python pacmanNEAT.py test pacmanConfigNEAT.txt
```

## Start a training a new population from scratch
Please note that running training a new population will overwrite the  `bestPacmanNEAT.pkl` 
file once it reaches the maximum number of generations, there's complete extinction or if the fitness criterion
defined in the configuration file is met. By default it will generate a checkpoint file every 10 generations.
To change this, the code in `pacmanNEAT.py` must be modified.
```
python pacmanNEAT.py start [configuration file] [max number of generations]
```

### Example: 
```
python pacmanNEAT.py start pacmanConfigNEAT.txt 10
```

## Continue training a population from a checkpoint
Please note that continuing training a population from a checkpoint will overwrite the  `bestPacmanNEAT.pkl` 
file once it reaches the maximum number of generations, there's complete extinction or if the fitness criterion
defined in the configuration file is met. By default it will generate a checkpoint file every 10 generations.
To change this, the code in `pacmanNEAT.py` must be modified.

```
python pacmanNEAT.py continue [configuration file] [checkpoint file] [max number of generations]
```

### Example:
```
python pacmanNEAT.py continue pacmanConfigNEAT.txt neat-checkpoint-183 10
```
