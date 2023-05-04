from game import *
from featureExtractors import *


import pacman
import neat
import os
import pickle

import random,util,math
import copy

from neat import parallel


####################
# NEAT Agent Class #
####################

class NEATAgent(Agent):
    def __init__(self, genome, config, **args):

        self.genome = genome
        self.config = config
        self.actions_map = [
            Directions.NORTH,
            Directions.SOUTH,
            Directions.EAST, 
            Directions.WEST
            ]
        
        self.network = neat.nn.FeedForwardNetwork.create(genome, config)

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # HERE IS WHERE THE NEURAL NETWORK IS CALLED
        inputs = self.getInputs(state, legal)
        outputs = self.network.activate(inputs)

        #print(outputs)
        action = self.actions_map[outputs.index(max(outputs))]
        if action not in legal:
            self.genome.fitness -= 2
            action = random.choice(legal)
        
        return action

    def getInputs(self, state, legal) -> tuple:
        '''Defines the inputs to the neural network
        based on the current state of the game'''

        #REMEMBER: The neat config file defines the number of inputs
        inputs = []

        # Pacman's Position
        x, y = state.getPacmanPosition()
        inputs.append(x)
        inputs.append(y)

        # Relative position of each ghost
        for x1, y1 in state.getGhostPositions():
            relative_x = x - x1
            relative_y = y - y1
            inputs.append(relative_x)
            inputs.append(relative_y)
        
        # Indicators if ghost is scared
        for ghost in state.getGhostStates():
            if ghost.scaredTimer > 0:
                inputs.append(1)
            else:
                inputs.append(0)

        # Relative position of nearest food
        x1, y1 = self.getPositionNearestFood(state)
        relative_x = x - x1
        relative_y = y - y1
        inputs.append(relative_x)
        inputs.append(relative_y)


        # Relative position of nearest capsule
        x1, y1 = self.getPositionNearestCapsule(state)
        relative_x = x - x1
        relative_y = y - y1
        inputs.append(relative_x)
        inputs.append(relative_y)


        # Indicators of legal actions
        actions_one_hot = [0, 0, 0, 0]
        for i, action in enumerate(legal):
            if action == Directions.NORTH:
                actions_one_hot[0] = 1
            elif action == Directions.SOUTH:
                actions_one_hot[1] = 1
            elif action == Directions.EAST:
                actions_one_hot[2] = 1
            elif action == Directions.WEST:
                actions_one_hot[3] = 1
    
        inputs.extend(actions_one_hot)
        
        
        return tuple(inputs)

    
    # TODO: Change so that distance is the actual number of actions required to get to the nearest food/capsule
    
    def getPositionNearestCapsule(self, state):
        '''Returns the position of the nearest capsule'''
        x, y = state.getPacmanPosition()
        capsules = state.getCapsules()
        min_dist = float('inf')
        min_x = 0
        min_y = 0
        for i, j in capsules:
            dist = manhattanDistance((x, y), (i, j))
            if dist < min_dist:
                min_dist = dist
                min_x = i
                min_y = j
        return min_x, min_y
    
    def getPositionNearestFood(self, state):
        '''Returns the position of the nearest food pellet'''
        x, y = state.getPacmanPosition()
        food = state.getFood()
        min_dist = float('inf')
        min_x = 0
        min_y = 0
        for i in range(food.width):
            for j in range(food.height):
                if food[i][j] == True:
                    dist = manhattanDistance((x, y), (i, j))
                    if dist < min_dist:
                        min_dist = dist
                        min_x = i
                        min_y = j
        return min_x, min_y

##################
# NEAT Functions #
##################


def continueNEAT(configFile, checkpointFile, maxGenerations):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, configFile)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
    population = continuePopulation(checkpointFile)

    bestGenome = population.run(evalGenomes, maxGenerations)

    # Save the best genome
    with open('bestPacmanNEAT.pkl', 'wb') as output:
        pickle.dump(bestGenome, output)

    testNEATAgent(configFile)

def continuePopulation(checkpoint_file):
    population = neat.Checkpointer.restore_checkpoint(checkpoint_file)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(10))
    return population

def startNEAT(configFile, max_generations):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, configFile)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
    population = initializePopulation(config)
    best = population.run(evalGenomes, max_generations)

    # Save the best genome
    with open('bestPacmanNEAT.pkl', 'wb') as output:
        pickle.dump(best, output, 1)

    testNEATAgent(configFile)

def initializePopulation(config):
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(10))
    return population

def evalGenomes(genomes, config):
    # Change to use the number of cores on the machine
    # My machine has 16 cores
    parallel.ParallelEvaluator(16, evalGenome).evaluate(genomes, config)


    # for genome_id, genome in genomes:
    #    genome.fitness = 0.0
    #    trainNEATAgent(genome, config)

def evalGenome(genome, config):
    
    genome.fitness = 0.0
    for _ in range(3):
        genome.fitness += float(trainNEATAgent(genome, config))
    genome.fitness = genome.fitness / 3.0
    return genome.fitness

def trainNEATAgent(genome, config):
        
        # Tricking the game to set the arguments for us as if
        # we are running pacman.py
        preset_args = ['-p', 'GreedyAgent', '-l', 'originalClassic', '-q']
        args = pacman.readCommand(preset_args)  # Get game components based on input

        # Initialize rules
        rules = pacman.ClassicGameRules(args["timeout"])

        # Set game components for NEAT
        args = getNewGameArgs(args, genome, config)
        args["quiet"] = True # Don't print output
        
        # Run game
        game = rules.newGame(**args)
        game.run()

        # Calculate the fitness
        return (rules.initialState.getNumFood() - game.state.getNumFood())

def testNEATAgent(configFile):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            configFile)
        
        # Load the best genome
        with open('bestPacmanNEAT.pkl', 'rb') as input_file:
            genome = pickle.load(input_file)

        # Tricking the game to set the arguments for us as if
        # we are running pacman.py
        preset_args = ['-p', 'GreedyAgent', '-l', 'originalClassic']
        args = pacman.readCommand(preset_args)  # Get game components based on input

        # Initialize rules
        rules = pacman.ClassicGameRules(args["timeout"])

        # Set game components for NEAT
        args = getNewGameArgs(args, genome, config)
        args["quiet"] = False # Print output

        # Run game
        game = rules.newGame(**args)
        game.run()

def getNewGameArgs(args, genome, config):
    new_game_args = dict()
    new_game_args["pacmanAgent"] = NEATAgent(genome, config)
    new_game_args["layout"] = args["layout"]
    new_game_args["ghostAgents"] = args["ghosts"]
    new_game_args["display"] = args["display"]
    new_game_args["horizon"] = args["horizon"]
    return new_game_args

def getMainArgs(argv):
    args = dict()
    if(argv[0] == "start"):
        args["type"] = "start"
        if(len(argv) < 3):
            print("To start NEAT, please enter the config file and maximum number of generations to run NEAT for")
        args["config_file"] = argv[1]
        args["max_generations"] = int(argv[2])
    elif(argv[0] == "continue"):
        args["type"] = "continue"
        if(len(argv) < 4):
            print("To continue NEAT, please enter the config file, checkpoint file, and maximum number of generations to run NEAT for")
        args["config_file"] = argv[1]
        args["checkpoint_file"] = argv[2]
        args["max_generations"] = int(argv[3])
    elif(argv[0] == "test"):
        args["type"] = "test"
        if(len(argv) < 2):
            print("To test NEAT, please enter the config file and checkpoint file")
        args["config_file"] = argv[1]
    else:
        print("Please enter either 'start' or 'continue' to start or continue NEAT")
    return args


if(__name__ == "__main__"):

    if(len(sys.argv) < 3):
        print("Please enter either 'start', 'continue', or 'test' followed by the appropriate arguments")
    else:
        args = getMainArgs(sys.argv[1:])
    
    if(args["type"] == "start"):
        startNEAT(args["config_file"], args["max_generations"])
    elif(args["type"] == "continue"):
        continueNEAT(args["config_file"], args["checkpoint_file"], args["max_generations"])
    elif(args["type"] == "test"):
        testNEATAgent(args["config_file"])
    

    
