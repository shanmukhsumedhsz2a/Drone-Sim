# Drone-Sim
A Simple 2D Target Tracking Drone Control Environment with a Controller Trained via Vanilla Genetic Algorithm and Custom Neural Network Implentation

Environment Description
- A 2D Drone controlled by 2 rockets on each side whose goal is to reach a target marked by a red circle and hover there for a fixed anount of time before receiving new targets
- Uses Python turtle animations and numerical timestep integration physics to model newtonian motion
- Motion is controlled via rocket power of each side

Genetic Algorithm
- Genetic algorithm consisitingo of mutation, elitism , crossover and new individual addition every generation
- Elitism - Transfers best perfomring indiviuals to the next generation as is
- Mutation - Mutated version of the best individuals are added to next generation
- CrossOver - Intermixed versions of best individual are added to next generation via random gene selection
- NewIndividual - Adds new random Individual to next generation to introduce completely new genomes

- Algorithm also tracks and graphs the avergae fitness of each class of individuals as well as tracks the best fitness, average fitness of best 10 individuals and also overall average fitness
- it also allows for checkpoints to pause training and change paremters in the algorithm or environment

Neural Networks
- Vanilla neural network library implemented using numpy
- made for modularity and customization with various activation functions
- does not include backpropogation

Training
- was done using curiculum learning enabled by checkpoints in the genetic algorithm
- Reward structure changed as follows
  1. Initially rewards were based solely on hovering time so as to teachit basic controls
  2. next rewards based on motion towards the target were given to incentivize target seeking
  3. next rewards based on motion towards target were gradually reduced and replaced by reward due to hovering at target
  4. finally rewards also included a time factor allowing for more dynamic and quick motions toward the target
 
- Fully Trained model can be used by running test.py

