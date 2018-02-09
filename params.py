"""Constants and hyperparameters used by the cifar10 program. """

# Constants describing the CIFAR-10 data set.
IMAGE_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 10
NUM_TRAIN_EXAMPLES = 50000
NUM_TEST_EXAMPLES = 10000

# Constants describing the training process.
BATCH_SIZE = 128                    # Batch size
MOVING_AVERAGE_DECAY = 0.9999       # The decay to use for the moving average.
LR_BOUNDARIES = [32000, 48000]      # Learning rate boundaries.
LR_VALUES = [0.1, 0.01, 0.001]      # Learning rates.
MOMENTUM = 0.9                      # Momentum
TRAIN_STEPS = 65000                 # Number of steps to run (K. He used 65000)
LOG_FREQUENCY = 1000                # How often to log results.

# Constants describing the input pipeline. 
TRAIN_BUFFER_SIZE = 50000           # Buffer size for the shuffled dataset.
NUM_THREADS = 8                     # Number of threads for image processing. 
OUTPUT_BUFFER_SIZE = BATCH_SIZE*2   # Buffer size for processed images. 

# Data directory. 
DATA_DIR = r'C:\Users\Sean Soleyman\Desktop\ml\datasets\cifar10'

# Directory where to write event logs and checkpoint.
TRAIN_DIR = r'.'
