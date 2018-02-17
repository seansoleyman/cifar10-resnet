"""Constants and hyperparameters used by the cifar10 program. """

# Constants describing the CIFAR-10 data set.
IMAGE_SIZE = 32                     # Width and height of CIFAR-10 images. 
CHANNELS = 3                        # Number of color channels (RGB). 
NUM_CLASSES = 10                    # Number of CIFAR-10 classes. 
NUM_TRAIN_CIFAR10 = 50000           # Number of CIFAR-10 training instances. 
NUM_TRAIN_EXAMPLES = 50000          # Number of CIFAR-10 test instances. 
NUM_TEST_EXAMPLES = 10000           # Number of instances to use for training. 

# Constants describing the training process.
BATCH_SIZE = 128                    # Batch size. 
LR_BOUNDARIES = [400, 32000, 48000] # Learning rate boundaries.
LR_VALUES = [0.01, 0.1, 0.01, 0.001]# Learning rates.
MOMENTUM = 0.9                      # Momentum. 
TRAIN_STEPS = 65000                 # Number of steps to run. 
LOG_FREQUENCY = 1000                # How often to log results.

# Network hyperparameters
BN_MOMENTUM = 0.9                   # Decay rate for batch normalization.
SHORTCUT_L2_SCALE = 0.0001          # Regularization for the skip connections. 
DEPTH = 3                           # Residual units per stack. 
WIDEN_FACTOR = 1                    # Scale up the number of feature maps.

# Constants describing the input pipeline. 
SHUFFLE_BUFFER = NUM_TRAIN_EXAMPLES # Buffer size for the shuffled dataset.
NUM_THREADS = 6                     # Number of threads for image processing. 
OUTPUT_BUFFER_SIZE = BATCH_SIZE*2   # Buffer size for processed images. 
TRAIN_OUTPUT_BUFFER = SHUFFLE_BUFFER//BATCH_SIZE # Train buffer size. 
VALIDATION_OUTPUT_BUFFER = 6        # Buffer size for validation dataset.

# Data directory. 
DATA_DIR = r'/home/sean/Desktop/ml/datasets/cifar10'

# Directory where to write event logs and checkpoint.
TRAIN_DIR = r'train'
