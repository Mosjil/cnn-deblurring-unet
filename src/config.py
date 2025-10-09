# Global parameters for all scripts

# Dataset
INPUT_DIR = "clean_images"
OUTPUT_DIR = "data/"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
IMG_SIZE = (512, 512)
NUM_BLURS_PER_IMAGE = 5
NOISE_STD_RANGE = (0.005, 0.02)

# Model
IN_CHANNELS = 3
OUT_CHANNELS = 3
BASE_CH = 64

# Training
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4

# Evaluation / Inference
CHECKPOINT_PATH = "results/checkpoints/best.pt"
RESULTS_DIR = "results/eval"
