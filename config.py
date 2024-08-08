PATH = "/workspace/a03-sgoel/PPI-CLIP"
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

BATCH_SIZE = 4
EPOCHS = 5
LR = 3e-7
LOSS_TYPE = 'bce_avg'  # Options: contrastive, cosine, bce_avg

MLP_LAYERS = 2
MLP_DROPOUT = 0.5
TRANSFORMER_LAYERS = 10
TRANSFORMER_HEADS = 10
TRANSFORMER_DROPOUT = 0.7
OUTPUT_RELU = True
INIT_TEMP = 0.07  # Learnable temperature parameter based on CLIP paper

TRAIN_CSV = PATH + "/data/train.csv"
TEST_CSV = PATH + "/data/test.csv"
VAL_CSV = PATH + "/data/val.csv"
POS_EMBED_PKL = PATH + "/data/pos_embeddings.pkl"
NEG_EMBED_PKL = PATH + "/data/neg_embeddings.pkl"


CKPT_DIR = PATH + "/models"