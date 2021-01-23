import transformers
import torch

TOKENIZER = transformers.BertTokenizer.from_pretrained("bert-base-cased")
MAX_LEN = 160
BATCH_SIZE = 64
EPOCHS = 100
RANDOM_SEED = 42
LOAD_MODEL = True
BERT_NAME = "bert-base-cased"
DEVICE = torch.device("cuda")