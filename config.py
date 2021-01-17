import transformers
import torch

TOKENIZER = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = 160
BATCH_SIZE = 64
EPOCHS = 100
RANDOM_SEED = 42
LOAD_MODEL = False
BERT_NAME = "bert-base-cased"
DEVICE = torch.device("cuda")