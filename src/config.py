import os
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

MAIN_URL = ["https://www.lrt.lt/en/tag/baltics-and-eastern-europe",
            "https://www.lrt.lt/en/tag/economy",
            "https://www.lrt.lt/en/tag/society",
            "https://www.lrt.lt/en/tag/features",
            "https://www.lrt.lt/en/tag/News"]

EMBEDDING_MODEL = 'all-mpnet-base-v2'
RERANK_MODEL = 'mixedbread-ai/mxbai-rerank-large-v1'
LLM_MODEL = 'google/gemma-7b-it'

TOKEN = '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'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')