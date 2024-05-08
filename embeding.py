from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from src import config as cfg
import torch
import faiss


def create_embedings(df: pd.DataFrame) -> torch.Tensor:
    model = SentenceTransformer(cfg.EMBEDDING_MODEL,
                                device='cuda',
                                )

    embeddings = model.encode(df['chunk'].to_numpy(),  # Is there an effect of using tensor instead of numpy? Need to check
                              batch_size=32,
                              show_progress_bar=True,
                              convert_to_tensor=True)

    return embeddings


def create_vector_index(embeddings: torch.Tensor) -> None:
    # Faiss version is for CPU, thus we transfer out embeddings to CPU
    embeddings = embeddings.to('cpu', dtype=torch.float32)

    d = embeddings.shape[1]
    nlist = 50
    m = 8
    bits = 8

    assert d % m == 0 # d must be a multiple of m

    quantizer = faiss.IndexFlatIP(d) # Because our embeddings are normalized, this performs cosine similarity
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)

    if not index.is_trained:
        index.train(embeddings)

    assert index.is_trained
    index.add(embeddings)

    faiss.write_index(index, os.path.join(cfg.DATA_DIR, 'faiss_index.bin'))

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(cfg.DATA_DIR, 'chunked_articles.csv'))
    embeddings = create_embedings(df)

    df['embedding'] = embeddings.tolist()
    df.to_csv(os.path.join(cfg.DATA_DIR, 'chunked_articles_with_embeddings.csv'), index=False)
    print("Embeddings created and saved to disk!")

    create_vector_index(embeddings)
    print("Vector index created and saved to disk!")




