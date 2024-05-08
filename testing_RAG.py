import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import pandas as pd
import os
from src import config as cfg
import torch
import ast
import numpy as np

def retrieve_data(query: str,
                  embeddings: torch.Tensor,
                  vector_index: faiss.IndexFlatIP,
                  model: SentenceTransformer,
                  k: int = 25) -> tuple:

    query_embedded = model.encode(query,
                                  convert_to_tensor=True)

    D, I = vector_index.search(query_embedded.to('cpu'), k)

    return D, I

def rerank_data(query: str,
                rerank_model: SentenceTransformer,
                I: list,
                df: pd.DataFrame,
                k: int = 5) -> tuple:


    df = df.to_dict(orient='records')

    documents = [df[i]['chunk'] for i in I[0]]
    query = ' '.join(query)
    results = rerank_model.rank(query,
                                documents,
                                top_k=k)
    return results

def print_results_before_rerank(I: list,
                  df: pd.DataFrame) -> None:

    df = df.to_dict(orient='records')
    for i in I[0][:6]:
        print(f'Index: {i}\nTitle: {df[i]["title"]}\nChunk: {df[i]["chunk"]}\nURL: {df[i]["url"]}\n\n')

def print_results_after_rerank(results: list,
                               I: list,
                               df: pd.DataFrame) -> None:

    df = df.to_dict(orient='records')
    for i, result in enumerate(results):
        old_rank = result['corpus_id']
        new_id = i
        data = df[I[0][old_rank]]
        print(f'New Rank: {new_id}\tOld Rank: {old_rank}')
        print(f'Title: {data["title"]}\nChunk: {data["chunk"]}\nURL: {data["url"]}\n\n')


if __name__ == '__main__':
    query = input("Enter a query: ")
    query = [query]

    df = pd.read_csv(os.path.join(cfg.DATA_DIR, 'chunked_articles_with_embeddings.csv'))
    df['embedding'] = df['embedding'].apply(lambda x: ast.literal_eval(x))
    embeddings = torch.tensor(np.array(df['embedding'].tolist())).to('cuda', dtype=torch.float32)

    vector_index = faiss.read_index(os.path.join(cfg.DATA_DIR, 'faiss_index.bin'))
    print('Total indexes', vector_index.ntotal)

    model = SentenceTransformer(cfg.EMBEDDING_MODEL,
                                device='cuda')

    rerank_model = CrossEncoder(cfg.RERANK_MODEL,
                                      device='cuda')

    D, I = retrieve_data(query=query,
                         embeddings=embeddings,
                         vector_index=vector_index,
                         model=model,
                         k=25)

    print_results_before_rerank(I=I,
                  df=df)
    results = rerank_data(query=query,
                      rerank_model=rerank_model,
                      I=I,
                      df=df,
                      k=5)

    print('-----------------------------------')

    print_results_after_rerank(results=results,
                                 I=I,
                                 df=df)