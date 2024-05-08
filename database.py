from pymongo.mongo_client import MongoClient
import pandas as pd
import os
from src import config as cfg
import ast

if __name__ == '__main__':
    uri = "mongodb+srv://edvardas_timoscenka:H3i9Cil0NNMz6Fai@cluster0.ykcaicq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(uri)
    try:
        client.admin.command('ping')
        print("Pinged  deployment. Successfully connected to MongoDB!")
        for db_info in client.list_database_names():
            print(db_info)

        embed_chunks = pd.read_csv(os.path.join(cfg.DATA_DIR, 'chunked_articles_with_embeddings.csv'))
        embed_chunks['embedding'] = embed_chunks['embedding'].apply(ast.literal_eval)

        embed_chunks_dict = embed_chunks.to_dict(orient='records')
        db = client['LRT']
        collection = db['LRT']
        collection.insert_many(embed_chunks_dict)

    except Exception as e:
        print(e)