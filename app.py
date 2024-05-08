from RAG_LLM_pipeline import RAG_LLM_pipeline
import gradio as gr
import pandas as pd
import os
from src import config as cfg
import ast

df = pd.read_csv(os.path.join(cfg.DATA_DIR, 'chunked_articles_with_embeddings.csv'))
df['embedding'] = df['embedding'].apply(lambda x: ast.literal_eval(x))


def gradio_interface(input_text: str) -> str:
    return RAG_LLM_pipeline(input_text, df)


iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs="text",
    title="Retrieval and Answer Generation System",
    description="Enter a query to retrieve and generate answers."
)

if __name__ == "__main__":
    iface.launch()