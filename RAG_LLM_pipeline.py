import os
import torch
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
import pandas as pd
import numpy as np
from src import config as cfg
import ast
import re

class Retrieval():
    def __init__(self,
                 vector_index_path: str,
                 embedding_model_name: str,
                 rerank_model_name: str,
                 device: str) -> None:
        """
        Class used for the retrieval and reranking of documents based on a given query/prompt.

        :param vector_index_path: path were the faiss index is stored
        :param embedding_model_name: sentence transformer model name
        :param rerank_model_name: cross encoder model name
        :param device: cuda or cpu
        """
        self.vector_index = faiss.read_index(vector_index_path)

        self.embedding_model = SentenceTransformer(embedding_model_name,
                                                   device=device)
        self.rerank_model = CrossEncoder(rerank_model_name,
                                                device=device)

    def embed_query(self,
                    query: str) -> torch.Tensor:
        """
        Embeds the query using the sentence transformer model.
        IMPORTANT: the embedding model should be the same as the one used to create the faiss index.
        """
        query = [query]
        query_embedded = self.embedding_model.encode(query,
                                                     convert_to_tensor=True)
        return query_embedded

    def initial_search(self,
                       query: str,
                       k: int = 25) -> tuple:
        """
        Perform an initial search using the faiss index.
        """
        query_embedded = self.embed_query(query)
        D, I = self.vector_index.search(query_embedded.to('cpu'), k)
        return D, I

    def rerank_search(self,
                      query: str,
                      I: list,
                      df: pd.DataFrame,
                      k: int = 5) -> tuple:
        """
        Rerank the initial search using the cross encoder model, for better results for that specific query/prompt.
        """
        df = df.to_dict(orient='records')
        documents = [df[i]['chunk'] for i in I[0]]

        results = self.rerank_model.rank(query,
                                         documents,
                                         top_k=k)
        return results

    def run_rag(self,
                query: str,
                df: pd.DataFrame) -> None:
        """
        Maind function that is used to run the retrieval and reranking process.
        """
        D, I = self.initial_search(query)

        results = self.rerank_search(query, I, df)

        df = df.to_dict(orient='records')
        rag_index_dict = {}
        for i, result in enumerate(results):
            old_rank = result['corpus_id']
            new_id = i
            data = df[I[0][old_rank]]
            rag_index_dict[new_id] = (old_rank, data)

        return rag_index_dict

class LLM():
    def __init__(self,
                 model_name: str,
                 access_token: str,
                 attention_implementation: str,
                 device: str) -> None:
        """
        Class used for the generation of answers based on a given query/prompt and context.
        Class used Gemma model(s) from the huggingface library.
        Attention_implentation can be either 'sdpa' or 'flash_attention_2'.
        Attention_implentation is not used in this implementation, due to difficulty getting to run
        flass_attention_2 on windows.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name,
                                                        token = access_token)

        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                               token = access_token,
                                               torch_dtype=torch.float16,
                                               low_cpu_mem_usage=False).to(device)
        # Chat template taken from the documentation of the Gemma model
        self.chat_template = [{
            'role': 'user',
            'content': None
        }]
        self.device = device

        self.base_prompt = """"After reviewing the provided news context items formulate an answer. Before answering the query extract essential information relevant to the query. Return only the final answer, omitting the intermediate thought processes. Your answers should be concise, comprehensive, and no longer than 256 tokens, following the style illustrated in these examples:
        \nExample 1:
        Query: What was the outcome of the recent presidential election in Lithuania?
        Answer: The recent presidential election in Lithuania resulted in the re-election of the incumbent president, who secured a second term by a wide margin, reflecting strong public approval. The election saw high voter turnout, underscoring significant civic participation.
        \nExample 2:
        Query: How is Lithuania addressing the issue of energy dependence on Russia?
        Answer: Lithuania is reducing its energy dependence on Russia by diversifying energy sources, including developing a national LNG terminal and increasing renewable energy projects like solar and wind, aligning with EU goals for energy independence.
        \nExample 3:
        Query: What are the latest developments in Lithuania's approach to educational reform?
        Answer: Lithuania's educational reforms focus on enhancing digital literacy, modernizing curricula, and improving teacher salaries. New policies promote technology in education, STEM subjects, and continuous professional development, preparing students for a digital future.
        Given the context and query below, produce a comprehensive yet succinct answer:
        Context: {context}
        \nRelevant Passages: <Extract essential information here>
        Query: {query}
        Answer:"""

    def format_prompt(self,
                      prompt: str,
                      context: list[dict]) -> None:
        """
        Format the prompt using the base prompt and the given context.
        """
        context = "- " + "\n- ".join([c['chunk'] for c in context])
        prompt = self.base_prompt.format(context=context,
                                            query=prompt)

        self.chat_template[0]['content'] = prompt

    def tokenize_prompt(self) -> torch.Tensor:
        """
        Tokenize the prompt using the gemma tokenizer.
        """
        prompt = self.tokenizer.apply_chat_template(conversation=self.chat_template,
                                                            tokenize=False,
                                                            add_generation_prompt=True)
        tokenized_prompt = self.tokenizer(prompt,
                                        return_tensors='pt').to(self.device)

        # Print statement used for logging information in regards to the token count of the prompt.
        print(f"Prompt token count: {tokenized_prompt['input_ids'].shape[1]}")
        return tokenized_prompt

    def generate_output(self,
                        tokenized_prompt: torch.Tensor,
                        max_tokens: int = 256) -> str:
        """
        Generate the output based on the tokenized prompt.
        There is a max token limit of 256 tokens, for more consistent and concise answers.
        """

        outputs = self.model.generate(**tokenized_prompt,
                                      temperature=0.9,
                                      do_sample = True,
                                      max_new_tokens = max_tokens)
        decoded_outputs = self.tokenizer.decode(outputs[0])
        return decoded_outputs

def pretify_output(output: str) -> str:
    """
    Function used to pretify the output, by taking only the start_of_turn and end_of_turn tokens of the model output.
    """
    pattern = r"<start_of_turn>model\s+(.*?)<eos>"
    return re.search(pattern, output, re.DOTALL).group(1).strip()

def RAG_LLM_pipeline(query: str,
                     df: pd.DataFrame) -> str:
    """
    General function used in connecting the retrieval and reranking of documents and the generation of answers,
    based on those documents.
    """
    rag = Retrieval(
        vector_index_path=os.path.join(cfg.DATA_DIR, 'faiss_index.bin'),
        embedding_model_name=cfg.EMBEDDING_MODEL,
        rerank_model_name=cfg.RERANK_MODEL,
        device=cfg.DEVICE
    )

    rag_index_dict = rag.run_rag(query, df)
    for i, (old_rank, data) in rag_index_dict.items():
        print(f'New Rank: {i}\tOld Rank: {old_rank}')
        print(f'Title: {data["title"]}\nChunk: {data["chunk"]}\nURL: {data["url"]}\n\n')

    llm = LLM(model_name=cfg.LLM_MODEL,
                access_token=cfg.TOKEN,
                attention_implementation='sdpa',
                device=cfg.DEVICE)

    context = [data for i, (old_rank, data) in rag_index_dict.items()]
    llm.format_prompt(query,
                      context)

    output = llm.generate_output(llm.tokenize_prompt())
    print(output)
    return pretify_output(output)


def test_llm():
    prompt = 'What is the outluck on Lithuanian economy after the covid pandemic?'
    llm = LLM(model_name=cfg.LLM_MODEL,
              access_token=cfg.TOKEN,
              attention_implementation='sdpa',
              device=cfg.DEVICE)

    context = [{'chunk': "According to Swedbank, “the pandemic highlighted the value of good policy and capable public administration”, with the report praising the early lockdown action taken by the government, despite “ill-preparedness” of the country’s health system. The country’s low-population density also helped keep the virus under control. Read more: Low population density is a blessing for Baltic states during Covid-19 pandemic – analysis The country is planning a “recovery spending spree” which presents an opportunity to improve digital infrastructure “that could facilitate green and sustainable growth”. Read more: Lithuania unveils €6.3bn 'DNA of the Future' economic stimulus plan The government’s willingness to dip into debt showed that, “probably for the first time since independence, the government has the fiscal firepower to enact a meaningful countercyclical policy”. However, the “excessive red tape and lack of administrative capacity, the distribution of support in many cases has been somewhat disappointing”. Read more: Saving businesses will not save the people – opinion According to Swedbank, “human capital is just as important,” and “the reversal migration trends will help in improving the medium-to-long-term growth outlook”. The full report on Nordic and Baltic countries is available here . Read more: Swedbank failed to mitigate scandal damage in Lithuania"},
               {'chunk': "Economic activity in Europe suffered a severe shock in the first half of the year and rebounded strongly in the third quarter as containment measures were gradually lifted. However, the resurgence of the pandemic in recent weeks is resulting in disruptions as national authorities introduce new public health measures to limit its spread. The epidemiological situation means that growth projections over the forecast horizon are subject to an extremely high degree of uncertainty and risks. The European Commission’s Autumn 2020 Economic Forecast – released on November 5 – projects that the euro area economy will contract by 7.8 per cent in 2020 before growing 4.2 per cent in 2021 and three per cent in 2022. The forecast also projects that the EU economy as a whole will contract by 7.4 per cent in 2020 before recovering with growth of 4.1 per cent in 2021 and three per cent in 2022. Compared to the summer forecast, growth projections for both the euro area and the EU are slightly higher for 2020 and lower for 2021. Output in both the euro area and the EU is not expected to recover its pre-pandemic level in 2022. The economic impact of the pandemic has differed widely across the EU and the same is true of recovery prospects. This reflects the spread of the virus, the stringency of public health measures taken to contain it, the sectoral composition of national economies and the strength of national policy responses. Coronavirus in Lithuania / Lithuania's Health Ministry Lithuania and Poland: shielded, to a point According to the forecast, the smallest fall in GDP anywhere in the EU is set to be Lithuania, where the economy will contract by just 2.2 per cent in 2020. Poland’s economy is also relatively well shielded, and is set to fall by 3.6 per cent. Lithuania was the only euro area member state that did not see real GDP decline in the first quarter of the year. Measures to stem the Covid-19 pandemic and general uncertainty took their toll in the second quarter when real GDP contracted by 5.9 per cent, with a decline in domestic demand the key reason behind the slump. Read more: Lithuania's GDP dip to be smallest in Europe, IMF expects Private consumption was markedly affected by the closure of most retail shops and the catering sector during the lockdown and uncertainty about labour income, while a drop in investments was already recorded in the fourth quarter of 2019. At the same time, net exports mitigated the situation as exports fell less than imports."},
               {'chunk': "Past figures show that epidemic-hit countries often face recession, he says. The good new, however, is that in most cases does not last more than two quarters, according to Mauricas, and an epidemic's economic impact is strongest during the first months. Read more: Lithuanian government announces school closures and travel restrictions to prevent coronavirus “If Lithuania does not become another epicentre of the epidemic and we manage to get through this with a fairly small number of cases, there's a really good chance of avoiding a recession,” Mauricas told BNS. Žygimantas Mauricas / BNS  nuotr. “ Lithuania is not too dependent on international tourism, and air transport sector is not very important. Moreover, our industry is not tightly integrated into global supply chains,” according to him. An indication of that is the fact that the industrial disruption in China is almost over and it had little effect on Lithuania, Mauricas said. Only several Lithuanian companies faced problems because it needed components from China. Meanwhile, other companies have been getting more orders as businesses have been looking to replace disrupted supply from China, often ready to pay more, according to Mauricas. Read more: No subsidies for businesses due to coronavirus – Lithuanian PM “I believe that, unless Lithuania becomes another virus epicentre, growth will slow, maybe to zero percent [or] we will have a formal recession in the first and second quarter, but the economy will recover at the end of the year and we will finish it with a plus,” he said. Lithuanian economy is less dependent on tourism than some other EU countries / BNS  nuotr. However, if the coronavirus epidemic in Lithuania reaches Italian proportions, “we will probably not manage to escape recession,” according to Mauricas. Even so, the recession would not last long. “ Looking at past cases, including the SARS outbreak in 2003, the swine flu in 2009 or even the Spanish flu which killed 50 million people, their most active periods lasted up to two quarters,” he said. “ Sometimes epidemics reemerge, but in that case their economic impact is weaker, since economic slowdown is due not only to the spread of the virus but also the accompanying emotions."}]
    llm.format_prompt(prompt,
                      context)

    print(llm.chat_template)
    tokenized_prompt = llm.tokenize_prompt()
    print(llm.generate_output(tokenized_prompt))

def test_retrieval():
    query = 'Lithuanian economy after the covid pandemic'

    rag = Retrieval(
        vector_index_path=os.path.join(cfg.DATA_DIR, 'faiss_index.bin'),
        embedding_model_name=cfg.EMBEDDING_MODEL,
        rerank_model_name=cfg.RERANK_MODEL,
        device=cfg.DEVICE
    )

    embeded_query = rag.embed_query(query)
    df = pd.read_csv(os.path.join(cfg.DATA_DIR, 'chunked_articles_with_embeddings.csv'))
    df['embedding'] = df['embedding'].apply(lambda x: ast.literal_eval(x))

    print(f'Query shape: {embeded_query.shape}')
    rag_index_dict = rag.run_rag(query, df)
    print(rag_index_dict)

    print(rag_index_dict)

    for i, (old_rank, data) in rag_index_dict.items():
        print(f'New Rank: {i}\tOld Rank: {old_rank}')
        print(f'Title: {data["title"]}\nChunk: {data["chunk"]}\nURL: {data["url"]}\n\n')

if __name__ == '__main__':
    # test_retrieval()
    # test_llm()
    query = 'How did education system change in Lithuania?'
    df = pd.read_csv(os.path.join(cfg.DATA_DIR, 'chunked_articles_with_embeddings.csv'))
    df['embedding'] = df['embedding'].apply(lambda x: ast.literal_eval(x))
    print('Started...')
    print(RAG_LLM_pipeline(query, df))