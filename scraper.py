from bs4 import BeautifulSoup
import requests
import os
import src.config as cfg
import pandas as pd
from tqdm import tqdm
from spacy.lang.en import English

class ArticleScraper:
    def __init__(self, base_url: str, chunk_size: int = 10):
        """
        Class for scraping articles from a given URL.
        Articles need to  be in english.
        The class extracts the main body of the article and metadata.
        """
        self.base_url = base_url
        self.article_dict = {}
        self.soup = None
        self.nlp = English()
        self.nlp.add_pipe('sentencizer')
        self.chunck_size = chunk_size

    def setup_connection(self) -> None:
        response = requests.get(self.base_url)
        response.encoding = 'utf-8'
        response.raise_for_status()
        self.soup = BeautifulSoup(response.text, 'html.parser')

    def get_main_content(self) -> None:
        self.main_content = self.soup.find('main', {'id': 'page-content', 'class': 'page__content container'})

    def get_title(self) -> str:
        return self.main_content.find('h1', class_='title-block__heading').text

    def get_author_and_date(self) -> tuple[str, str]:
        author_and_date = self.main_content.find('div', class_='avatar-group__description').text.split('\n')
        return author_and_date[1], author_and_date[2]

    def clean_article(self):
        """
        Some html code had comments in them in side the article content.
        This function removes specific kind of comments
        """
        article_div = self.main_content.find('div', class_='article-content js-text-selection')
        tags_with_class = article_div.find_all(class_="text-lead")
        for tag in tags_with_class:
            tag.decompose()

        article_html = str(article_div)
        while '<!--googleoff: all-->' in article_html and '<!--googleon: all-->' in article_html:
            googleoff_index = article_html.find('<!--googleoff: all-->')
            googleon_index = article_html.find('<!--googleon: all-->', googleoff_index)

            if googleon_index == -1:
                break

            article_html = article_html[:googleoff_index] + article_html[googleon_index + len('<!--googleon: all-->'):]

        self.cleaned_main_content = BeautifulSoup(article_html, 'html.parser')

    def get_article_id(self) -> str:
        target_div = self.cleaned_main_content.find('div', class_='article-content js-text-selection')
        try:
            id = target_div['id']
        except KeyError:
            id = 'ID not found'
        return id

    def get_text(self) -> str:
        return self.cleaned_main_content.get_text(separator=' ', strip=True)

    def get_sentences(self, text) -> list[str]:
        """
        Function to split the text into sentences using NLP library
        """
        doc = self.nlp(text)
        return [str(sent.text) for sent in doc.sents]

    def read_article(self) -> dict:
        self.get_main_content()

        title = self.get_title()
        author, date = self.get_author_and_date()

        self.clean_article()

        article_id = self.get_article_id()

        text = self.get_text()
        sentences = self.get_sentences(text)

        article_dict = {'url': self.base_url,
                        'article_id': article_id,
                        'title': title,
                        'author': author,
                        'date': date,
                        'char_count': len(text),
                        'word_count': len(text.split(" ")),
                        'sentence_count': len(sentences),
                        'token_count': len(text) / 4,  # 1 token ~= 4 characters
                        'text': text,
                        'sentences': sentences}

        return article_dict


def read_all_articles(data_dir: str) -> list[dict]:
    url_df = pd.read_csv(os.path.join(data_dir, 'urls.csv'))

    urls = url_df['URLs'].to_numpy()
    tags = url_df['Tags'].to_numpy()
    n_articles = len(urls)

    all_articles = []
    for i in tqdm(range(n_articles)):
        url = urls[i]
        scraper = ArticleScraper(url)
        scraper.setup_connection()
        article_dict = scraper.read_article()
        article_dict['tag'] = tags[i].lower()
        all_articles.append(scraper.read_article())

    return all_articles


if __name__ == '__main__':
    all_articles = read_all_articles(cfg.DATA_DIR)
    df = pd.DataFrame(all_articles)
    df.to_csv(os.path.join(cfg.DATA_DIR, 'articles.csv'), encoding='utf-8-sig', index=False)