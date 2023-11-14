"""
A simply essay scraper for Paul Graham's essays.
"""
from bs4 import BeautifulSoup

import requests
import pprint
from tqdm import tqdm

save_path = "all_essays.txt"

def get_article_text(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    # Main text is within <table> tags
    # and the second table contains the article text.
    if len(soup.find_all('table')) < 2:
        print("bad layout")
        return "" # bad layout
    article_text = soup.find_all('table')[1].get_text()
    return article_text


pp = pprint.PrettyPrinter(indent=4)
base_url = "http://www.paulgraham.com/"
r = requests.get("http://www.paulgraham.com/articles.html")
data = r.text
articles = {}

soup = BeautifulSoup(data, "html.parser")

all_articles_text = ""

# Extract articles URLs
for link in tqdm(soup.select('font > a')):
    article_url = base_url + link.get('href')
    all_articles_text += get_article_text(article_url) + "\n\n"


# Save all articles text into one file
with open("all_pg_essays.txt", "w") as file:
    file.write(all_articles_text)