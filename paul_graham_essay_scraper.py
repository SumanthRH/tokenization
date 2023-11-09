"""
A simply essay scraper for Paul Graham's essays. Adapted from: https://github.com/Arieg419/Paul-Graham-Essay-Scraper
"""
from bs4 import BeautifulSoup

import requests
import sys
import pprint
import time

save_path = "all_essays.txt"

def find_nth_occurrence(text, sub, n):
    if n == 1:
        return text.find(sub)
    start = text.find(sub)
    return find_nth_occurrence(text[start + 1:], sub, n - 1)


pp = pprint.PrettyPrinter(indent=4)
base_url = "http://www.paulgraham.com/"
r = requests.get("http://www.paulgraham.com/articles.html")
data = r.text
articles = {}

soup = BeautifulSoup(data, "html.parser")

# Get List of Articles to scrape. Hacky but works
fonts = soup.find_all('font')
for f in fonts:
    article_anchor_tag = f.find('a')
    anchor_string = str(article_anchor_tag)
    ts, te = anchor_string.find(
        '>'), find_nth_occurrence(anchor_string, '<', 2)
    us, ue = find_nth_occurrence(
        anchor_string, '=', 1), find_nth_occurrence(anchor_string, '>', 1)
    title = anchor_string[ts + 1:te + 1]
    url = base_url + anchor_string[us + 2:ue - 1]
    articles[title] = url

for k, v in articles.items():
    r = requests.get(v)
    data = r.text
    soup = BeautifulSoup(data, "html.parser")
    tables = soup.find_all('table')
    if len(tables) < 2:  # weird layout
        continue
    curr_table = tables[1]
    pp.pprint(curr_table)
    article = open(k.replace(" ", "").replace("/", "") + ".html", "w+")
    try:
        article.write(str(curr_table))
        article.close()
    except Exception as e:
        print("Something went wrong...")
        print(type(e))
        print(str(e))