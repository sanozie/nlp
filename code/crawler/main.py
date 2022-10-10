import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pickle
import pandas as pd

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

max_link_count = 15
stop_words = set(stopwords.words('english'))


def scrape():
    queue = ['https://en.wikipedia.org/wiki/Fiber-optic_cable']
    search_param = []
    count = 0
    url_match = r'(http|ftp|https):\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?'

    while len(queue) != 0 and count < max_link_count:
        url = queue[0]
        del queue[0]

        print(url)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.find_all(text=True)

        for data in soup(['style', 'script']):
            # Remove tags
            data.decompose()

            # return data by retrieving the tag content
        output = ' '.join(soup.stripped_strings)
        print(output)

        with open(f"link_{count}.txt", 'w') as f:
            f.write(output)

        count += 1
        for link in soup.findAll('a'):
            link = link.get('href')
            print(link)
            if link:
                match = re.match(url_match, link)
                print(match)
                if match is not None:
                    queue.append(link)


def clean():
    for i in range(max_link_count):
        with open(f"link_{i}.txt", 'r') as f:
            raw_text = f.read().replace('\n', '').replace('\t', '')

        with open(f"sent_token_{i}.pickle", 'wb') as handle:
            pickle.dump(sent_tokenize(raw_text), handle, protocol=pickle.HIGHEST_PROTOCOL)


def freq():
    vectorizer = TfidfVectorizer()
    documents = []
    for i in range(max_link_count):
        with open(f'sent_token_{i}.pickle', 'rb') as handle:
            sents = pickle.load(handle)

        sentences = []
        for sent in sents:
            word_tokens = word_tokenize(sent)
            filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
            sentences.append(' '.join(filtered_sentence))

        documents.append(' '.join(sentences))

    vectors = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)

    print(df.max().sort_values(ascending=False)[: 40])


def build_data():
    term_list = ['fiber', 'cables', 'cord', 'light', 'optic', 'jacket', 'fibers', 'software', 'electrical', 'traffic']
    data = {}
    for term in term_list:
        data[term] = []

    for i in range(max_link_count):
        with open(f'sent_token_{i}.pickle', 'rb') as handle:
            sents = pickle.load(handle)

        for sent in sents:
            for term in term_list:
                if term in sent:
                    data[term].append(sent)

    print(data)

scrape()
clean()
freq()
build_data()
