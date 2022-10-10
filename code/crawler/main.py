import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pickle
import pandas as pd

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

# Add constants for use
max_link_count = 15
stop_words = set(stopwords.words('english'))


def scrape():
    # Use breadth first iteration on the first link to scrape related links
    queue = ['https://en.wikipedia.org/wiki/Fiber-optic_cable']
    count = 0
    url_match = r'(http|ftp|https):\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?'

    # Check to see if we've used enough links
    while len(queue) != 0 and count < max_link_count:
        url = queue[0]
        del queue[0]

        print(url)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        for data in soup(['style', 'script']):
            # Remove tags
            data.decompose()

        # get only text from url. Write result to file
        output = ' '.join(soup.stripped_strings)
        with open(f"link_{count}.txt", 'w') as f:
            f.write(output)

        count += 1

        # find the next valid url
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
    # Use Vectorizer to determine importance of words.
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
    # list of terms deemed important by vectorizor and domain expert
    term_list = ['fiber', 'cables', 'cord', 'light', 'optic', 'jacket', 'fibers', 'software', 'electrical', 'traffic']
    data = {}
    for term in term_list:
        data[term] = []

    # add sentences to this dictionary if a term exists in the sentence
    for i in range(max_link_count):
        with open(f'sent_token_{i}.pickle', 'rb') as handle:
            sents = pickle.load(handle)

        for sent in sents:
            for term in term_list:
                if term in sent:
                    data[term].append(sent)

    with open("output.pickle", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


scrape()
clean()
freq()
build_data()
