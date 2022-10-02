from nltk import word_tokenize
from nltk.util import ngrams
import pickle

training_list = {
    'english': 'ngram_files/LangId.Train.English',
    'french': 'ngram_files/LangId.Train.French',
    'italian': 'ngram_files/LangId.Train.Italian'
}


def main():
    unigram_count = 0
    for language, filepath in training_list.items():
        (unigram, bigram) = grams(filepath)
        unigram_count += len(unigram)

        with open(f'{language}_unigram.pickle', 'wb') as handle:
            pickle.dump(unigram, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{language}_bigram.pickle', 'wb') as handle:
            pickle.dump(bigram, handle, protocol=pickle.HIGHEST_PROTOCOL)


def grams(filename):
    with open(filename) as file:
        raw_text = file.read().replace('\n', '')

    unigrams = word_tokenize(raw_text)
    bigrams = list(ngrams(unigrams, 2))
    unigram_dict = {t: unigrams.count(t) for t in set(unigrams)}
    bigram_dict = {b: bigrams.count(b) for b in set(bigrams)}

    return unigram_dict, bigram_dict


main()
