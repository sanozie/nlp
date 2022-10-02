from nltk import word_tokenize
from nltk.util import ngrams
import pickle

training_list = {
    'english': 'ngram_files/LangId.Train.English',
    'french': 'ngram_files/LangId.Train.French',
    'italian': 'ngram_files/LangId.Train.Italian'
}


def predict():
    with open("ngram_files/LangId.test") as file:
        test_lines = file.readlines()

    grams = {}
    count = 0

    for language in training_list:
        grams[language] = {}
        with open(f'{language}_unigram.pickle', 'rb') as handle:
            grams[language]["unigram"] = pickle.load(handle)

        with open(f'{language}_bigram.pickle', 'rb') as handle:
            grams[language]["bigram"] = pickle.load(handle)

        count += len(grams[language]["unigram"])

    predictions = []
    for line in test_lines:
        probabilities = []
        bigrams_test = list(ngrams(word_tokenize(line), 2))

        for language in training_list:
            laplace = 1

            for bigram in bigrams_test:
                n = grams[language]["bigram"][bigram] if bigram in grams[language]["bigram"] else 0
                d = grams[language]["unigram"][bigram[0]] if bigram[0] in grams[language]["unigram"] else 0

                laplace = laplace * ((n + 1) / (d + count))

            probabilities.append((language, laplace))

        max_probability = 0
        probable_language = None
        for probability in probabilities:
            if max_probability < probability[1]:
                (probable_language, max_probability) = probability

        predictions.append(probable_language)

    with open('predictions.pickle', 'wb') as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("ngram_files/LangId.sol") as solution_file:
        solutions = solution_file.readlines()

    correct_count = 0
    for solution in solutions:
        [line_num, lang] = solution.split(' ')
        line_num = int(line_num)
        lang = lang.replace('\n', '').lower()
        if lang.lower() != predictions[line_num - 1]:
            print(f"Incorrect Prediction on line {line_num}, got {predictions[line_num - 1]}")
        else:
            correct_count += 1

    print(f"Overall Accuracy: {(correct_count / len(solutions)) * 100}%")


predict()
