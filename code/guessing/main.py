import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from random import seed
from random import randint

seed(1234)


def main():
    # Get data from file
    filename = sys.argv[1]
    if filename is None:
        print("No filename specified. example: setup.py ./path.txt");
        exit(1)

    # Read from file
    with open(filename) as file:
        raw_text = file.read().replace('\n', '')

    # Compute lex diversity
    tokens = word_tokenize(raw_text)
    token_count = len(tokens)
    unique_count = len(set(tokens))

    lex_diversity = unique_count / token_count

    print(f"Lex Diversity: {lex_diversity}")

    # Preprocess the text to get nouns & noun counts
    (alpha_tokens, nouns) = preprocess(tokens)

    noun_count = {}

    for noun in nouns:
        noun_count[noun] = alpha_tokens.count(noun)

    noun_count = dict(sorted(noun_count.items(), key=lambda item: item[1], reverse=True))
    common_nouns = list(map(lambda item: item[0], list(noun_count.items())[:50]))

    game(common_nouns)


def preprocess(text):
    # compute list of valid alpha words
    stops = set(stopwords.words('english'))
    lower = [w.lower() for w in text]
    alpha = list(filter(lambda word: word not in stops and word.isalpha() and len(word) > 5, lower))

    # lemmatize and tag the tokens
    lemmatizer = WordNetLemmatizer()
    lemmed = set([lemmatizer.lemmatize(word) for word in alpha])
    tags = nltk.pos_tag(lemmed)
    print(f"First 20 Tags: {tags[:20]}")

    # Gather nouns from tags
    nouns = list(map(lambda tag: tag[0], filter(lambda tag: tag[1][0] == 'N', tags)))
    print(nouns)

    # Print number of retrieved items
    print(f"Token Length: {len(alpha)}")
    print(f"Noun Length: {len(nouns)}")

    return alpha, nouns


def game(words):
    score = 5
    print("Play the word guessing game!")
    # random word generation
    i = randint(0, 49)
    target = words[i]

    # map characters in target word for easy access
    target_map = {}
    for idx, char in enumerate(target):
        target_map[char] = [*target_map[char], idx] if char in target_map else [idx]

    # Initialize guesses and user input
    guesses = ['_'] * len(target)
    print(' '.join(guesses))
    guess = input("Enter your guess: ")

    # Game while loop
    while guess != '!' and score > 0:
        if guess in target_map:
            score += 1
            print(f"Correct! Score: {score}")
            for index in target_map[guess]:
                guesses[index] = guess
            del target_map[guess]
        else:
            score -= 1
            print(f"Incorrect! Score: {score}")

        print(' '.join(guesses))
        if len(target_map) > 0:
            guess = input("Enter your guess: ")
        else:
            break

    if guess == '!':
        print(f"Goodbye. Score: {score}")
    elif score < 0:
        print("You lost, that's crazy.")
    else:
        print(f"You won! Your Score: {score}")


main()
