"""
Samuel Anozie
November 15, 2022
Human Language Technologies: ChatBot
A ChatBot built around stereotypical and sentimental data used in psychological research.
Data Source: https://osf.io/3ea4b
"""

import pickle
import random
import time
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from info import train, generate_df
import pandas as pd

stoplist = set(stopwords.words('english'))
wnl = WordNetLemmatizer()
tags_of_interest = {'JJ', 'NN', 'NNP', 'RB'}
df_cols = ['Friendly_Sociable', 'Competent_Skilled', 'Selfconfident_Assertive', 'Trustworthy_Moral',
           'Wealthy_Highstatus']
df_dict = {}
target_set = set()
threshold = 3
incomprehensible_responses = ['No comment. Try again?',
                              "Honestly, I know what you said, but you didn't give me enough details to figure out how I feel. Please try again. ",
                              "Oh, so I'm basically you? \nNo, actually I just didn't get enough info, try again:",
                              "Is that English? I can't understand. Either you're gaslighting me too hard or whoever created needs more practice. Try again?"]
target_responses = {
    'Friendly_Sociable': {
        'low': [
            "People might not think I'm not too friendly (they're wrong but alright). What other information do you have for me?",
            "I hope people think I'm approachable, I might not seem it at first. What else can you think of?",
            "Hm. The data is telling me I'm not too friendly. That's problematic, honestly. Cheer me up?"],
        'high': ["I must be pretty friendly then! What else can you think of?",
                 "Yep, it sounds like I'm in a sociable mood. Got anything else?",
                 "People must think that I'm an approachable person? At least the data says so. Anything else?"]
    },
    'Competent_Skilled': {
        'low': ["Why does the internet think I'm not generally competent? Odd... moving on:",
                "People might be underestimating me, but I'll show them! Let me know what else you're thinking.",
                "Interesting, very interesting. Anything else?"],
        'high': ["Yep, I'm good at what I do too. At least people might assume that. Got anything else?",
                 "I'm guessing I'm the go-getter type person too. What else you got?",
                 "The internet thinks those types of people are 'competent'. Hopefully for good and not bad reason? Anything else?",
                 "Stereotypically speaking, people must think I'm generally competent. Why? Answer for yourself for now. Anything else?"]
    },
    'Selfconfident_Assertive': {
        'low': ["Maybe it takes a bit for me to get out of my shell. Anything else?",
                "Hopefully people aren't out here underestimating me, I do well when I can. And?",
                "I may be just alright on the outside, but I'm beaming on the inside. Sometimes. Anything else?"],
        'high': ["And I'm proud to hear it! Let me know what else you're thinking.",
                 "Sounds like I got some guts, I like that. Anything else?",
                 "Main character energy? Or toxic energy? We'll find out soon enough. Anything else?"]
    },
    'Trustworthy_Moral': {
        'low': ["It's a bit sad. The Internet is telling me I'm not trustworthy. That's problematic... anything else?",
                "Judging off of my internet scrapes, people might be wary of me. That's not a me problem though. Y'all should fix your prejudices. What else you got?",
                "A bit concerning that the data is indicating I might not be moral. I think everybody is moral. What's next?"],
        'high': [
            "Seems like I'm the type of person people would like to confide in. Maybe that's a good thing, if its justified. Is it justified...? Anything new?",
            "Based off of what you said, the internet thinks I'm trustworthy. So I'll come clean: this bot was built using stereotype data. So take care to not be problematic. Next?",
            "I'm trustworthy! I wonder what makes me trustworthy. Just look in the black box that is my code real quick. Anything else?"]
    },
    'Wealthy_Highstatus': {
        'low': [
            "The internet is doing me dirty, making me think that I'm broke stereotypically. No. You're broke, alright? Alright! Moving on:",
            "It is what it is. People might assume I'm cheap, but even cheap has charm. No need to be problematic about it. What else you got?",
            "Look. Stereotypes don't really help anybody, and the data makes me look cheap based off of what you said. Do humans feel proud about the world they've built? Moving on:"],
        'high': [
            "The human species would assume I got some good change in my pocket. Not nessesarily against that.... anything else?",
            "Internet people think I have money bags based off of what you said. I like that. How else do you feel?",
            "Looks like people think I'm rich. Good or bad, depending on how you look at it. Anything new?"]
    }
}


def main():
    # Set up data
    print("Loading in data...")
    df = generate_df()
    for word in df.tv:
        target_set.add(word)

    # Train on columns
    print("Training models...")
    for key in df_cols:
        df_dict[key] = train(df.loc[:, ['tv', key]].copy(), key)
    print("Done training!")

    # Starting chatbot
    profile = input("Welcome to the GaslightBot. Do you have a profile? (yes/no) ")
    if profile == "no":
        chance = random.random()
        if chance > 0.9:
            print("No, I'm sure you do, I recognize you. I got access to your camera ;) just kidding. Unless....")
            time.sleep(5)
        userdata = create_profile()
    else:
        userdata = profile_loop()

    print(f"{userdata['username']}, please, tell me about myself below. Be as creative as you'd like, the more info, "
          f"the better! (Enter EXIT to end the program.)")
    converse(df, userdata)


def converse(df, userdata):
    info = input()
    while info != 'EXIT':
        bot_sentiments, query = predict(df, info)
        userdata['queries'].append({'bot_sentiment': bot_sentiments, 'query': query})
        with open(f"{userdata['username']}.pickle", "wb") as handle:
            pickle.dump(userdata, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(bot_sentiments)
        print(respond(bot_sentiments))
        info = input()

    print("See you later!")


def respond(sentiments):
    max_vibe = 0
    max_sentiment = None
    trait = None

    for sentiment, value in sentiments.items():
        vibe = abs(value - threshold)
        if vibe > max_vibe:
            max_vibe = vibe
            max_sentiment = sentiment
            trait = 'low' if value < threshold else 'high'

    return random.choice(target_responses[max_sentiment][trait])


def predict(df, info):
    while True:
        bot_sentiment = {}
        meta_tokens = word_tokenize(info)
        meta_tagged = nltk.pos_tag(meta_tokens)
        characteristics = [item[0].lower() for item in filter(lambda tag: tag[1] in tags_of_interest, meta_tagged) if
                           not item[0].lower() in [*stoplist, 'bit']]
        try:
            series = pd.Series(characteristics, copy=False)
            for col in df_dict:
                classifier, vectorizer = df_dict[col]
                vectors = vectorizer.transform(series)
                pred = classifier.predict(vectors)
                bot_sentiment[col] = sum(pred) / len(pred)

            return bot_sentiment, info
        except:
            pass
            print(random.choice(incomprehensible_responses))
            info = input()


def profile_loop():
    flow = "try again"
    while flow != "create":
        username = input("What's your username? ")
        try:
            with open(f'{username}.pickle', 'rb') as handle:
                userdata = pickle.load(handle)
            query_num = len(userdata['queries'])
            print(f"Your information has been loaded! Looks like you've got {query_num} quer{'y' if query_num == 1 else 'ies'} so far.")
            return userdata
        except:
            pass
            flow = input("Sorry, I can't find your information. Would you like to create an account or try again? (create/try again) ")
    return create_profile()


def create_profile():
    username = None
    while username is None:
        username = input("Alright, what would you like your username to be? ")
        try:
            with open(f'{username}.pickle', 'rb') as handle:
                print("That username is taken. try another one")
                username = None
        except:
            pass
            chance = random.random()
            if chance > 0.9:
                print("That username is taken. try another one.")
                time.sleep(3)
                print("You probably thought I messed up the code, but no, just some gas!")
                time.sleep(5)
            print(f"Welcome, {username}")
            break

    userdata = {"username": username, "queries": []}
    print(userdata)
    with open(f"{username}.pickle", "wb") as handle:
        pickle.dump(userdata, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return userdata


main()
