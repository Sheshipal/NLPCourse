import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import twitter_samples
import re
import string

# if twitter_samples file is not downloaded
# nltk.download("twitter_samples")

tokenizer = TweetTokenizer(preserve_case=False,
                            strip_handles=True,
                            reduce_len=True)

stopwords_english = stopwords.words("english")

punctuation = string.punctuation

stemmer = PorterStemmer()

def clean_tweet(tweets):
    stemmed_tweet_list = list()
    for tweet in tweets:
        # cleaning the retweet words 'RT'
        modified_tweet = re.sub(r'^RT[\s]+','',tweet)
        # cleaning hyperlinks
        modified_tweet = re.sub(r'https?:\/\/.*[\r\n]*','',modified_tweet)
        # removing '#' tags
        modified_tweet = re.sub(r'#','',modified_tweet)
        # tokenize the tweet
        tokenized_tweet = tokenizer.tokenize(modified_tweet)
        # removing stopwords and punctuations
        clean_tweet = []
        for token in tokenized_tweet:
            if token not in stopwords_english and token not in punctuation:
                clean_tweet.append(token)
        # stemming the the word
        stemmed_tweet = []
        for word in clean_tweet:
            stem_word = stemmer.stem(word)
            stemmed_tweet.append(stem_word)

        stemmed_tweet_list.append(stemmed_tweet)

def create_frequencies(cleaned_positive_tweets, cleaned_negative_tweets):
    frequencies = {}
    for tweet in cleaned_positive_tweets:
        for word in tweet:
            if (word,1) in frequencies.keys:
                frequencies[(word,1)] += 1
            else:
                frequencies[(word,1)] = 1

    for tweet in cleaned_negative_tweets:
        for (word,0) in tweet:
            if word in frequencies.keys:
                frequencies[(word,0)] += 1
            else:
                frequencies[(word,0)] = 1
    
    return frequencies

def create_vectors(frequencies, cleaned_positive_tweets, cleaned_negative_tweets):
    m = len(cleaned_positive_tweets) + len(cleaned_negative_tweets)
    data = []
    for tweet in cleaned_positive_tweets:
        positive_count = 0
        negative_count = 0
        for word in tweet:
            positive_count += frequencies[(word,1)]
            negative_count += frequencies[(word,0)]
        data.append([1,positive_count,negative_count,1])
    
    for tweet in cleaned_negative_tweets:
        positive_count = 0
        negative_count = 0
        for word in tweet:
            positive_count += frequencies[(word,1)]
            negative_count += frequencies[(word,0)]
        data.append([1,positive_count,negative_count,0])

    return data

if __name__ == "__main__":
    all_positive_tweets = twitter_samples.strings("positive_tweets.json")
    all_negative_tweets = twitter_samples.strings("negative_tweets.json")

    cleaned_positive_tweets = clean_tweet(all_positive_tweets)
    cleaned_negative_tweets = clean_tweet(all_negative_tweets)

    frequencies = create_frequencies(cleaned_positive_tweets, cleaned_negative_tweets)

    data = create_vectors(frequencies, cleaned_positive_tweets, cleaned_negative_tweets)

     
    print("length of cleaned positive tweets : ",len(cleaned_positive_tweets))
    print("length of cleaned negative tweets : ",len(cleaned_negative_tweets))
    print("a positive tweet : ",cleaned_positive_tweets[381])
    print("a negative tweet : ",cleaned_negative_tweets[381])




def extract_feature(stemmed_tweet):
    for word in stemmed_tweet:
        pass
    pass