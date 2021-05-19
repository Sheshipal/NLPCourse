import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import twitter_samples
import re
import string

tokenizer = TweetTokenizer(preserve_case=False,
                            strip_handles=True,
                            reduce_len=True)

stopwords_english = stopwords.words("english")

punctuation = string.punctuation

stemmer = PorterStemmer()

def clean_tweet(tweet):
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

    return stemmed_tweet

if __name__ == "__main__":
    all_positive_tweets = twitter_samples.strings("positive_tweets.json")
    all_negative_tweets = twitter_samples.strings("negative_tweets.json")

    cleaned_positive_tweets = []
    for tweet in all_positive_tweets:
        cleaned_tweet = clean_tweet(tweet)
        cleaned_positive_tweets.append(cleaned_tweet)

    cleaned_negative_tweets = []
    for tweet in all_negative_tweets:
        cleaned_tweet = clean_tweet(tweet)
        cleaned_negative_tweets.append(cleaned_tweet)
    
    print("length of cleaned positive tweets : ",len(cleaned_positive_tweets))
    print("length of cleaned negative tweets : ",len(cleaned_negative_tweets))
    print("a positive tweet : ",cleaned_positive_tweets[381])
    print("a negative tweet : ",cleaned_negative_tweets[381])