from re import split
from pandas.tseries import frequencies
from nltk.corpus import twitter_samples
from utils.process_tweet import (create_frequencies, create_loglihoods, 
                                 clean_tweet, derive_probabilities)
import numpy as np

# load tweets
all_positive_tweets = twitter_samples.strings("positive_tweets.json")
all_negative_tweets = twitter_samples.strings("negative_tweets.json")

# split the dataset
train_positive_tweets = all_positive_tweets[:4000]
test_positive_tweets = all_positive_tweets[4000:]
train_negative_tweets = all_negative_tweets[:4000]
test_negative_tweets = all_negative_tweets[4000:]

# clean positive and negative tweets
cleaned_positive_tweets = clean_tweet(train_positive_tweets)
cleaned_negative_tweets = clean_tweet(train_negative_tweets)

# create frequencies
frequencies = create_frequencies(cleaned_positive_tweets, cleaned_negative_tweets)

# create probabilities
probabilities, words = derive_probabilities(frequencies)

# create loglihood values
loglihoods = create_loglihoods(probabilities, words)

# predict with the sample tweet data
tweet = "i am happy and delighted that i am full of confidence and boundless courage :)"
processed_tweet = clean_tweet([tweet])[0]
loglihood = 0
for word in processed_tweet:
    loglihood += loglihoods[word]

print(tweet, ":", loglihood)

# find accuracy
x = test_positive_tweets + test_negative_tweets
y = np.append(np.ones((1,1000)), np.zeros((1,1000)))
yhat = np.zeros(y.shape)
i = 0
for tweet in x:
    processed_tweet = clean_tweet([tweet])[0]
    loglihood = 0
    for word in processed_tweet:
        loglihood += loglihoods[word]
    if loglihood > 0:
        yhat[i] = 1.0
    else:
        yhat[i] = 0.0
    i += 1

diff = np.squeeze(y)-yhat
error = diff.sum()/len(x)
accuracy = 1 - error
print(accuracy)

# error analysis
for key,value in enumerate(diff):
    if value:
        tweet = x[key]
        processed_tweet = clean_tweet([tweet])[0]
        loglihood = 0
        for word in processed_tweet:
            loglihood += loglihoods[word]
        print(y[key], ":", tweet, " : ", processed_tweet, " : ", loglihood)