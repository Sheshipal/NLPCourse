import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import string

# if twitter_samples file is not downloaded
# nltk.download("twitter_samples")

def clean_tweet(tweets):
    """
    Clean Tweet.
    Input:
        tweets: list of tweets
    Output:
        stemmed_tweet_list: list of stemmed tweets
    """

    tokenizer = TweetTokenizer(preserve_case=False,
                            strip_handles=True,
                            reduce_len=True)

    stemmer = PorterStemmer()

    stopwords_english = stopwords.words("english")

    punctuation = string.punctuation

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
    
    return stemmed_tweet_list

def create_frequencies(cleaned_positive_tweets, cleaned_negative_tweets):
    """
    Creates Frequencies.
    Input:
        cleaned_positive_tweets: list of cleaned positive tweets
        cleaned_negative_tweets: list of cleaned negative tweets
    Output:
        frequencies: default dictionary with frequency of words
    """
    #create a defaultdict variable
    frequencies = defaultdict(lambda:0)

    #creating frequencies for all positive and negative tweets
    for tweet in cleaned_positive_tweets:
        for word in tweet:
            frequencies[(word,1)] += 1
    
    for tweet in cleaned_negative_tweets:
        for word in tweet:
            frequencies[(word,0)] += 1
    
    return frequencies

def create_vectors(frequencies, cleaned_positive_tweets, cleaned_negative_tweets):
    """
    Creates Vectors.
    Input:
        frequencies: default dictionary of frequency of words
        cleaned_positive_tweets: list of cleaned positive tweets
        cleaned_negative_tweets: list of cleaned negative tweets
    Output:
        data: numpy array containing vectors
    """
    m = len(cleaned_positive_tweets) + len(cleaned_negative_tweets)
    data = np.zeros((m,4))
    
    i = 0 
    #generating the vectors for each tweet
    for tweet in cleaned_positive_tweets:
        positive_count = 0
        negative_count = 0
        for word in tweet:
            positive_count += frequencies[(word,1)]
            negative_count += frequencies[(word,0)]
        data[i] = [1,positive_count,negative_count,1]
        i += 1
    
    for tweet in cleaned_negative_tweets:
        positive_count = 0
        negative_count = 0
        for word in tweet:
            positive_count += frequencies[(word,1)]
            negative_count += frequencies[(word,0)]
        data[i] = [1,positive_count,negative_count,0]
        i += 1

    # un comment this block if you want to save the vectors as a csv file
    # df = pd.DataFrame(data, columns=['bias','positive_frequency','negative_frequency','label'])
    # df.to_csv("vectors.csv")

    return data

def create_vector(frequencies, stemmed_tweet):
    positive_count = 0
    negative_count = 0
    for word in stemmed_tweet:
        positive_count += frequencies[(word,1)]
        negative_count += frequencies[(word,0)]
    vector = [1,positive_count,negative_count]
    return np.array(vector)

def derive_probabilities(frequencies):
    """
    Input:
        frequencies: default dictionary containing frequency of words
    Output:
        probabilities: dictionary containing probabilities
    """
    # create word count for laplacian smoothing
    words = { word for word,_ in frequencies.keys() }
    word_count = len(words)
 
    # calculate summation of total number of words in each group
    pos_words = 0
    neg_words = 0
    for key,value in frequencies.items():
        if key[-1] == 1.0:
            pos_words += value
        else:
            neg_words += value

    # compute probabilities of each word
    probabilities = defaultdict()
    for word in words:
        key = (word,0.0)
        value = frequencies[key]
        probabilities[key] = (value + 1)/(neg_words + word_count)
        key = (word,1.0)
        value = frequencies[key]
        probabilities[key] = (value + 1)/(pos_words + word_count)

    return probabilities, words

def create_loglihoods(probabilities, words):
    """
    Input:
        probabilities: default dictionary of probabilities
    Output:
        loglihoods: default dictionary of loglihoods
    """
    loglihoods = defaultdict(lambda:0)

    # find loglihood for every word in corpus
    for word in words:
        loglihoods[word] = np.log(probabilities[(word, 1.0)]/probabilities[(word, 0.0)])

    return loglihoods
    
if __name__ == "__main__":
    all_positive_tweets = twitter_samples.strings("positive_tweets.json")
    all_negative_tweets = twitter_samples.strings("negative_tweets.json")
    print("length of all positive tweets : " ,len(all_positive_tweets))
    print("length of all negative tweets : " ,len(all_negative_tweets))

    cleaned_positive_tweets = clean_tweet(all_positive_tweets)
    cleaned_negative_tweets = clean_tweet(all_negative_tweets)
    print("length of cleaned positive tweets : " ,len(cleaned_positive_tweets))
    print("length of cleaned negative tweets : " ,len(cleaned_negative_tweets))
    
    frequencies = create_frequencies(cleaned_positive_tweets, cleaned_negative_tweets)
    print("length of frequencies : ",len(frequencies))

    data = create_vectors(frequencies, cleaned_positive_tweets, cleaned_negative_tweets)

    for vector in data[:5]:
        print(vector[:-1])
    for vector in data[-5:]:
        print(vector[:-1])

    keys = ['happi', 'merri', 'nice', 'good', 'bad', 'sad', 'mad', 'best', 'pretti',
        '???', ':)', ':(', '????', '????', '????', '????', '???',
        'song', 'idea', 'power', 'play', 'magnific']

    x = []
    y = []
    fig, ax = plt.subplots(figsize = (8, 8))
    x = np.log([frequencies[(key,1)] for key in keys])
    y = np.log([frequencies[(key,0)] for key in keys])

    for i in range(len(keys)):
        ax.annotate(keys[i], (x[i],y[i]), fontsize=12)

    plt.xlabel("Log Positive count")
    plt.ylabel("Log Negative count")

    ax.plot([0, 9], [0, 9], color = 'red')

    ax.scatter(x, y)

    plt.show()