from re import split
from pandas.tseries import frequencies
from nltk.corpus import twitter_samples
from utils.process_tweet import (create_frequencies, create_vector,
                                 create_vectors, clean_tweet)
import numpy as np

# nltk.download("twitter_samples")

def sigmoid(z):
    """
    Applys sigmoid activation function:
    Input:
        z : an array
    Output:
        a : array of shape same as z
    """
    a = 1/(1+np.exp(-z))
    return a

def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Computes gradient descent
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    '''
    m = len(x)
    features = len(x[0])

    for i in range(0, num_iters):
        print(f"Iteration {i}")
        # get z, the dot product of x and theta
        z = np.dot(x,theta)
        assert(z.shape == (m,1))

        # get the sigmoid of z
        h = sigmoid(z)
        assert(h.shape == (m,1))
        
        # calculate the cost function
        J = (-1.0/m)*(np.dot(np.transpose(y),np.log(h)) + np.dot((1-np.transpose(y)),np.log(1-h)))
        
        # update the weights theta
        theta = theta - (alpha/m) * np.dot(np.transpose(x), h-y)
        # we can also use below code
        # theta = theta - np.multiply((alpha/m),np.dot(np.transpose(x), h-y))
        assert(theta.shape == (features,1))

    J = float(J)
    return J, theta

def predict_tweet(tweet, frequencies, theta):
    """
    Predict Tweet
    Input:
        tweet:
        frequencies:
        theta:
    Output:
        prediction: predicted value
    """
    #clean tweet
    stemmed_tweet = clean_tweet([tweet])[0]
    
    #convert cleaned tweet into a vector
    input_vector = create_vector(frequencies, stemmed_tweet)

    #perform sigmoid of dot product of vector and theta
    prediction = sigmoid(np.dot(input_vector, theta))

    return prediction

def test_logistic_regression(test_x, test_y, freqs, theta):
    """
    Input: 
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output: 
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    
    # the list for storing predictions
    y_hat = []
    
    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)
        
        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1)
        else:
            # append 0 to the list
            y_hat.append(0)

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    accuracy = (y_hat==np.squeeze(test_y)).sum()/len(test_y)

    return accuracy

#load tweets
all_positive_tweets = twitter_samples.strings("positive_tweets.json")
all_negative_tweets = twitter_samples.strings("negative_tweets.json")

#split tweets
positive_train_tweets = all_positive_tweets[:4000]
positive_test_tweets = all_positive_tweets[4000:]
negative_train_tweets = all_negative_tweets[:4000]
negative_test_tweets = all_negative_tweets[4000:]

#clean training tweets
cleaned_positive_tweets = clean_tweet(positive_train_tweets)
cleaned_negative_tweets = clean_tweet(negative_train_tweets)

#find frequencies
frequencies = create_frequencies(cleaned_positive_tweets, cleaned_negative_tweets)

#convert cleaned tweets into vectors
data = create_vectors(frequencies, cleaned_positive_tweets, cleaned_negative_tweets)

#split numpy array into X and Y
X = data[:,:3]
Y = np.reshape(data[:,-1], (len(data),1))

#train model
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 5)
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")

# Test model with own data
for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    print(tweet , predict_tweet(tweet, frequencies, theta))

# test model with testing data
test_x = positive_test_tweets + negative_test_tweets
test_y = np.append(np.ones((len(positive_test_tweets),1)), np.zeros((len(negative_test_tweets),1)), axis=0)
print("accuracy : ", test_logistic_regression(test_x, test_y, frequencies, theta))

### Error Analysis ###
print('Label Predicted Tweet')
for x,y in zip(test_x,test_y):
    y_hat = predict_tweet(x, frequencies, theta)

    if np.abs(y - (y_hat > 0.5)) > 0:
        print('THE TWEET IS:', x)
        print('THE PROCESSED TWEET IS:', clean_tweet([x])[0])
        print('%d\t%0.8f\t%s' % (y, y_hat, ' '.join(clean_tweet([x])[0]).encode('ascii', 'ignore')))