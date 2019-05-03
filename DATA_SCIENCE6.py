# imports needed
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.io
from skimage import io
import cv2

# setting seed, DON'T modify
np.random.seed(10)
from pylab import rcParams
rcParams['figure.figsize'] = 10, 7

def loadImages(files = ['gumballs.jpg','snake.jpg','twins.jpg'], grayscale = False):
    images = []
    for filename in files:
        # load in image, using grayscale parameter
        x = 0

        if not grayscale:
            # make sure to scale to [0,1] if color image
            x = cv2.imread(filename)
        else:
            x = cv2.imread(filename, 0)
        images.append(x)
    return images

def loadFilterBank():
    return scipy.io.loadmat('filterBank.mat')['F']


'''
Inputs:
    - imageStack : list of numpy arrays of dimension N x M x 3 (where N and M can differ between each image)
    - F          : filter bank of dimension 49 x 49 x d where d = # of filters
    - subsample  : subsampling rate to only return a percentage of pixel features
Outputs:
    - features   : 2D numpy array containing the filter responses of imageStack using F
'''


def getFilterResponses(imageStack, F, subsample=0.05):
    features = []
    for image in imageStack:
        N, M, d = image.shape[0], image.shape[1], F.shape[2]
        responses = np.zeros((N, M, d))
        for i in range(d):
            filter = F[:, :, i]
            responses[:, :, i] = scipy.ndimage.convolve(image, filter)

        responses = responses.reshape(N * M, d)
        sample_idx = np.random.choice(N * M, int(N * M * subsample), replace=False)
        sampled_responses = responses[sample_idx, :]
        features.append(sampled_responses)
    features = np.concatenate(features, axis=0)
    return features


# PART B
'''
Inputs:
    - filterResponses : 2D numpy array containing a set of filter responses from pixels. 
                            - Each row should contain d filter responses where d = size of filter bank used
    - numTextons      : int, specifying the number of cluster centers to find from filterResponses
Outputs:
    - textons         : scikit-learn KMeans module that has been fitted on filterResponses
'''
from sklearn.cluster import KMeans

def createTextons(filterResponses, numTextons=50):
    # Now fit on filterResponses
    textons = KMeans(n_clusters=numTextons, random_state=0).fit(filterResponses)

    return textons



'''
Inputs:
    - filterResponses : 2D numpy array containing a set of filter responses from pixels. 
                            - Each row should contain d filter responses where d = size of filter bank used
    - numTextons      : int, specifying the number of cluster centers to find from filterResponses
Outputs:
    - textons         : scikit-learn KMeans module that has been fitted on filterResponses
'''


def getTextonFeatures(image, textons, F, window_size=7):
    N, M, d = image.shape[0], image.shape[1], F.shape[2]
    k = textons.n_clusters

    print("Step 1...")
    # Step (1)
    responses = np.zeros((N, M, d))
    for i in range(d):
        filter = F[:, :, i]
        responses[:, :, i] = scipy.ndimage.convolve(image, filter)

    print("Step 2...")
    # Step (2)
    # add in as much code to obtain a final quantized_responses array
    quantized_responses =responses.reshape((N*M, d))
    predicts = textons.predict(quantized_responses)
    quantized_responses= predicts.reshape((N, M))
    print("Step 3...")
    # Step (3)
    histograms = np.zeros((N, M, k))
    for i in range(N):
        for j in range(M):
            # current histogram should have k entries where:
            #    i-th element will contain the # of times a quantized centroid occurs
            hist = np.array([0 for i in range(k)])

            # go through the neighborhood
            for x in range(max(0, i - window_size // 2), min(N, i + window_size // 2 + 1)):
                for y in range(max(0, j - window_size // 2), min(M, j + window_size // 2 + 1)):
                    # hist is the array of range k
                    current_quantized =  quantized_responses[x, y]# collect the neighborhood stats
                    # don't forget to update hist
                    hist[current_quantized] = hist[current_quantized] + 1

            # set the pixel's completed histogram
            histograms[i, j] = hist

    return histograms


'''
Inputs:
    - features     : 3D numpy array containing a set of features for each pixel in an image
    - k            : int, specifying the number of image segments desired
Outputs:
    - segmentation : 2D numpy array containing a label for each pixel based on features
'''


def createSegmentation(features, k=5):
    N, M, d = features.shape
    #kmeans = createTextons(features, numTextons=k) # create kmeans module
    quantized_responses = features.reshape((N * M, d))
    textons = KMeans(n_clusters=k, random_state=0).fit(quantized_responses)

    #quantized_responses = features.reshape((N * M, d))
    predicts = textons.predict(quantized_responses)
    quantized_responses2 = predicts.reshape((N, M))
    #quantized_responses = scipy.misc.imresize(features, (N * M, d))
    #predicts = kmeans.predict(quantized_responses)
    #quantized_responses = scipy.misc.imresize(predicts, (N* M, d))

    # first fit then predict label of each pixel

    return quantized_responses2 # return segmentation of size N*M x d

# PART 5

imageStack = loadImages(grayscale=False)
imageStackGray = loadImages(grayscale=True)
F = loadFilterBank()
image = 0
imageGray = 0
F_responses = getFilterResponses(imageStackGray, F)
textons = createTextons(F_responses)

i =0
for image in imageStackGray:
    histograms = getTextonFeatures(image, textons, F)
    segment = createSegmentation(histograms)

    # Color
    colorImage = imageStack[i]
    segmentColor = createSegmentation(colorImage)
    plt.imshow(colorImage)
    plt.imshow(segment)
    plt.imshow(segmentColor)






    i = i + 1



