import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
np.random.seed(0)


# a function to calculate the gammas/responsibilities
def get_gammas(x, m_k, s_k, p_k):
    N = x.shape[0]
    K = p_k.shape[0]

    # at every step of the algorithm gamma is recomputed through the new values of p_k, m_k, s_k
    # at the first step we compute it with the randomly initialized p_k, m_k, s_k
    gammas = np.zeros((N, K))
    for k in range(K):
        denominator = 1 / np.sqrt(2 * np.pi * s_k[k])
        numerator = np.exp(-np.power((x - m_k[k]), 2) / (2 * s_k[k]))
        gammas[:, k] = p_k[k] * np.prod(denominator * numerator, axis=1)

    denominator_gamma = np.sum(gammas, axis=1)
    # 2nd
    for k in range(K):
        gammas[:, k] = gammas[:, k] / denominator_gamma

    return gammas, denominator_gamma


# a function to calculate the reconstruction error of the image
def get_reconstruction_error(og_image, reconstructed_img):
    error = np.sum((np.linalg.norm(og_image - reconstructed_img, axis=1) ** 2)) / og_image.shape[0]
    return error


# a function to calculate the maximum log-likelihood---p_x is the denominator_gamma from the get_gammas function
def get_log_likelihood(p_x):
    tmp = np.log(p_x)
    return np.sum(tmp)


# a function that calculates the log-likelihood using the logsumexp trick
def get_log_likelihood_trick(f_s, m):
    tmp = np.log(np.sum(np.exp(f_s - m[:, np.newaxis]), axis=1))
    return np.sum(m + tmp, axis=0)


# create f_s to use them for the logsumexp trick
def calculate_f_s(X, p_k, m_k, s_k):
    K = s_k.shape[0]

    # for every pixel we calculate k -> f_S
    f_s = np.zeros((X.shape[0], K))
    for k in range(K):
        m = m_k[k, :]
        tmp = np.sum(((X - m)**2 / s_k[k]) + np.log(2*np.pi*s_k[k]), axis=1)
        f_s[:, k] = np.log(p_k[k]) - (tmp / 2)

    m = f_s.max(axis=1)
    return f_s, m


# a function to train the EM algorithm
def train(X, k, iterations):
    N = X.shape[0]
    D = X.shape[1]

    # initialize parameters to random numbers
    m_k, s_k, p_k = initialize_parameters(k, D)
    tolerance = 1e-6

    # initialize the first loss to -Inf so the first loss will be larger.
    loss_old = -np.Inf
    gammas = None

    # Algorithm: (1) E-step  (2) M-step  (3) Calculate Loss
    for _ in tqdm(range(iterations)):

        # E-step
        gammas, _ = get_gammas(X, m_k, s_k, p_k)

        # M-step
        # update the parameters / at the first iteration we update with the random gammas (from the random m_k, s_k, p_k)
        m_k, s_k, p_k = update_parameters(gammas, X)

        # # calculate the new loss with the updated parameters
        f, m = calculate_f_s(X, p_k, m_k, s_k)
        loss_new = get_log_likelihood_trick(f, m)

        if loss_new - loss_old < 0:
            print('There is an error in the implementation!!!')
            print('The losses are: {}/{}'.format(loss_new, loss_old))
            exit(0)

        if np.abs(loss_new - loss_old) < tolerance:
            print('Tolerance reached')
            print('The losses are: {}/{}'.format(loss_new, loss_old))
            print(np.sum(p_k))
            return gammas, m_k

        print('The losses are: {}/{}'.format(loss_new, loss_old))
        loss_old = loss_new

    print('Maximum number of iterations reached!!')
    return gammas, m_k


# a function that executes a step of the EM algorithm
# def EM_step(X, m_k, p_k, s_k):
#
#     # calculate the new responsibilities
#     gammas, p_x = get_gammas(X, p_k, m_k, s_k)
#
#     # update the parameters
#     m_k, p_k, s_k = update_parameters(gammas, X)
#
#     return gammas, p_x, m_k, p_k, s_k


# a function to initialize all the values that we will need
def initialize_parameters(k, d):

    # initialize the means m_k
    m_k = np.random.uniform(0.1, 0.9, (k, d))

    # initialize the variance
    s_k = np.random.uniform(0.1, 0.9, k)

    # initialize the mixing coefficients p_k
    # fill the array with numbers = 1/clusters, so they sum up to 1.
    p_k = np.full(k, 1/k)

    return m_k, s_k, p_k


# a function to update the parameters
def update_parameters(gammas, X):
    N = X.shape[0]
    D = X.shape[1]
    K = gammas.shape[1]

    m_k_new = np.zeros((K, D))
    p_k_new = np.zeros(K)
    s_k_new = np.zeros(K)

    # sum up all the gammas for each x_n
    N_k = np.sum(gammas, axis=0)

    # for each K
    for k in range(K):
        # for the new m_k (means)
        gamma_k = gammas[:, k]
        # print(gamma_k.shape)
        for d in range(D):
            m_k_new[k, d] = np.sum(gamma_k * X[:, d], axis=0) / N_k[k]

        # for the new s_k (variance)
        tmp = np.power((X - m_k_new[k, :]), 2)
        tmp = np.sum(tmp, axis=1)
        s_k_new[k] = np.sum(tmp * gamma_k, axis=0) / (N_k[k] * D)

        # for the new p_k (mixing coefficients)
        p_k_new[k] = N_k[k] / N

    return m_k_new, s_k_new, p_k_new


# a function to run the experiments for various K
def experiments(k, X):
    # train for a certain k
    gammas, m_k = train(X, k, iterations=100)
    #
    reconstructed_img = np.zeros(X.shape)
    # for each pixel
    for i in range(X.shape[0]):
        # find the gamma with the maximum value for this particular pixel
        idx = gammas[i].argmax()
        # get the corresponding mean / where the 3 values of the mean represent the RGB values.
        reconstructed_img[i] = m_k[idx]

    og_image = X
    error = get_reconstruction_error(og_image, reconstructed_img)
    print('The reconstruction error for {} is: {}'.format(k, error))

    # reshape the image to its original shape
    reconstructed_img = reconstructed_img.reshape(og_height, og_width, dimensions)
    plt.imshow(reconstructed_img)
    plt.savefig("reconstructed_img_{}.jpg".format(k))


# image = plt.imread('drive/My Drive/MachineLearning/GMM_EM/Image_Segmentation_Compression_GMM/im.jpg')
image = plt.imread('im.jpg')
print('Image dimensions: ({},{})'.format(image.shape[0], image.shape[1]))
og_height = image.shape[0]
og_width = image.shape[1]
dimensions = image.shape[2]
pixels = og_height * og_width

data = image.reshape(pixels, dimensions)
# normalize the data
data = data / 255
experiments(k=2, X=data)
