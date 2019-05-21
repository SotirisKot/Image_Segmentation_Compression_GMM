import numpy as np
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
np.random.seed(0)


# a function to calculate the gammas/responsibilities
def get_gammas(x, p_k, m_k, s_k):
    N = x.shape[0]
    K = p_k.shape[0]

    # at every step of the algorithm gamma is recomputed through the new values of p_k, m_k, s_k
    gammas = np.zeros((N, K))
    for k in range(K):
        denominator = 1 / np.sqrt(2 * np.pi * s_k[k])
        numerator = np.exp(- np.power(x - m_k[k], 2) / (2 * s_k[k]))
        gammas[:, k] = p_k[k] * np.prod(numerator * denominator, axis=1)

    denominator_gamma = np.sum(gammas, axis=1)

    # check if these two ways are equal TODO
    # 1st
    # gammas = gammas / denominator_gamma
    # print(gammas)

    # 2nd
    for k in range(K):
        gammas[:, k] = gammas[:, k] / denominator_gamma
    # print(gammas)

    return gammas, denominator_gamma


# a function to calculate the reconstruction error of the image
def get_reconstruction_error(N, og_image, reconstructed_img):
    diff = np.power(og_image - reconstructed_img, 2)
    sum_diff = np.sum(diff)
    error = sum_diff / N
    return error


# a function to calculate the maximum log-likelihood
def get_log_likelihood(p_x):
    tmp = np.log(p_x)
    return np.sum(tmp, axis=0)


# a function to visualize the image
def visualize_image():
    pass


# a function to train the EM algorithm
def train(X, k, iterations):
    N = X.shape[0]
    D = X.shape[1]

    # initialize parameters to random numbers
    m_k, s_k, p_k = initialize_parameters(k, D)
    tolerance = 1e-6

    # # calculate the initial gammas to obtain p(x)-to be used for the first calculation of log-likelihood
    # gammas, p_x = get_gammas(X, p_k, m_k, s_k)
    loss_old = -1e6
    gammas = None

    for _ in tqdm(range(iterations)):

        # calculate the old loss/first loss when we are at the first iteration
        # loss_old = get_log_likelihood(p_x)

        # execute a step of the EM algorithm
        gammas, p_x, m_k, p_k, s_k = EM_step(X, m_k, p_k, s_k)

        # calculate the new loss with the updated parameters
        loss_new = get_log_likelihood(p_x)

        if loss_new - loss_old < 0:
            print('There is an error in the implementation!!!')
            exit(0)

        if np.abs(loss_new - loss_old) < tolerance:
            print('Tolerance reached')
            return gammas, m_k

        print('The losses are: {}/{}'.format(loss_new, loss_old))
        loss_old = loss_new

    print('Maximum number of iterations reached!!')
    return gammas, m_k


# a function that executes a step of the EM algorithm
def EM_step(X, m_k, p_k, s_k):

    # calculate the new responsibilities
    gammas, p_x = get_gammas(X, p_k, m_k, s_k)

    # update the parameters
    m_k, p_k, s_k = update_parameters(gammas, X)

    return gammas, p_x, m_k, p_k, s_k


# a function to initialize all the values that we will need
def initialize_parameters(k, d):

    # initialize the means m_k
    m_k = np.ones((k, d))
    for i in range(k):
        m_k[i, :] = np.random.uniform(0.1, 0.9, d)

    # initialize the variance
    s_k = np.random.uniform(0.2, 0.8, k)

    # initialize the mixing coefficients p_k
    # fill the array with numbers = 1/clusters, so they sum up to 1.
    p_k = np.full(k, 1/k)

    # print(type(m_k))
    # print(type(s_k))
    # print(type(p_k))
    # print(40*'@')
    #
    # print(m_k)
    # print(40*'-')
    # print(s_k)
    # print(40*'-')
    # print(p_k)
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
    gammas_sum = np.sum(gammas, axis=0)

    # for each K
    for k in range(K):
        # for the new m_k (means)
        gamma_k = gammas[:, k]
        for d in range(D):
            m_k_new[k, d] = np.sum(gamma_k * X[:, d], axis=0) / gammas_sum[k]

        # for the new s_k (variance)
        tmp = np.power(X - m_k_new[k, :], 2)
        tmp = np.sum(tmp, axis=1) * gamma_k
        tmp = np.sum(tmp, axis=0)

        s_k_new[k] = tmp / (gammas_sum[k] * D)

        # for the new p_k (mixing coefficients)
        p_k_new[k] = gammas_sum[k] / N

    return m_k_new, p_k_new, s_k_new


# a function to run the experiments for various K
def experiments():
    # we will start at K = 2
    k = 64
    while k <= 64:

        # train for a certain k
        gammas, m_k = train(X, k, iterations=20)
        print(np.sum(gammas, axis=1))
        print(m_k)
        reconstructed_img = np.zeros(X.shape)
        for i in range(X.shape[0]):
            idx = gammas[i].argmax()
            reconstructed_img[i] = m_k[idx]

        N = X.shape[0]
        og_image = X
        error = get_reconstruction_error(N, og_image, reconstructed_img)
        print('The reconstruction error for {} is: {}'.format(k, error))

        # reshape the image to its original shape
        reconstructed_img = reconstructed_img.reshape(og_height, og_width, dimensions)
        plt.imshow(reconstructed_img)
        plt.savefig("reconstructed_img_{}.jpg".format(k))
        k *= 2


# initialize_parameters(10, 3)
image = plt.imread('im.jpg')
print('Image dimensions: ({},{})'.format(image.shape[0], image.shape[1]))
og_height = image.shape[0]
og_width = image.shape[1]
dimensions = image.shape[2]
pixels = og_height * og_width

data = image.reshape(pixels, dimensions)
# normalize the data
X = data / 255
experiments()
