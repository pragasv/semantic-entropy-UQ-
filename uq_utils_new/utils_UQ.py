import numpy as np


def try_rishab_UQ(un_perturb_eigen_vectors, sigma_kernel = 0.6):
    L, M = un_perturb_eigen_vectors.shape

    ratio_main = np.zeros((L,M), dtype=np.complex64)

    for i in range(L):
        input_array = np.abs(un_perturb_eigen_vectors[i,:])
        laplacian_array = np.gradient(np.gradient(input_array))

        ratio_array = ((sigma_kernel**2)/2) * laplacian_array/input_array

        ratio_main[i,:] = ratio_array - np.min(ratio_array)
    
    return ratio_main

def pair_wise_eucledian_dist(unperturbed_eigenvectors_arr):
    """
    Picks the max EV and then proceeds pair wise Eucledian distance 
    """
    N,L,D = unperturbed_eigenvectors_arr.shape
    max_mode_arr = np.zeros((N,L), dtype=np.complex64)

    for i in range(N):
        max_mode_arr[i,:] = np.max(unperturbed_eigenvectors_arr[i,:,:], axis=1)

    pairwise_matrix = np.zeros((N,N), dtype=np.complex64)
    for i in range(N):
        for j in range(N):
            pairwise_matrix[i,j] = np.linalg.norm(max_mode_arr[i,:] - max_mode_arr[j,:], ord=2)
    return pairwise_matrix
