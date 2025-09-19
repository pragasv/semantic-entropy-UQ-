import numpy as np
from scipy.linalg import svd
from numpy.linalg import qr
from scipy import signal
import scipy.sparse.linalg as arp

# Python Plotting
from matplotlib import pyplot as plt
from scipy.linalg import null_space
from scipy.linalg import sqrtm, eig
from scipy.linalg import lstsq

from tqdm import tqdm
import itertools
import cvxpy as cp


def gaussian_function(x, x_center, sigma):
    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x - x_center) ** 2 / (2 * sigma ** 2))

def generate_sequences(max_bits, elements, L):
    count = 0
    for i in range(1, max_bits + 1):
        for seq in itertools.product(elements, repeat=i):
            yield seq
            count += 1
            if count == L:
                return

def generate_hamiltonians(max_bits, elements, L):

    # combinations = generate_combinations(num_operators)
    combinations = list(generate_sequences(max_bits, elements, L))

    # Define Pauli matrices
    sX = np.array([[0, 1], [1, 0]])
    sY = np.array([[0, -1j], [1j, 0]])
    sZ = np.array([[1, 0], [0, -1]])
    sI = np.eye(2)

    # Initialize list to store Hamiltonians
    hamiltonians = []

    for comb in combinations:
        # print(comb)

        rotation_count = max_bits-len(comb)+1

        hamiltonians_sub = np.zeros((2**max_bits, 2**max_bits), dtype=np.complex_)
        count_identity_before = 0
        

        for k in range(rotation_count):
            hamiltonian_list = []
            count_identity_after = int(max_bits - len(comb) - count_identity_before)

            for id_c in range(count_identity_before):
                hamiltonian_list.append(sI)

            for term in comb: 
                if term == 1:
                    hamiltonian_list.append(sX)
                elif term == 2:
                    hamiltonian_list.append(sY)
                elif term == 3:
                    hamiltonian_list.append(sZ)
                elif term == 0:
                    hamiltonian_list.append(sI)

            for id_c in range(count_identity_after):
                hamiltonian_list.append(sI)

            for i in range(len(hamiltonian_list)):
                if i==0:
                    # initialize hamiltonian 
                    hamiltonian =  hamiltonian_list[i]
                else:
                    hamiltonian = np.kron(hamiltonian, hamiltonian_list[i])
            
            hamiltonians_sub += hamiltonian
            count_identity_before += 1
            count_identity_after -= 1

        ## normalize hamiltonian such that <h_i|h_i> = 1
        trace_H_i = np.trace(np.dot(hamiltonians_sub, np.conj(hamiltonians_sub).T))

        # Normalize each row by dividing by the norms
        hamiltonians_sub = hamiltonians_sub / np.sqrt(trace_H_i)

        hamiltonians.append(hamiltonians_sub)

    return hamiltonians

    
def expectation_value(H, psi_T):
    numerator = np.matmul(np.transpose(np.conjugate(psi_T)), np.matmul(H, psi_T))
    denominator = np.matmul(np.transpose(np.conjugate(psi_T)), psi_T)
    return numerator / denominator

def generate_QCM(hamiltonians, psi_T): 
    N = len(hamiltonians)


    QCM = np.zeros([N,N], dtype=complex)

    for i in tqdm(range(N)):
        for j in range(N):

            # anti-commutator operator in the middle
            second_element =  hamiltonians[j]
            first_term = expectation_value((np.matmul(hamiltonians[i], second_element) + np.matmul(second_element, hamiltonians[i])), psi_T)

            QCM[i,j] =  0.5*first_term - (expectation_value(hamiltonians[i], psi_T))*(expectation_value(second_element , psi_T))


    return QCM

def generate_QCM_upper_triangular(hamiltonians, psi_T):
    N = len(hamiltonians)

    # Initialize QCM matrix
    QCM = np.zeros([N, N], dtype=complex)

    # Iterate only over the upper triangular part, including the diagonal
    for i in range(N):
        for j in range(i, N):
            second_element = hamiltonians[j]

            # Calculate the first term (anti-commutator expectation)
            first_term = expectation_value(
                (np.matmul(hamiltonians[i], second_element) + np.matmul(second_element, hamiltonians[i])),
                psi_T
            )

            # Compute the matrix element
            QCM[i, j] = 0.5 * first_term - (
                expectation_value(hamiltonians[i], psi_T) * expectation_value(second_element, psi_T)
            )

            # Since the matrix is symmetric, mirror the value
            if i != j:
                QCM[j, i] = QCM[i, j]

    return QCM

def pick_significant_singular_values(S,V, Threshold, mode=None):
    # pick only the singular values which are less than a threshold 
    # 
    if mode==None:
        null_space_indices = np.where(S <= Threshold)[0]

        # Extract the corresponding null space vectors from V
        null_space_vector = V[null_space_indices]

    elif mode=="Lowest":
        null_space_indices = np.where(S <= min(S))[0]

        # Extract the corresponding null space vectors from V
        null_space_vector = V[null_space_indices]

    return null_space_vector



def generate_hamiltonian_set(hamiltonians, S,V, Threshold=1e-8, mode=None):
    # for the moment generates only the direct hamiltonians from the null space vector - omitts the linear combination
    null_space_vector = pick_significant_singular_values(S,V, Threshold, mode=mode)

    hamiltonians_array = np.array(hamiltonians)

    hamiltonian_solutions = np.dot(hamiltonians_array.T, null_space_vector.T).T

    return hamiltonian_solutions


def variance_check(H_reconstruct_eigs, reshaped_psi, threshold=3):
## variance check of H 

    H_expectation = np.dot(reshaped_psi.T, np.dot(H_reconstruct_eigs, reshaped_psi))

    # Compute the expectation value of the squared Hamiltonian: <H^2>
    H_squared_expectation = np.dot(reshaped_psi.T, np.dot(np.dot(H_reconstruct_eigs,H_reconstruct_eigs.T), reshaped_psi))

    # Compute the variance of the Hamiltonian: Var(H) = <H^2> - <H>^2
    variance_H = H_squared_expectation - H_expectation**2

    if variance_H > threshold:
        print("Expectation value of the Hamiltonian (<H>):", H_expectation)
        print("Expectation value of the squared Hamiltonian (<H^2>):", H_squared_expectation)
        print("Variance of the Hamiltonian (Var(H)) via eigs:", variance_H)
        

def threshold_complex_elements(matrix, threshold=1e-10):
    # Create a copy of the input matrix
    thresholded_matrix = matrix.copy()
    
    # Find the indices of elements with absolute value below the threshold
    below_threshold_indices = np.abs(thresholded_matrix) < threshold
    
    # Set those elements to zero
    thresholded_matrix[below_threshold_indices] = 0

    # threshold imaginary parts and real seperate 
    thresholded_matrix_imag = thresholded_matrix.imag
    thresholded_matrix_real = thresholded_matrix.real
    below_threshold_indices_real = np.abs(thresholded_matrix_real) < threshold
    below_threshold_indices_imag = np.abs(thresholded_matrix_imag) < threshold
    
    try:
        thresholded_matrix_real[below_threshold_indices_real] = 0
    except ValueError:
        pass
        # print("skipping real thresholding to zero as exception occuered")

    try:
        thresholded_matrix_imag[below_threshold_indices_imag] = 0
    except ValueError:
        pass
        # print("skipping imag thresholding to zero as exception occuered")
        

    thresholded_matrix_final = thresholded_matrix_real + thresholded_matrix_imag

    # print("MSE : ", np.mean((thresholded_matrix_final - matrix)**2))
    
    return thresholded_matrix_final

def find_eigen_value_location(hamiltonian_solutions, psi_0): 
    # Solve the eigenvalue problem to find E
    E, psi_eigen = np.linalg.eig(hamiltonian_solutions)

    # Extract the energy eigenvalues (E) and the corresponding eigenfunctions (psi_eigen)
    # Since E is an array, we need to find the corresponding eigenvalue for our wave function
    # We can compare the eigenfunctions to find the one that closely matches our given wave function
    idx = None
    mse_psi = np.mean(np.abs(psi_eigen.real - psi_0), axis=0)
    mini_mse_psi_idx = np.argmin(mse_psi)
    mini_mse_psi = mse_psi[mini_mse_psi_idx]

    if mini_mse_psi < 9e-2:
        energy_eigenvalue = E[mini_mse_psi_idx]
        idx=mini_mse_psi_idx
        return idx
    else:
        raise ValueError("psi is not part of H, MSE:",mini_mse_psi)


def find_smallest_energy_hamiltonian(hamiltonians_array, eigenvectors_QCM, null_space_idx, psi_0):
    min_energy_idx = np.inf

    energy_index_arr = []

    for i in null_space_idx:
        smallest_eigenvector_QCM = eigenvectors_QCM[:, i]
        hamiltonian_solutions = np.dot(hamiltonians_array.T, smallest_eigenvector_QCM.T).T

        energy_idx = find_eigen_value_location(hamiltonian_solutions, psi_0)
        energy_index_arr.append(energy_idx)

        if energy_idx < min_energy_idx:
            min_energy_idx = energy_idx
            smallest_energy_idx = i

    # print("psi is expressed as the EV index :", min_energy_idx)
    return smallest_energy_idx



##### still in works ##### 
def find_smallest_energy_hamiltonian_via_innerproduct(hamiltonians_array, eigenvectors_QCM, null_space_idx, psi_0):
    '''
    Uses the approach psi_0^T H_i psi_0 = lambda_i
    to find the smallest lambda_i
    '''
    smallest_eigenvector_QCM = eigenvectors_QCM[:, null_space_idx]
    hamiltonian_solutions = np.dot(hamiltonians_array.T, smallest_eigenvector_QCM).T   ### 109 X 128 X 128 


    matrix_first_expanded = psi_0.T[np.newaxis, :, :]
    result_1 = np.matmul(matrix_first_expanded, hamiltonian_solutions) 
    final_results = np.matmul(result_1, psi_0)

    smallest_energy_idx = np.argmin(final_results.flatten())

    return smallest_energy_idx


def generate_hamiltonian_set_eigenvalue_base(hamiltonians, QCM, Threshold=1e-10, mode=None, psi_0=None, return_null_space_vector=False):
    # for the moment generates only the direct hamiltonians from the null space vector - omitts the linear combination
    # if psi_0 is given we will fetch the smallest energy related hamiltonian 
    hamiltonians_array = np.array(hamiltonians)
    
    if mode==None:
        if not np.allclose(QCM, np.conjugate(QCM).T):
            raise ValueError("QCM not hermitian")

        eigenvalues_QCM, eigenvectors_QCM = np.linalg.eigh(QCM)
        eigenvalues_QCM = threshold_complex_elements(eigenvalues_QCM)
        eigenvectors_QCM = threshold_complex_elements(eigenvectors_QCM)

        null_space_idx = np.where(eigenvalues_QCM==0)[0]
        # print("# of eigen values zero",len(null_space_idx))


        if isinstance(psi_0, np.ndarray):
            ## psi is given
            if len(null_space_idx)==1:
                smallest_eigenvalue_index = np.argmin(eigenvalues_QCM)
            elif len(null_space_idx) > 1:
                # smallest_eigenvalue_index = find_smallest_energy_hamiltonian(hamiltonians_array, eigenvectors_QCM, null_space_idx, psi_0)
                smallest_eigenvalue_index = find_smallest_energy_hamiltonian_via_innerproduct(hamiltonians_array, eigenvectors_QCM, null_space_idx, psi_0)
            else:
                ### psi 0 given but no null space 
                smallest_eigenvalue_index = np.argmin(eigenvalues_QCM)
        else:
            ## psi_0 not
            # Find the index of the smallest eigenvalue
            smallest_eigenvalue_index = np.argmin(eigenvalues_QCM)

        # Extract the corresponding eigenvector
        smallest_eigenvector_QCM = eigenvectors_QCM[:, smallest_eigenvalue_index]

        smallest_eigen_value = eigenvalues_QCM[smallest_eigenvalue_index]
        # print(("smallest eigen val for psi expressed at idx(%d): %f"%(smallest_eigenvalue_index, smallest_eigen_value)))

        if np.abs(smallest_eigen_value)>Threshold:
            # raise ValueError("smallest eigen val is very high: %f"%smallest_eigen_value)
            print(("smallest eigen val is very high: %f"%smallest_eigen_value))

    elif mode=="threshold_SVD":
        U, s, VT = np.linalg.svd(QCM)

        # Identify small singular values
        epsilon = min(s) # Set a threshold for small singular values
        null_mask = s <= epsilon

        # Extract right singular vectors corresponding to small singular values
        smallest_eigenvector_QCM = VT[null_mask]
        smallest_eigenvector_QCM = smallest_eigenvector_QCM[0]

    # hamiltonians_matrix = np.transpose(hamiltonians, (1, 2, 0))

    hamiltonian_solutions = np.dot(hamiltonians_array.T, smallest_eigenvector_QCM.T).T

    hamiltonian_solutions = hamiltonian_solutions.reshape(1, hamiltonian_solutions.shape[0], hamiltonian_solutions.shape[1])
    # hamiltonian_solutions = np.sum(hamiltonians_matrix*null_space_vector, axis=2)

    #### variance check

    if return_null_space_vector:
        return hamiltonian_solutions, smallest_eigenvector_QCM
    else:
        return hamiltonian_solutions


def sort_eigen_vectors(eigenvalues, eigenvectors, reshaped_psi, F=6, approach=None, return_idx=False):
    """
    This code sorts and picks only the smallest F number of eiegen values and their corresponding eigen vector
    """

    slice_range_post = int((F/2))
    slice_range_pre = int((F/2))

    # Sort eigenvalues and eigenvectors based on eigenvalues
    if approach==None:
        sorted_indices = np.argsort(eigenvalues)
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        # Pick the smallest F eigenvalues and eigenvectors
        smallest_eigenvalues = sorted_eigenvalues[:F]
        smallest_eigenvectors = sorted_eigenvectors[:, :F]

    elif approach=="mini_mse":
        ### pick the nextF number of modes which are next to eigen
        min_mse=np.Inf
        mse_array = np.array([])

        for i in range(len(eigenvectors)):
            mse_now = np.mean((eigenvectors[:,i] - reshaped_psi)**2)

            # print(mse_now)
            if min_mse > mse_now:
                min_mse = mse_now
            
            mse_array = np.append(mse_array, mse_now)

        loc_min_mse = np.argmin(mse_array)

        diff_location = len(eigenvectors) - loc_min_mse
        if diff_location >=F:
            adjustment=0
        else:
            ### this is when the loc is too close to the length of the eigen vector
            adjustment = F-diff_location


        smallest_eigenvalues = eigenvalues[loc_min_mse-adjustment - slice_range_pre:loc_min_mse+ slice_range_post -adjustment]
        smallest_eigenvectors = eigenvectors[:, loc_min_mse-adjustment - slice_range_pre:loc_min_mse+ slice_range_post -adjustment]
    
    elif approach=="mini_mae":
        ### pick the nextF number of modes which are next to eigen
        mse_array = np.array([])

        for i in range(len(eigenvectors)):
            mse_now = np.mean(np.abs(eigenvectors[:,i] - reshaped_psi))           
            mse_array = np.append(mse_array, mse_now)

        loc_min_mse = np.argmin(mse_array)

        diff_location = len(eigenvectors) - loc_min_mse
        if diff_location >= int(F/2):
            adjustment=0
        else:
            ### this is when the loc is too close to the length of the eigen vector
            adjustment = F-diff_location
        
        ## slice range is too big for the location min_mse
        if loc_min_mse - adjustment - slice_range_pre<0:
            slice_range_pre = loc_min_mse- adjustment ## we start from zero
            slice_range_post =  F - slice_range_pre

        elif loc_min_mse + slice_range_post - adjustment>=len(eigenvectors):
            slice_range_post = len(eigenvectors) - (loc_min_mse - adjustment)
            slice_range_pre = F - slice_range_post


        smallest_eigenvalues = eigenvalues[loc_min_mse- adjustment - slice_range_pre : loc_min_mse+slice_range_post -adjustment]
        smallest_eigenvectors = eigenvectors[:, loc_min_mse-adjustment - slice_range_pre:loc_min_mse+ slice_range_post-adjustment]

    if return_idx:
        return smallest_eigenvalues, smallest_eigenvectors, (loc_min_mse-adjustment- slice_range_pre, loc_min_mse+ slice_range_post-adjustment)
    else:
        return smallest_eigenvalues, smallest_eigenvectors


def check_commutive_prop(delta_H):
    delta_H = delta_H[0,:,:]
    
    n = len(delta_H)  # Assuming a 16x16 matrix for a 16 spin system

    # Create an identity matrix of size n
    S = np.eye(n)

    # Reverse the order of rows to create the symmetry operator
    S = S[::-1]

    delta_H_prime = 0.5*(delta_H + np.matmul(S, np.matmul(delta_H, S)))
    
    commutator = np.dot(delta_H_prime, S) - np.dot(S, delta_H_prime)
    # print("Norm of commutator (should be close to 0):", np.linalg.norm(commutator))

    return delta_H_prime.reshape(1, n, n)

def generate_normal_pertubation(H,  mu=0, sigma=1, method="random"):
    """
    Generate an array of size (N, M, L) from a normal distribution with specified mean and standard deviation.

    Parameters:
    mu (float): Mean of the normal distribution.
    sigma (float): Standard deviation of the normal distribution.
    N (int): Number of elements along the first dimension.
    M (int): Number of elements along the second dimension.
    L (int): Number of elements along the third dimension.

    Returns:
    numpy.ndarray: Array of size (N, M, L) sampled from the normal distribution.
    Normalize the matrix to haveunit integral 
    """
    N,M,L = H.shape

    if method=="random":
        delta_H = np.random.normal(mu, sigma, size=(N, M, L))
        delta_H = (delta_H + delta_H.T) / 2 # make Hermitian matrix
    elif method=="sym_amp":
        delta_H = np.random.normal(mu, sigma, size=(N, M, L))
        delta_H = check_commutive_prop(delta_H)

    integrals = np.sum(delta_H, axis=(1,2))

    normalized_delta_H = delta_H/integrals[:, None, None]
    return normalized_delta_H


def apply_perubation(H, reshaped_psi, lambda_value=0.001, number_of_modes=6):
        
    delta_H = generate_normal_pertubation(H,  mu=0, sigma=1)

    perturbed_H = H + lambda_value*delta_H

    N, M, L = H.shape

    # Initialize lists to store eigenvalues and eigenvectors
    unperturbed_eigenvalues_list = []
    unperturbed_eigenvectors_list = []

    perturbed_eigenvalues_list = []
    perturbed_eigenvectors_list = []

    
    # Iterate over each MxL matrix
    for i in range(N):
        # Find eigenvalues and eigenvectors of the i-th MxL matrix
        eigenvalues, eigenvectors = np.linalg.eig(H[i])
        eigenvalues, eigenvectors = sort_eigen_vectors(eigenvalues, eigenvectors, reshaped_psi, F=number_of_modes)
        
        # Append eigenvalues and eigenvectors to the lists
        unperturbed_eigenvalues_list.append(eigenvalues)
        unperturbed_eigenvectors_list.append(eigenvectors)

        # Find eigenvalues and eigenvectors of the i-th MxL matrix
        eigenvalues, eigenvectors = np.linalg.eig(perturbed_H[i])
        eigenvalues, eigenvectors = sort_eigen_vectors(eigenvalues, eigenvectors, reshaped_psi, F=number_of_modes)

        # Append eigenvalues and eigenvectors to the lists
        perturbed_eigenvalues_list.append(eigenvalues)
        perturbed_eigenvectors_list.append(eigenvectors)
    
    # Convert lists to numpy arrays
    unperturbed_eigenvalues_array = np.array(unperturbed_eigenvalues_list)
    unperturbed_eigenvectors_array = np.array(unperturbed_eigenvectors_list)

    perturbed_eigenvalues_array = np.array(perturbed_eigenvalues_list)
    perturbed_eigenvectors_array = np.array(perturbed_eigenvectors_list)

    return  unperturbed_eigenvalues_array, unperturbed_eigenvectors_array, perturbed_eigenvalues_array, perturbed_eigenvectors_array


def least_squares_solve(A, B):
    """
    Solve for AW = b
    Compute a vector x such that the 2-norm |b - A x| is minimized.
    """
    W_1, residuals, rank, s = lstsq(A, B)

    return W_1


def least_squares_solve_qp(A, B):
    """
    Solve the least squares problem AW = B with the constraint that W must be real.
    Uses quadratic programming (QP) to enforce real values while considering complex inputs.
    """
    # Define W as a real-valued variable
    W_real = cp.Variable(A.shape[1])  # Enforce real constraint
    
    # Quadratic objective function: ||A W - B||^2
    objective = cp.Minimize(cp.norm(A @ W_real - B, 2))
    
    # Solve the constrained least squares problem
    problem = cp.Problem(objective)
    problem.solve(solver=cp.SCS)

    return W_real.value


def perturb_KME_type_2(H, reshaped_psi, hamiltonians_array, lambda_value=0.001, perturbation_amplitude=0.001, mu=0, sigma=1):

    # Number of states and Hamiltonians
    num_states, state_dim = reshaped_psi.shape
    num_hamiltonians = hamiltonians_array.shape[0]

    # Initialize matrix A and vector b
    A = np.zeros((num_hamiltonians, num_hamiltonians), dtype=np.complex64)  # Excluding the last row for k ≠ N
    b = np.zeros(num_hamiltonians, dtype=np.complex64)

    # Matrix A calculation
    psi_N_0 = reshaped_psi  # psi_N^(0), the last row of reshaped_psi

    # Find E_N_0 
    mse_array = np.array([])
    eigenvalues, eigenvectors = np.linalg.eigh(H[0])    
    for i in range(len(eigenvectors)):
        mse_now = np.mean(np.abs(eigenvectors[:,i] - reshaped_psi))           
        mse_array = np.append(mse_array, mse_now)

    loc_min_mse = np.argmin(mse_array)
    E_N_0 = eigenvalues[loc_min_mse]

    # perturb KME - generated from linear combination of all other eigen vectors 
    weights = np.random.normal(mu, sigma, size=num_states) 
    weights[loc_min_mse] = 0
    psi_N_1 = lambda_value * perturbation_amplitude * (eigenvectors @ weights)
    # psi_N_1 = lambda_value * perturbation_amplitude * np.random.normal(mu, sigma, size=num_states) 


    for i in range(num_hamiltonians):
        H_i_0 = hamiltonians_array[i, :, :]

        for j in range(num_hamiltonians):
            H_j_0 = hamiltonians_array[j, :, :]
            
            A[i, j] = np.vdot(psi_N_0, np.matmul(H_i_0, H_j_0) @ psi_N_0) -  (np.vdot(psi_N_0, H_i_0 @ psi_N_0) * np.vdot(psi_N_0, H_j_0 @ psi_N_0))  # Inner product calculation

        b[i] = np.vdot(psi_N_0, (np.matmul(H, H_i_0) - np.matmul(H_i_0, H)) @ psi_N_1)

    W_1 = least_squares_solve_qp(A, b)

    return A,b, W_1


def perturb_KME(H, reshaped_psi, hamiltonians_array, lambda_value=0.001, perturbation_amplitude=0.001, mu=0, sigma=1):

    # Number of states and Hamiltonians
    num_states, state_dim = reshaped_psi.shape
    num_hamiltonians = hamiltonians_array.shape[0]

    # Initialize matrix A and vector b
    A = np.zeros((num_states, num_hamiltonians), dtype=np.complex64)  # Excluding the last row for k ≠ N
    b = np.zeros(num_states, dtype=np.complex64)

    # Matrix A calculation
    psi_N_0 = reshaped_psi  # psi_N^(0), the last row of reshaped_psi

    # Find E_N_0 
    mse_array = np.array([])
    eigenvalues, eigenvectors = np.linalg.eigh(H[0])    
    for i in range(len(eigenvectors)):
        mse_now = np.mean(np.abs(eigenvectors[:,i] - reshaped_psi))           
        mse_array = np.append(mse_array, mse_now)

    loc_min_mse = np.argmin(mse_array)
    E_N_0 = eigenvalues[loc_min_mse]

    # perturb KME - generated from linear combination of all other eigen vectors 
    weights = np.random.normal(mu, sigma, size=num_states) 
    weights[loc_min_mse] = 0
    psi_N_1 = lambda_value * perturbation_amplitude * (eigenvectors @ weights)
    # psi_N_1 = lambda_value * perturbation_amplitude * np.random.normal(mu, sigma, size=num_states) 

    
    rows_to_keep = []
    for k in range(len(eigenvectors)):  # For k ≠ N (exclude the last state)
        if k != loc_min_mse:
            psi_k_0 = eigenvectors[:,k]
            for i in range(num_hamiltonians):
                H_i_0 = hamiltonians_array[i, :, :]
                A[k, i] = np.vdot(psi_k_0, H_i_0 @ psi_N_0)  # Inner product calculation
            
            b[k] = -np.vdot(psi_k_0, (H - E_N_0) @ psi_N_1)
            rows_to_keep.append(k)
    
    A = A[rows_to_keep, :] # Excluding the last row for k ≠ N
    b = b[rows_to_keep]

    W_1 = least_squares_solve_qp(A, b)  ## use quadratic programming - should enforce W as real 


    return A,b, W_1

def frame_align(V):
    """
    Compute the frame alignment matrix A from matrix V.
    
    Parameters:
    V : ndarray
        Input complex-valued matrix.
    
    Returns:
    A : ndarray
        Frame alignment matrix.
    """
    # Eqn 4.6
    D = V.T.conj() @ V + (V.T.conj() @ V).T  
    Di = np.linalg.pinv(D)  # Pseudo-inverse of D

    # Near equation 4.10
    psi = np.angle(np.diag(V @ Di @ V.T.conj()))  # Extract phase angles

    # Eqn 4.10 (corrected typo from the paper)
    A = Di @ (V.T @ np.diag(np.exp(-0.5j * psi)) + V.T.conj() @ np.diag(np.exp(0.5j * psi)))

    return A


def apply_perubation_theory(H, reshaped_psi, lambda_value=0.001, number_of_modes=6, mu=0, sigma=1, null_space_vector=None, Hamiltonians_list=None, \
    perurbation_method=None, perturbation_amplitude=0.001, return_raw_values=False, PERTURBATION_APPROACH='type_1'):
    """
    Uses perturbation theory on shrodinger equation
    """

    if perurbation_method=="perturb_nullspace_vector":
        hamiltonians_array = np.array(Hamiltonians_list)
        L = null_space_vector.shape
        
        perturb_w = lambda_value * perturbation_amplitude * np.random.normal(mu, sigma, size=L)
        # perturb_w = perturb_w + null_space_vector
        
        delta_H = np.dot(hamiltonians_array.T, perturb_w.T).T
        delta_H = delta_H.reshape(1, delta_H.shape[0], delta_H.shape[1])

        integrals = np.sum(delta_H, axis=(1,2))
        delta_H = delta_H/integrals[:, None, None]

        perturbed_H = H + delta_H

        if not np.allclose(perturbed_H[0], np.conjugate(perturbed_H[0]).T):
            raise ValueError("perturbed_H not hermitian")

    elif perurbation_method=="perturb_KME_vector":
        hamiltonians_array = np.array(Hamiltonians_list)

        if PERTURBATION_APPROACH == 'type_1':
            A,b, W_1 = perturb_KME(H, reshaped_psi, hamiltonians_array, lambda_value=lambda_value, perturbation_amplitude=perturbation_amplitude, mu=mu, sigma=sigma)
        elif PERTURBATION_APPROACH == 'type_2':
            A,b, W_1 = perturb_KME_type_2(H, reshaped_psi, hamiltonians_array, lambda_value=lambda_value, perturbation_amplitude=perturbation_amplitude, mu=mu, sigma=sigma)
        
        # W_1 = frame_align(W_1)
        delta_H = np.dot(hamiltonians_array.T, W_1.T).T

        delta_H = delta_H.reshape(1, delta_H.shape[0], delta_H.shape[1])
        perturbed_H = H + delta_H


    else:
        delta_H = generate_normal_pertubation(H,  mu=mu, sigma=sigma, method="sym_amp")

        perturbed_H = H + lambda_value* perturbation_amplitude* delta_H

    N, M, L = H.shape

    if N!=1:
        raise ValueError("Too many hamiltonian to handle. Code exists for only 1")

    # Initialize lists to store eigenvalues and eigenvectors
    unperturbed_eigenvalues_list = []
    unperturbed_eigenvectors_list = []

    perturbed_1st_order_eigenvalues_list = []
    perturbed_1st_order_eigenvectors_list = []

    # perturbed_1st_order_energyvalues_list = []
    # perturbed_1st_order_energyvectors_list = []

    perturbed_1st_order_energyvalues_list_all = []
    perturbed_1st_order_energyvectors_list_all = []


    eigenvalues, eigenvectors = np.linalg.eigh(H[0])
    # eigenvalues, eigenvectors = np.linalg.eigh(H[0], UPLO='L')
    
    eigenvalues_sorted, eigenvectors_sorted, (start_idx, end_idx) = sort_eigen_vectors(eigenvalues, eigenvectors, reshaped_psi, F=number_of_modes,  approach="mini_mae", return_idx=True)
    
    # Append eigenvalues and eigenvectors to the lists
    unperturbed_eigenvalues_list.append(eigenvalues_sorted) 
    unperturbed_eigenvectors_list.append(eigenvectors_sorted)

    ## calculate enegery terms of 1st order correction 
    number_of_evs = len(eigenvalues)
    for i in range(number_of_evs): 
        psi_n_0 = eigenvectors[:,i]

        E_n_k = np.dot(np.dot(psi_n_0.T, delta_H[0]),psi_n_0)
        perturbed_1st_order_energyvalues_list_all.append(E_n_k)

    
    for i in range(number_of_evs): 
        runing_sum = np.zeros(psi_n_0.shape)

        for j in range(number_of_evs): 
            if j!=i:
                psi_n_0 = eigenvectors[:,i]
                psi_m_0 = eigenvectors[:,j]

                ## <psi_m| deltH | psi_n> psi_m
                nominator = np.dot(np.dot(np.dot(psi_m_0.T, delta_H[0]), psi_n_0), psi_m_0).T

                ## E_n - E_m
                denominator = perturbed_1st_order_energyvalues_list_all[i] - perturbed_1st_order_energyvalues_list_all[j]

                runing_sum = runing_sum + (nominator/denominator)
                
        perturbed_1st_order_energyvectors_list_all.append(runing_sum)
    
    # Convert lists to numpy arrays
    unperturbed_eigenvalues_array = np.array(unperturbed_eigenvalues_list)
    unperturbed_eigenvectors_array = np.array(unperturbed_eigenvectors_list)

    # pick the EVs near the smallest mse
    perturbed_1st_order_energyvalues_list = perturbed_1st_order_energyvalues_list_all[start_idx:end_idx]
    perturbed_1st_order_energyvectors_list = np.array(perturbed_1st_order_energyvectors_list_all)[start_idx:end_idx, :]

    perturbed_eigenvalues_array = np.array(perturbed_1st_order_energyvalues_list)
    perturbed_eigenvectors_array = np.array(perturbed_1st_order_energyvectors_list)

    perturbed_1st_order_energyvectors_list_all_numpy =  np.array(perturbed_1st_order_energyvectors_list_all)

    if return_raw_values:
        if perurbation_method=="perturb_KME_vector":
            return unperturbed_eigenvalues_array, unperturbed_eigenvectors_array, perturbed_eigenvalues_array, perturbed_eigenvectors_array, perturbed_1st_order_energyvectors_list_all_numpy, start_idx, A, b
        else:
            return unperturbed_eigenvalues_array, unperturbed_eigenvectors_array, perturbed_eigenvalues_array, perturbed_eigenvectors_array, perturbed_1st_order_energyvectors_list_all_numpy, start_idx
    else:
        return unperturbed_eigenvalues_array, unperturbed_eigenvectors_array, perturbed_eigenvalues_array, perturbed_eigenvectors_array



def find_closest_groundstate_hamiltonian(psi_0, psi_0_original, unperturbed_eigenvectors_array):


    norms = np.linalg.norm(unperturbed_eigenvectors_array - psi_0, axis=1)
    min_index = np.unravel_index(np.argmin(norms), norms.shape)

    # Get the value of the smallest element
    min_value = norms[min_index]
    print("ED --> Smallest element: %.4f  && Index of the smallest element: %s"%(min_value, min_index))

    # Plot the 2-norms
    plt.figure(figsize=(8, 6))

    for i in range(norms.shape[1]):
        plt.plot(norms[:, i], label=f'Modes {i+1}')

    plt.title('ED: 2-Norms between psi_0 and eigenvector of hamiltonians')
    plt.xlabel('Vector Index')
    plt.ylabel('2-Norm')
    plt.grid(True)
    plt.legend()
    plt.show()


    norms = np.linalg.norm(unperturbed_eigenvectors_array - psi_0_original, axis=1)
    min_index = np.unravel_index(np.argmin(norms), norms.shape)

    # Get the value of the smallest element
    min_value = norms[min_index]
    print("Original --> Smallest element: %.4f  && Index of the smallest element: %s"%(min_value, min_index))

    # Plot the 2-norms
    plt.figure(figsize=(8, 6))

    for i in range(norms.shape[1]):
        plt.plot(norms[:, i], label=f'Modes {i+1}')

    plt.title('Original: 2-Norms between psi_0 and eigenvector of hamiltonians')
    plt.xlabel('Vector Index')
    plt.ylabel('2-Norm')
    plt.grid(True)
    plt.legend()
    plt.show()


def test_case_function(Hamiltonian_returned, reshaped_psi, lambda_value = 0.001, mu=0,sigma=1, null_space_vector=None, Hamiltonians_list=None,\
     perurbation_method=None, perturbation_amplitude=0.001, return_raw_values=False, PERTURBATION_APPROACH='type_1'):

    number_of_modes = 8
    ITERATE_COUNT = 200

    ## we are only woking with 1 hamiltonian now 
    hamiltonian_selected=0

    MAE_array = np.zeros((ITERATE_COUNT, number_of_modes))
    unperturbed_energy_array = np.zeros((ITERATE_COUNT, number_of_modes), dtype=np.complex64)
    perturbed_energy_array = np.zeros((ITERATE_COUNT, number_of_modes), dtype=np.complex64)
    per_change_out_array = np.zeros((ITERATE_COUNT, number_of_modes, Hamiltonian_returned.shape[1]),dtype=np.complex64)

    unperturbed_eigenvectors_array_all = np.zeros((ITERATE_COUNT, Hamiltonian_returned.shape[1], number_of_modes),dtype=np.complex64)
    perturbed_eigenvectors_array_all = np.zeros((ITERATE_COUNT, number_of_modes, Hamiltonian_returned.shape[1]),dtype=np.complex64)

    perturbed_eigenvectors_array_all_modes_together = np.zeros((ITERATE_COUNT, Hamiltonian_returned.shape[1], Hamiltonian_returned.shape[1]),dtype=np.complex64)
    start_idx_all = np.zeros((ITERATE_COUNT,1))

    if PERTURBATION_APPROACH == 'type_1':
        A_everything = np.zeros((ITERATE_COUNT, Hamiltonian_returned.shape[1] - 1, len(Hamiltonians_list)),dtype=np.complex64)
        b_everything = np.zeros((ITERATE_COUNT, Hamiltonian_returned.shape[1] - 1),dtype=np.complex64)
    elif PERTURBATION_APPROACH == 'type_2':
        A_everything = np.zeros((ITERATE_COUNT, len(Hamiltonians_list), len(Hamiltonians_list)),dtype=np.complex64)
        b_everything = np.zeros((ITERATE_COUNT, len(Hamiltonians_list)),dtype=np.complex64)

    np.random.seed(42)
    for iterate_count in tqdm(range(ITERATE_COUNT)):
        if return_raw_values:
            if perurbation_method=="perturb_KME_vector":
            
                unperturbed_eigenvalues_array, unperturbed_eigenvectors_array, perturbed_eigenvalues_array, \
                perturbed_eigenvectors_array, perturbed_eigenvectors_array_all_modes, start_idx, A, b = apply_perubation_theory(Hamiltonian_returned,
                                                                                            reshaped_psi, 
                                                                                            lambda_value=lambda_value, 
                                                                                            number_of_modes=number_of_modes,
                                                                                            mu=mu, 
                                                                                            sigma=sigma,
                                                                                            null_space_vector=null_space_vector, 
                                                                                            Hamiltonians_list=Hamiltonians_list, 
                                                                                            perurbation_method=perurbation_method, 
                                                                                            perturbation_amplitude=perturbation_amplitude, 
                                                                                            return_raw_values=return_raw_values,
                                                                                            PERTURBATION_APPROACH=PERTURBATION_APPROACH
                                                                                            )
                A_everything[iterate_count, :, :] = A
                b_everything[iterate_count, :] = b
            
            else:
                unperturbed_eigenvalues_array, unperturbed_eigenvectors_array, perturbed_eigenvalues_array, \
                perturbed_eigenvectors_array, perturbed_eigenvectors_array_all_modes, start_idx = apply_perubation_theory(Hamiltonian_returned,
                                                                                            reshaped_psi, 
                                                                                            lambda_value=lambda_value, 
                                                                                            number_of_modes=number_of_modes,
                                                                                            mu=mu, 
                                                                                            sigma=sigma,
                                                                                            null_space_vector=null_space_vector, 
                                                                                            Hamiltonians_list=Hamiltonians_list, 
                                                                                            perurbation_method=perurbation_method, 
                                                                                            perturbation_amplitude=perturbation_amplitude, 
                                                                                            return_raw_values=return_raw_values
                                                                                            )
            
            perturbed_eigenvectors_array_all_modes_together[iterate_count, :, :] = perturbed_eigenvectors_array_all_modes
            start_idx_all[iterate_count, :] = start_idx

        else:
            unperturbed_eigenvalues_array, unperturbed_eigenvectors_array, perturbed_eigenvalues_array, perturbed_eigenvectors_array = apply_perubation_theory(
                                                                                                                                    Hamiltonian_returned,
                                                                                                                                    reshaped_psi, 
                                                                                                                                    lambda_value=lambda_value, 
                                                                                                                                    number_of_modes=number_of_modes,
                                                                                                                                    mu=mu, 
                                                                                                                                    sigma=sigma,
                                                                                                                                    null_space_vector=null_space_vector, 
                                                                                                                                    Hamiltonians_list=Hamiltonians_list, 
                                                                                                                                    perurbation_method=perurbation_method, 
                                                                                                                                    perturbation_amplitude=perturbation_amplitude
                                                                                                                                    )

        for i in range(number_of_modes):
            per_change_out_nom = np.abs(unperturbed_eigenvectors_array[hamiltonian_selected,:,i] - perturbed_eigenvectors_array[i, :])
            per_change_out_den = np.abs(unperturbed_eigenvectors_array[hamiltonian_selected,:,i])

            per_change_out = per_change_out_nom/per_change_out_den
            MAE = np.mean(per_change_out_nom)

            per_change_out_array[iterate_count, i, :] = per_change_out
            MAE_array[iterate_count, i] = MAE

        perturbed_energy_array[iterate_count, :] = perturbed_eigenvalues_array
        unperturbed_energy_array[iterate_count, :] = unperturbed_eigenvalues_array


        ### something new - needs the average eiggen vectors - update the mean CHANGED ON 10/31/2024
        unperturbed_eigenvectors_array_all[iterate_count, :, :] = unperturbed_eigenvectors_array[hamiltonian_selected,:,:]
        perturbed_eigenvectors_array_all[iterate_count, :, :] = perturbed_eigenvectors_array
    
    if return_raw_values:
        if perurbation_method=="perturb_KME_vector":
            return per_change_out_array, MAE_array, perturbed_energy_array, unperturbed_energy_array, \
                np.mean(unperturbed_eigenvectors_array_all, axis=0), np.mean(perturbed_eigenvectors_array_all, axis=0), unperturbed_eigenvectors_array_all, \
                    perturbed_eigenvectors_array_all, perturbed_eigenvectors_array_all_modes_together, start_idx_all, A_everything, b_everything
        else:
            return per_change_out_array, MAE_array, perturbed_energy_array, unperturbed_energy_array, \
                np.mean(unperturbed_eigenvectors_array_all, axis=0), np.mean(perturbed_eigenvectors_array_all, axis=0), unperturbed_eigenvectors_array_all, \
                    perturbed_eigenvectors_array_all, perturbed_eigenvectors_array_all_modes_together, start_idx_all
    else:
        return per_change_out_array, MAE_array, perturbed_energy_array, unperturbed_energy_array, \
            np.mean(unperturbed_eigenvectors_array_all, axis=0), np.mean(perturbed_eigenvectors_array_all, axis=0)


def plot_function(per_change_out_array, MAE_avg, unperturbed_energy_array, perturbed_energy_array, psi_0, ax, loc, kernel_size, x): 
    ### Normalize the plot and merge together 
    
    color_list = ['cyan', 'green', 'blue', 'yellow', 'orange', 'purple']

    per_change_out_avg = np.mean(per_change_out_array, axis=0)
    MAE_avg = np.mean(MAE_avg, axis=0)

    unperturbed_energy_avg = np.mean(unperturbed_energy_array, axis=0)
    perturbed_energy_avg = np.mean(perturbed_energy_array, axis=0)

    mode_count = per_change_out_array.shape[1]

    mode_x = list(range(mode_count))

    # Min-Max Normalization
    min_val = per_change_out_avg.min()
    max_val = per_change_out_avg.max()
    normalized_arr_minmax = (per_change_out_avg - min_val) / (max_val - min_val)

    for i in range(mode_count):

        per_change_out = normalized_arr_minmax[i,:]
        unperturbed_energy_val = unperturbed_energy_avg[i]
        perturbed_energy_val = perturbed_energy_avg[i]
        energy_diff = np.abs(np.abs(perturbed_energy_val - unperturbed_energy_val)/unperturbed_energy_val)

        ax[0,loc].plot(x, per_change_out, color=color_list[i], label="Mode:%d"%i)
        ax[1,loc].stem(mode_x[i], unperturbed_energy_val, color_list[i], label="Mode:%d"%i)
        ax[2,loc].stem(mode_x[i], perturbed_energy_val, color_list[i], label="Mode:%d"%i)

    ax[0,loc].plot(x, psi_0, color='red', label='KME')

    # Normalized Avg change ratio across different modes 
    ax[0,loc].set_title("NACR across modes (kernel:%.4f)"%kernel_size)
    ax[0,loc].legend()

    ax[1,loc].set_title("un-P Energy across modes (kernel:%.2f)"%kernel_size)
    ax[1,loc].legend()

    ax[2,loc].set_title("P Energy across modes (kernel:%.2f)"%kernel_size)
    ax[2,loc].legend()
    
def plot_eigen_vectors(unperturbed_eigenvectors_array_forplot_final, psi_0, x):
    unperturbed_eigenvectors_array_forplot_final = np.array(unperturbed_eigenvectors_array_forplot_final)
    
    length_kernels,length_vectors, length_modes = unperturbed_eigenvectors_array_forplot_final.shape

    fig, axs = plt.subplots(nrows=length_kernels, ncols=length_modes, figsize=(18, 12))


    for i in range(length_kernels):
        for j in range(length_modes):
            axs[i,j].plot(x, unperturbed_eigenvectors_array_forplot_final[i,:,j], label="Mode:%d"%j)
            axs[i,j].plot(x, psi_0, color='red', label='KME', marker='o', linestyle='dashed')
            axs[i,j].legend()


def test_QCM_properties(QCM):
    is_symetry = np.allclose(QCM, QCM.T)
    is_hermitian = np.allclose(QCM, QCM.conj().T)
    eigenvalues_QCM = np.linalg.eigvalsh(QCM)
    is_psd = np.all(eigenvalues_QCM >= -9e-10)

    if is_symetry==True & is_hermitian==True & is_psd==True:
        return True
    else:
        print('Is the matrix symettry? ', is_symetry)
        print('Is the matrix Hermitian? ', is_hermitian)
        print('Is the matrix positive semi-definite? ', is_psd)
        print('smallest E.value? ', np.min(eigenvalues_QCM))
        return False

def generate_test_KME(sigma_kernel = 0.6, signal_name="sine", N_num_data = 256, x=None, x_data=None, frequency=50):
    # Define parameters
      #sigma_kernel = 0.6 -->  Variance of Gaussian kernel
    # M = 6  # Maximum number of modes

    if signal_name=="sine":
        # Data samples
        F0 = frequency
        Omega0 = 2 * np.pi * F0  # Signal frequency
        Fs = 6000
        Ts = 1 / Fs
        Omegas = 2 * np.pi * Fs  # Sampling frequency
        t_data = np.arange(-0.25, 0.25 + Ts, Ts)
        x_data = np.sin(Omega0 * t_data)
        x_data = (x_data - np.mean(x_data)) / np.std(x_data)
    
    elif signal_name=="delta":
        Fs = 6000
        Ts = 1 / Fs
        t_data = np.arange(-0.25, 0.25 + Ts, Ts)
        x_data = np.where(np.abs(t_data - 0) < 0.00000001, 1, 0)
        x_data = (x_data - np.mean(x_data)) / np.std(x_data)

    elif signal_name=="square":
        Fs = 6000
        duty_cycle = 0.5
        F0 = 50
        Omega0 = 2 * np.pi * F0  # Signal frequency
        Ts = 1 / Fs
        t_data = np.arange(-0.25, 0.25 + Ts, Ts)

        x_data = signal.square(2 * np.pi * F0 * t_data)

        x_data = (x_data - np.mean(x_data)) / np.std(x_data)
    elif signal_name=="input_given":
        pass


    N_data = len(x_data)
    # Kernel mean embedding (KME) (Gaussian window-based) 
    if isinstance(x, np.ndarray):
        print("KME range provided")
    else:
        print("KME range used as -6:6")
        x = points = np.linspace(-6, 6, N_num_data)

    p = np.zeros((len(x)))
    for i in range(len(x)):
        p[i] = (1 / N_data) * np.sum(np.exp(-(np.power(x[i] - x_data, 2)) / (2 * sigma_kernel ** 2)))

    psi_0 = np.sqrt(p)

    norm = np.linalg.norm(psi_0)

    # Normalize psi_0
    psi_0 = psi_0 / norm


    # Define the maximum number of bits
    max_bits = int(np.log2(N_num_data))
    # Reshape psi into a column vector
    N = max_bits  # Example: 8-qubit system
    d = 2  # Example: qubit dimension (pos spn and neg spin)

    try:
        reshaped_psi = psi_0.reshape(d**N, int(N_num_data/d**N))
    except ValueError:
        print("unable to reshape hence returning the original psi_0")
        return psi_0
        
    return reshaped_psi


def generate_test_KME_for_NN_weights(sigma_kernel = 0.6, signal_name="sine", N_num_data = 256, x=None, x_data=None, frequency=50):
    from scipy.interpolate import interp1d

    if signal_name=="sine":
        # Data samples
        F0 = frequency
        Omega0 = 2 * np.pi * F0  # Signal frequency
        Fs = 6000
        Ts = 1 / Fs
        Omegas = 2 * np.pi * Fs  # Sampling frequency
        t_data = np.arange(-0.25, 0.25 + Ts, Ts)
        x_data = np.sin(Omega0 * t_data)
        x_data = (x_data - np.mean(x_data)) / np.std(x_data)
    
    elif signal_name=="delta":
        Fs = 6000
        Ts = 1 / Fs
        t_data = np.arange(-0.25, 0.25 + Ts, Ts)
        x_data = np.where(np.abs(t_data - 0) < 0.00000001, 1, 0)
        x_data = (x_data - np.mean(x_data)) / np.std(x_data)

    elif signal_name=="square":
        Fs = 6000
        duty_cycle = 0.5
        F0 = 50
        Omega0 = 2 * np.pi * F0  # Signal frequency
        Ts = 1 / Fs
        t_data = np.arange(-0.25, 0.25 + Ts, Ts)

        x_data = signal.square(2 * np.pi * F0 * t_data)

        x_data = (x_data - np.mean(x_data)) / np.std(x_data)
    elif signal_name=="input_given":
        # x_data = (x_data - np.mean(x_data)) / np.std(x_data)
        pass


    N_data = len(x_data)
    # Kernel mean embedding (KME) (Gaussian window-based) 
    if isinstance(x, np.ndarray):
        print("KME range provided")
    else:
        print("KME range used as -6:6")
        x = points = np.linspace(-6, 6, N_num_data)

    # p = np.zeros(N_num_data)
    # for i in range(N_data):
    #     x0 = x_data[i]
    #     G = gaussian_function(x, x0, sigma_kernel)
    #     p += G
    # p /= N_data


    #### check who has the max length and then interpolate and subsample later
    p = np.zeros((len(x)))
    for i in range(len(x)):
        p[i] = (1 / N_data) * np.sum(np.exp(-(np.power(x[i] - x_data, 2)) / (2 * sigma_kernel ** 2)))   #### summed over x_data


    #### interpolate and get back to N_num_data 
    interpolation_function = interp1d(x, p, kind='linear', fill_value="extrapolate")

    ## subsampled for N_num_data
    x_re_ranged = np.linspace(min(x), max(x), N_num_data) 
    p = interpolation_function(x_re_ranged)

    psi_0 = np.sqrt(p)

    norm = np.linalg.norm(psi_0)

    # Normalize psi_0
    psi_0 = psi_0 / norm


    # Define the maximum number of bits
    max_bits = int(np.log2(N_num_data))
    # Reshape psi into a column vector
    N = max_bits  # Example: 8-qubit system
    d = 2  # Example: qubit dimension (pos spn and neg spin)
    try:
        reshaped_psi = psi_0.reshape(d**N, int(N_num_data/d**N))
    except ValueError:
        print("unable to reshape hence returning the original psi_0")
        return psi_0

    return reshaped_psi



def generate_hamiltonian_testing(reshaped_psi, Hamiltonians_list, lambda_value=0.001, return_hamiltonian=False, perturbation_amplitude = 0.001, return_raw_values=False, perurbation_method="perturb_nullspace_vector"):

    ## only using the real part of the QCM 
    # QCM = generate_QCM(Hamiltonians_list, reshaped_psi)
    QCM = generate_QCM_upper_triangular(Hamiltonians_list, reshaped_psi)  ##saves time 

    QCM = threshold_complex_elements(QCM)
    # print("QCM generated")
    
    if not test_QCM_properties(QCM):
        raise ValueError("QCM property fail")
        # print("QCM property fail")

    Hamiltonian_returned, returned_null_space_vector  = generate_hamiltonian_set_eigenvalue_base(Hamiltonians_list, QCM, Threshold=1e-10, mode=None, psi_0=reshaped_psi, return_null_space_vector=True)
    # print("Hamiltonian_returned: ", Hamiltonian_returned.shape)

    if Hamiltonian_returned.shape[0] !=1: 
        raise ValueError("Too many/less Hamiltonians")

    if not np.allclose(Hamiltonian_returned[0], np.conjugate(Hamiltonian_returned[0]).T):
        raise ValueError("Hamiltonian not hermitian")

    
    if return_raw_values:
        per_change_out_array, MAE_array, perturbed_energy_array, unperturbed_energy_array, \
            unperturbed_eigenvectors_array_forplot, perturbed_eigenvectors_array_forplot, \
                unperturbed_eigenvectors_array_raw, perturbed_eigenvectors_array_raw, \
                    perturbed_eigenvectors_array_all_modes_together, start_idx_all = test_case_function( Hamiltonian_returned, 
                                                                                                                reshaped_psi=reshaped_psi,
                                                                                                                lambda_value=lambda_value,                                                                                                                                                                        
                                                                                                                null_space_vector=returned_null_space_vector, 
                                                                                                                Hamiltonians_list=Hamiltonians_list, 
                                                                                                                perurbation_method= perurbation_method, 
                                                                                                                perturbation_amplitude = perturbation_amplitude,
                                                                                                                return_raw_values=return_raw_values
                                                                                                                )

        if return_hamiltonian:
            ## this retuurns more outputs for analysis 
            return per_change_out_array, MAE_array, perturbed_energy_array, unperturbed_energy_array, unperturbed_eigenvectors_array_forplot, \
                perturbed_eigenvectors_array_forplot, Hamiltonian_returned, returned_null_space_vector, unperturbed_eigenvectors_array_raw, \
                    perturbed_eigenvectors_array_raw, perturbed_eigenvectors_array_all_modes_together, start_idx_all
        else:
            return per_change_out_array, MAE_array, perturbed_energy_array, unperturbed_energy_array, unperturbed_eigenvectors_array_forplot, perturbed_eigenvectors_array_forplot, unperturbed_eigenvectors_array_raw, perturbed_eigenvectors_array_raw
    
    else:
        per_change_out_array, MAE_array, perturbed_energy_array, unperturbed_energy_array, \
            unperturbed_eigenvectors_array_forplot, perturbed_eigenvectors_array_forplot = test_case_function( Hamiltonian_returned, 
                                                                                                                reshaped_psi=reshaped_psi,
                                                                                                                lambda_value=lambda_value,                                                                                                                                                                        
                                                                                                                null_space_vector=returned_null_space_vector, 
                                                                                                                Hamiltonians_list=Hamiltonians_list, 
                                                                                                                perurbation_method= perurbation_method, 
                                                                                                                perturbation_amplitude = perturbation_amplitude,
                                                                                                                )

        if return_hamiltonian:
            ## this retuurns more outputs for analysis 
            return per_change_out_array, MAE_array, perturbed_energy_array, unperturbed_energy_array, unperturbed_eigenvectors_array_forplot, perturbed_eigenvectors_array_forplot, Hamiltonian_returned, returned_null_space_vector
        else:
            return per_change_out_array, MAE_array, perturbed_energy_array, unperturbed_energy_array, unperturbed_eigenvectors_array_forplot, perturbed_eigenvectors_array_forplot