import numpy as np
from utils_qcm import gaussian_function

def generate_mean_jumps(change_point_frequency = 100):
    np.random.seed(42)
    # Initialize parameters
    n_samples = 5000
    y = np.zeros(n_samples)
    std_dev = 1.5  # Initial standard deviation

    # Generate the noise array
    noise = np.zeros(n_samples)
    psi_N = 0  # Initial value of psi_N for N = 1

    change_point_counts = int(n_samples/change_point_frequency)

    y[0]=0
    y[1]=0

    mean_array = np.zeros(change_point_counts)
    # Calculate noise values for each segment
    for N in range(change_point_counts):
        current_segment_start=change_point_frequency*(N)
        if N == 0:
            # First segment, psi_N = 0
            noise[current_segment_start:current_segment_start+change_point_frequency] = np.random.normal(0, std_dev, size=change_point_frequency)
            mean_array[N]=0
        else:
            mean_array[N]= mean_array[N-1] + N/16
            noise[current_segment_start:current_segment_start+change_point_frequency] = np.random.normal(mean_array[N], std_dev, size=change_point_frequency)
    

    # Generate the AR process
    for t in range(2, n_samples):
        y[t] = 0.6 * y[t-1] - 0.5 * y[t-2] + noise[t]
    
    return y

def generate_mean_jumps_approach2(change_point_frequency = 100):
    np.random.seed(42)
    # Initialize parameters
    n_samples = 5000
    y = np.zeros(n_samples)
    std_dev = 1.5  # Initial standard deviation

    # Generate the noise array
    noise = np.zeros(n_samples)
    psi_N = 0  # Initial value of psi_N for N = 1
    change_point_counts = int(n_samples/change_point_frequency)

    y[0]=0
    y[1]=0

    mean_array = np.arange(0, change_point_counts)
    # Calculate noise values for each segment
    for N in range(change_point_counts):
        current_segment_start=change_point_frequency*(N)

        y[current_segment_start:current_segment_start+change_point_frequency] = mean_array[N]*np.ones(change_point_frequency)
    
    return y


def generate_std_jumps(change_point_frequency=200):
    np.random.seed(42)
    # Initialize parameters
    n_samples = 5000
    y = np.zeros(n_samples)

    # Generate the noise array
    noise = np.zeros(n_samples)
    change_point_counts = int(n_samples/change_point_frequency)

    y[0]=0
    y[1]=0

    std_dev_array = np.zeros(change_point_counts)
    # Calculate noise values for each segment
    for N in range(change_point_counts):
        current_segment_start=change_point_frequency*(N)
        if N == 0:
            # First segment, psi_N = 0
            std_dev_array[N]=1
            noise[current_segment_start:current_segment_start+change_point_frequency] = np.random.normal(0, std_dev_array[N], size=change_point_frequency)
            
        else:
            std_dev_array[N]= np.log(np.e + N/4)
            noise[current_segment_start:current_segment_start+change_point_frequency] = np.random.normal(0, std_dev_array[N], size=change_point_frequency)


    # Generate the AR process
    for t in range(2, n_samples):
        y[t] = 0.6 * y[t-1] - 0.5 * y[t-2] + noise[t]
    
    return y

def generate_rishab_KME(x_data, x, sigma_kernel=0.6): 
    ### x is the range of the KME
    x_data = (x_data - np.mean(x_data)) / np.std(x_data)

    N_data = len(x_data)
    
    p = np.zeros(N_data)
    for i in range(N_data):
        x0 = x_data[i]
        G = gaussian_function(x, x0, sigma_kernel)
        p += G
    p /= N_data

    psi_0 = np.sqrt(p)

    norm = np.linalg.norm(psi_0)

    # Normalize psi_0
    psi_0 = psi_0 / norm

    return psi_0
