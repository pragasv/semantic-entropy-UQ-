
import numpy as np
from matplotlib import pyplot as plt

def fetch_only_psi_perturbation(unperturbed_eigenvectors_array_forplot, perturbed_eigenvectors_array_forplot, reshaped_psi, lambda_val):
    squared_error = (np.abs(unperturbed_eigenvectors_array_forplot) - reshaped_psi)**2

    mse = np.mean(squared_error, axis=0)
    loc_min = np.argmin(mse)

    plt.figure(figsize=(18,5))
    plt.subplot(1,5,1)
    plt.plot(unperturbed_eigenvectors_array_forplot[:,loc_min].real, label='psi^0 r')
    plt.plot(unperturbed_eigenvectors_array_forplot[:,loc_min].imag, label='psi^0 i')
    plt.plot(perturbed_eigenvectors_array_forplot[loc_min, :].real, label='psi^1 real')
    plt.plot(perturbed_eigenvectors_array_forplot[loc_min, :].imag, label='psi^1 imag')
    plt.plot(reshaped_psi, color='red', label='KME', marker='o', linestyle='dashed')
    plt.legend(loc='upper right')

    plt.subplot(1,5,4)
    NARC = np.abs((reshaped_psi[:,0] -perturbed_eigenvectors_array_forplot[loc_min, :])/reshaped_psi[:,0]) * lambda_val

    ## abs(psi-lambdapsi'/psi)
    metric2 = np.abs((reshaped_psi[:,0] - perturbed_eigenvectors_array_forplot[loc_min, :]* lambda_val)/reshaped_psi[:,0]) 

    # plt.plot(unperturbed_eigenvectors_array_forplot[:,loc_min].real, label='psi^0 r')
    # plt.plot(unperturbed_eigenvectors_array_forplot[:,loc_min].imag, label='psi^0 i')
    plt.plot(NARC, label='MAPE * lambda')
    # plt.plot(metric2, label='metric2')
    plt.plot(reshaped_psi, color='red', label='KME', marker='o', linestyle='dashed')
    plt.ylim(0,10)
    plt.legend(loc='upper right')

    plt.subplot(1,5,5)
    plt.plot(metric2, label='abs(psi- lambda*psi /psi')
    plt.plot(reshaped_psi, color='red', label='KME', marker='o', linestyle='dashed')
    plt.ylim(0,10)
    plt.legend(loc='upper right')


    plt.subplot(1,5,2)
    plt.plot(perturbed_eigenvectors_array_forplot[loc_min, :].real * lambda_val, label='lambda* psi^1 real')
    plt.plot(perturbed_eigenvectors_array_forplot[loc_min, :].imag * lambda_val, label='lambda* psi^1 imag')
    plt.plot(reshaped_psi, color='red', label='KME', marker='o', linestyle='dashed')
    plt.legend(loc='upper right')

    plt.subplot(1,5,3)
    plt.plot(np.abs(perturbed_eigenvectors_array_forplot[loc_min, :]) * lambda_val, label='lambda* psi^1 abs')
    plt.plot(reshaped_psi, color='red', label='KME', marker='o', linestyle='dashed')
    plt.legend(loc='upper right')


def plot_eigen_vectors(unperturbed_eigenvectors_array_forplot_final, psi_0_list, method=None):
    unperturbed_eigenvectors_array_forplot_final = np.array(unperturbed_eigenvectors_array_forplot_final)
    
    length_kernels,length_vectors, length_modes = unperturbed_eigenvectors_array_forplot_final.shape

    if method==None:

        fig, axs = plt.subplots(nrows=length_kernels, ncols=length_modes, figsize=(18, 6))


        for i in range(length_kernels):
            psi_0 = psi_0_list[i]
            for j in range(length_modes):
                try: 
                    axs[i,j].plot(x, np.abs(unperturbed_eigenvectors_array_forplot_final[i,:,j]), label="Mode:%d"%j)
                    axs[i,j].plot(x, psi_0, color='red', label='KME', marker='o', linestyle='dashed')
                    axs[i,j].legend()
                    plt.title("abs eigen vector")
                except IndexError:
                    ### this happens when only one kernel size is tested
                    # axs[j].plot(np.abs(unperturbed_eigenvectors_array_forplot_final[i,:,j]), label="Mode:%d"%j)
                    axs[j].plot(unperturbed_eigenvectors_array_forplot_final[i,:,j].real, color='green', label="real")
                    axs[j].plot(unperturbed_eigenvectors_array_forplot_final[i,:,j].imag, color='black', label="imag")
                    axs[j].plot(psi_0, color='red', label='KME', marker='o', linestyle='dashed', markersize=2)
                    axs[j].legend()
                    axs[j].set_ylim(-0.3,0.3)
                    plt.title("abs eigen vector")
    
    elif method=="smallest_MSE":
        # plot only the smallest MSE ev
        # fig, axs = plt.subplots(length_kernels)

        for i in range(length_kernels):
            plt.figure(figsize=(18, 6))

            psi_0 = psi_0_list[i]

            min_idx = 0
            min_mse = np.inf

            for j in range(length_modes):
                mse = np.mean((unperturbed_eigenvectors_array_forplot_final[i,:,j] - psi_0)**2)

                if min_mse > mse:
                    min_mse = mse 
                    min_mse_ev = unperturbed_eigenvectors_array_forplot_final[i,:,j]
                    min_idx = j
            
            plt.plot(np.abs(min_mse_ev), label="Mode:%d"%min_idx)
            plt.plot(min_mse_ev.real, color='green', label="real")
            plt.plot(min_mse_ev.imag, color='black', label="imag")
            plt.plot(np.abs(psi_0), color='red', label='KME', marker='o', linestyle='dashed')
            plt.legend()
            plt.title("abs mini mse eigen vector. MSE:%.3f "%min_mse)

def plot_rishab_ratio_array(ratio_main, psi_0, method="plot_seperate", normalize=True):
    L,M = ratio_main.shape

    plt.figure(figsize=(15,5))
    for i in range(M):
        if method=="plot_seperate":

            plt.subplot(1,M,i+1)
            data = ratio_main[:,i]
            # Min-Max Normalization
            if normalize:
                data = (data - data.min()) / (data.max() - data.min())
            plt.plot(data, label="M:%d"%i)
            plt.plot(psi_0, label="psi")

        elif method=="plot_all":
            ratio_main_normalized = (ratio_main - ratio_main.min()) / (ratio_main.max() - ratio_main.min())
            plt.plot(ratio_main_normalized[:,i], label="M:%d"%i)
            
        
        plt.legend()
    plt.plot(psi_0, label="psi")
    plt.show()


def plot_standard_deviation(std_dev_arr, shift_value, window_size, y_data):
    plt.figure()
    plt.subplot(2,1,1)
    stop = window_size + shift_value * len(std_dev_arr)
    x_axis = np.arange(window_size, stop, shift_value)
    plt.plot(x_axis, std_dev_arr, marker=".",  markersize=10, label="std deviation of eigen vectors")
    plt.subplot(2,1,2)
    plt.plot(y_data, label="original data")