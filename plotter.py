import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_results(results_file):
    """ Loads a results file and generates comparison plots. """
    
    data = torch.load(results_file)
    
    # Unpack data based on what was run
    if 'results_noncvx_sgd1' in data:
        num_epochs1 = data['num_epochs1']
        results_noncvx_sgd1 = data['results_noncvx_sgd1']
        results_noncvx_sgd2 = data['results_noncvx_sgd2']
        results_noncvx_sgd3 = data['results_noncvx_sgd3']
    
    if 'results_cvx1' in data:
        num_epochs2 = data['num_epochs2']
        results_cvx1 = data['results_cvx1']
        results_cvx2 = data['results_cvx2']
        results_cvx_conv1 = data['results_cvx_conv1']
        results_cvx_conv2 = data['results_cvx_conv2']

    fsize = 20
    fsize_legend = 12
    plt.rcParams.update({'font.size': fsize})
    
    # ====== Plot 1: Test Accuracy vs. Time ======
    plt.figure(figsize=(12, 8))
    plt.xlabel('Time (s)', fontsize=fsize)
    plt.ylabel('Test Accuracy', fontsize=fsize)
    plt.title('Test Accuracy vs. Time', fontsize=fsize)
    plt.grid(True, which="both")
    
    if 'results_noncvx_sgd1' in data:
        times = results_noncvx_sgd1[4] - results_noncvx_sgd1[4][0]
        plt.plot(times[::50], results_noncvx_sgd1[3], '--', label=r"SGD ($\mu=1e-2$)")
        times = results_noncvx_sgd2[4] - results_noncvx_sgd2[4][0]
        plt.plot(times[::50], results_noncvx_sgd2[3], '--', label=r"SGD ($\mu=5e-3$)")
        times = results_noncvx_sgd3[4] - results_noncvx_sgd3[4][0]
        plt.plot(times[::50], results_noncvx_sgd3[3], '--', label=r"SGD ($\mu=1e-3$)")

    if 'results_cvx1' in data:
        times = results_cvx1[4] - results_cvx1[4][0]
        plt.plot(times[::60], results_cvx1[3], 'o-', label=r"Convex-Random ($\mu=1e-6$)")
        times = results_cvx2[4] - results_cvx2[4][0]
        plt.plot(times[::60], results_cvx2[3], 'o-', label=r"Convex-Random ($\mu=5e-7$)")
        times = results_cvx_conv1[4] - results_cvx_conv1[4][0]
        plt.plot(times[::60], results_cvx_conv1[3], 's-', label=r"Convex-Conv ($\mu=1e-6$)")
        times = results_cvx_conv2[4] - results_cvx_conv2[4][0]
        plt.plot(times[::60], results_cvx_conv2[3], 's-', label=r"Convex-Conv ($\mu=5e-7$)")
        
    plt.legend(prop={'size': fsize_legend})
    plt.ylim(0.3, 0.6)
    plt.savefig('test_accuracy_comparison.png', bbox_inches='tight')
    plt.show()

    # ====== Plot 2: Training Loss vs. Time ======
    plt.figure(figsize=(12, 8))
    plt.xlabel('Time (s)', fontsize=fsize)
    plt.ylabel('Training Objective Value', fontsize=fsize)
    plt.title('Training Objective vs. Time', fontsize=fsize)
    plt.yscale('log')
    plt.grid(True, which="both")

    if 'results_noncvx_sgd1' in data:
        times = results_noncvx_sgd1[4] - results_noncvx_sgd1[4][0]
        plt.plot(times[1:], results_noncvx_sgd1[0], '--', label=r"SGD ($\mu=1e-2$)")
    
    if 'results_cvx1' in data:
        times = results_cvx1[4] - results_cvx1[4][0]
        plt.plot(times[1:], results_cvx1[5], 'o-', label=r"Convex-Random ($\mu=1e-6$)")
        times = results_cvx_conv1[4] - results_cvx_conv1[4][0]
        plt.plot(times[1:], results_cvx_conv1[5], 's-', label=r"Convex-Conv ($\mu=1e-6$)")

    plt.legend(prop={'size': fsize_legend})
    plt.savefig('training_loss_comparison.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # Example usage:
    # After running main.py, a results file will be created.
    # Pass that file path to this script.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', type=str, required=True, help="Path to the .pt results file.")
    args = parser.parse_args()
    
    plot_results(args.results_file)
