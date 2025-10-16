import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
import random
from datetime import datetime

from models import FCNetwork, CustomCvxLayer
from trainers import sgd_solver_pytorch_v2, sgd_solver_cvxproblem
from utils import PrepareData3D, generate_sign_patterns, generate_conv_sign_patterns
from plotter import plot_results

def parse_args():
    parser = argparse.ArgumentParser(description="Run Convex vs. Non-Convex NN Training")
    parser.add_argument('--GD', type=int, required=True, help="1 to run non-convex model, 0 to skip.")
    parser.add_argument('--CVX', type=int, required=True, help="1 to run convex model, 0 to skip.")
    parser.add_argument('--n_epochs', nargs=2, type=int, required=True, help="Epochs for GD and CVX models (e.g., 50 20).")
    parser.add_argument('--solver_cvx', type=str, default="adam", help="Optimizer for the convex model (adam or sgd).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    return parser.parse_args()

def main():
    ARGS = parse_args()
    
    # Set seeds for reproducibility
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 1. Load CIFAR-10 Data ---
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    
    # Extract all data for sign pattern generation
    full_train_loader = DataLoader(train_dataset, batch_size=50000, shuffle=False)
    A_patches, y_full = next(iter(full_train_loader))
    A_flat = A_patches.view(A_patches.shape[0], -1).numpy()
    y_full = y_full.numpy()
    n, d = A_flat.shape
    
    # --- 2. Define Problem Parameters ---
    P = 4096  # Number of neurons / sign patterns
    beta = 1e-3 # Regularization parameter
    num_epochs_gd, num_epochs_cvx = ARGS.n_epochs
    batch_size = 1000
    
    # --- 3. Run Non-Convex (GD) Experiment ---
    results_to_save = {}
    if ARGS.GD:
        print("\n--- Starting Non-Convex (SGD) Training ---")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        # Experiment with different learning rates
        learning_rates = [1e-2, 5e-3, 1e-3]
        for i, lr in enumerate(learning_rates):
            print(f"\nTraining SGD with learning rate: {lr}")
            model_gd = FCNetwork(H=P, input_dim=d).to(device)
            results = sgd_solver_pytorch_v2(train_loader, test_loader, num_epochs_gd, model_gd, beta, lr, 
                                            batch_size, "sgd", 0, [10, 4], device, train_len=n)
            results_to_save[f'results_noncvx_sgd{i+1}'] = results
        results_to_save['num_epochs1'] = num_epochs_gd

    # --- 4. Run Convex Experiment ---
    if ARGS.CVX:
        print("\n--- Starting Convex Formulation Training ---")
        rho = 1e-2 # Penalty for constraint violations
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        # a) With random sign patterns
        print("\nGenerating random sign patterns...")
        _, _, u_vectors_list = generate_sign_patterns(A_flat, P, verbose=True)
        u_vectors = np.array(u_vectors_list).T # d x P
        sign_patterns = (A_flat @ u_vectors >= 0) # N x P
        
        train_data_cvx = PrepareData3D(A_flat, y_full, sign_patterns)
        train_loader_cvx = DataLoader(train_data_cvx, batch_size=batch_size, shuffle=True)
        
        learning_rates_cvx = [1e-6, 5e-7]
        for i, lr in enumerate(learning_rates_cvx):
            print(f"\nTraining Convex (Random Patterns) with learning rate: {lr}")
            model_cvx = CustomCvxLayer(d, P).to(device)
            results = sgd_solver_cvxproblem(train_loader_cvx, test_loader, num_epochs_cvx, model_cvx, beta, lr, rho, u_vectors.T, ARGS.solver_cvx, device, n=n)
            results_to_save[f'results_cvx{i+1}'] = results

        # b) With convolutional sign patterns
        print("\nGenerating convolutional sign patterns...")
        _, _, u_vectors_list_conv = generate_conv_sign_patterns(A_patches.numpy(), P, verbose=True)
        u_vectors_conv = np.array(u_vectors_list_conv).T
        sign_patterns_conv = (A_flat @ u_vectors_conv >= 0)

        train_data_cvx_conv = PrepareData3D(A_flat, y_full, sign_patterns_conv)
        train_loader_cvx_conv = DataLoader(train_data_cvx_conv, batch_size=batch_size, shuffle=True)
        
        for i, lr in enumerate(learning_rates_cvx):
            print(f"\nTraining Convex (Conv Patterns) with learning rate: {lr}")
            model_cvx_conv = CustomCvxLayer(d, P).to(device)
            results = sgd_solver_cvxproblem(train_loader_cvx_conv, test_loader, num_epochs_cvx, model_cvx_conv, beta, lr, rho, u_vectors_conv.T, ARGS.solver_cvx, device, n=n)
            results_to_save[f'results_cvx_conv{i+1}'] = results
        results_to_save['num_epochs2'] = num_epochs_cvx

    # --- 5. Save and Plot Results ---
    if results_to_save:
        now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        filename = f'results_{now}.pt'
        torch.save(results_to_save, filename)
        print(f"\nResults saved to {filename}")
        
        # Generate plots
        print("Generating plots...")
        plot_results(filename)
        print("Plots saved as .png files.")

if __name__ == '__main__':
    main()
