**Convex Relaxation for Training Two-Layer Neural Networks**

This repository contains a PyTorch implementation and comparative analysis of training a two-layer ReLU neural network on the CIFAR-10 dataset. A hobby project I had worked on back in 2021. 
The project explores two optimization paradigms:

- **Standard Non-Convex Training:** The conventional approach using *Stochastic Gradient Descent (SGD)* and its variants to train the network end-to-end.  
- **Convex Relaxation Method:** A technique from convex optimization where the notoriously non-convex problem of training a neural network is reformulated into a convex one.

This project was developed as part of a course on convex optimization and serves as a practical investigation into advanced optimization techniques applied to machine learning.

---

## Project Overview

Training neural networks is fundamentally a non-convex optimization problem, meaning we are not guaranteed to find a global minimum.  
This project confronts this challenge by implementing a convex relaxation of the problem.

The key insight is that if we can fix the activation patterns of the ReLU neurons (i.e., whether their input is positive or negative), the problem of finding the optimal weights becomes convex.  
This project leverages this by:

1. **Generating Sign Patterns:** Pre-determining a rich set of activation patterns using random and convolutional projection methods.  
2. **Solving a Convex Formulation:** Training a custom model layer whose objective function is convex, given the fixed sign patterns.  
3. **Comparing Performance:** Empirically comparing the test accuracy, training accuracy, and objective value of the convex method against the standard, non-convex gradient descent approach.

---

## Mathematical Framework

### 1. The Standard Non-Convex Problem

The standard objective for a two-layer ReLU network can be formulated as follows, where \( W_1 \) and \( W_2 \) are the weights of the two layers:

$$
\min_{W_1, W_2} \frac{1}{2} \left\| \sum_{j=1}^{H} \max(0, x^T w_{1j}) w_{2j} - y \right\|^2 
+ \frac{\beta}{2} \|W_1\|_F^2 + \frac{\beta}{2} \sum_{j=1}^{H} \|w_{2j}\|_1^2
$$

This is implemented in `loss_func_primal`.  
This function is **non-convex** due to the ReLU activation and the multiplication of weights \( W_1 \) and \( W_2 \).

---

### 2. The Convex Relaxation

We can reformulate the problem into a convex one by introducing new variables and constraints.  
The core idea is to replace the non-convex ReLU activation with a set of linear constraints based on pre-generated **sign patterns**.

Let \( D \in \{0, 1\}^{N \times P} \) be a matrix of \( P \) sign patterns for \( N \) data points.  
The prediction \( \hat{y} \) is then modeled as:

$$
\hat{y} = \sum_{p=1}^{P} D_p \odot (X(v_p - w_p))
$$

The objective function becomes a **convex problem** over the new weight variables \( v \) and \( w \):

$$
\min_{v, w} \frac{1}{2} \|\hat{y} - y\|^2 
+ \beta \sum_{p,c} (\|v_{pc}\|_1 + \|w_{pc}\|_1) 
+ \text{Constraint Penalties}
$$

This is implemented in `loss_func_cvxproblem`, which includes terms for the prediction error, regularization, and penalties to enforce the ReLU constraints.

---

## How to Run

You can run the main script from the command line and control which models are trained using flags:
```bash
python main.py --GD 1 --CVX 1 --n_epochs 50 20


| Argument                 | Description                                                                              |
| ------------------------ | ---------------------------------------------------------------------------------------- |
| `--GD <0 or 1>`          | Set to `1` to run the non-convex (Gradient Descent) model, `0` to skip.                  |
| `--CVX <0 or 1>`         | Set to `1` to run the convex model, `0` to skip.                                         |
| `--n_epochs <int> <int>` | Pair of integers specifying epochs for the non-convex and convex models (e.g., `50 20`). |
| `--solver_cvx <str>`     | Optimizer to use for the convex problem (e.g., `adam`, `sgd`).                           |
| `--seed <int>`           | Random seed for reproducibility.                                                         |

## The script will - 
- Download the CIFAR-10 dataset
- Run the specified training jobs
- Save the results to a .pt file
- Generate plots comparing the performance of the models

# Code Structure

main.py — The main entry point to run the experiments.

models.py — Contains the PyTorch nn.Module definitions for both the standard and convex networks.

trainers.py — Includes the training and validation loops for both optimization approaches.

utils.py — Helper functions for data loading, sign pattern generation, and one-hot encoding.

plotter.py — A script to generate plots from the saved results file.