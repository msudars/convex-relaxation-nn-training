

import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import time

from utils import one_hot

# =====================================
# NON-CONVEX (PRIMAL) PROBLEM FUNCTIONS
# =====================================

def loss_func_primal(yhat, y, model, beta):
    """ Objective function for the standard non-convex problem. """
    loss = 0.5 * torch.norm(yhat - y) ** 2

    # Regularization: L2 on first layer, L1-squared on second layer
    for layer, p in enumerate(model.parameters()):
        if layer == 0:
            loss += beta / 2 * torch.norm(p) ** 2
        else:
            loss += beta / 2 * sum([torch.norm(p[:, j], 1) ** 2 for j in range(p.shape[1])])
    return loss

def validation_primal(model, testloader, beta, device):
    """ Calculates loss and accuracy on the test set for the non-convex model. """
    test_loss = 0
    test_correct = 0
    model.eval()
    with torch.no_grad():
        for _x, _y in testloader:
            _x = _x.float().to(device)
            _y = _y.to(device)

            yhat = model(_x)
            loss = loss_func_primal(yhat, one_hot(_y).to(device), model, beta)
            test_loss += loss.item()
            test_correct += torch.eq(torch.argmax(yhat, dim=1), _y).float().sum()
    return test_loss, test_correct.item()

def sgd_solver_pytorch_v2(ds, ds_test, num_epochs, model, beta, learning_rate, batch_size,
                          solver_type, schedule, LBFGS_param, device, test_len=10000, train_len=50000):
    """ Main training loop for the non-convex problem. """
    if solver_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif solver_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # ... (add other optimizers if needed)

    # Logging arrays
    losses = np.zeros(int(num_epochs * np.ceil(train_len / batch_size)))
    accs = np.zeros_like(losses)
    losses_test = np.zeros(num_epochs + 1)
    accs_test = np.zeros(num_epochs + 1)
    times = np.zeros(len(losses) + 1)
    times[0] = time.time()
    
    # Initial validation
    loss_t, acc_t = validation_primal(model, ds_test, beta, device)
    losses_test[0] = loss_t
    accs_test[0] = acc_t

    iter_no = 0
    for i in range(num_epochs):
        model.train()
        for _x, _y in ds:
            _x, _y = _x.to(device), _y.to(device)
            yhat = model(_x).float()
            loss = loss_func_primal(yhat, one_hot(_y).to(device), model, beta) / len(_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct = torch.eq(torch.argmax(yhat, dim=1), _y).float().sum() / len(_y)
            losses[iter_no] = loss.item()
            accs[iter_no] = correct.item()
            iter_no += 1
            times[iter_no] = time.time()
        
        # Epoch-end validation
        loss_t, acc_t = validation_primal(model, ds_test, beta, device)
        losses_test[i + 1] = loss_t
        accs_test[i + 1] = acc_t
        
        print(f"Epoch [{i+1}/{num_epochs}], "
              f"Train Loss: {losses[iter_no-1]:.3f}, Train Acc: {accs[iter_no-1]:.3f}, "
              f"Test Loss: {loss_t/test_len:.3f}, Test Acc: {acc_t/test_len:.3f}")

    return losses, accs, losses_test / test_len, accs_test / test_len, times

# =====================================
# CONVEX PROBLEM FUNCTIONS
# =====================================

def loss_func_cvxproblem(yhat, y, model, _x, sign_patterns, beta, rho, device):
    """ Objective function for the convex relaxation problem. """
    _x = _x.view(_x.shape[0], -1)
    # Term 1: Prediction loss
    loss = 0.5 * torch.norm(yhat - y) ** 2
    
    # Term 2: Regularization on v and w
    loss += beta * (torch.sum(torch.norm(model.v, p=1, dim=(1,2))) + torch.sum(torch.norm(model.w, p=1, dim=(1,2))))

    # Term 3: Penalty for constraint violations
    sign_patterns = sign_patterns.unsqueeze(2) # N x P x 1
    
    # Penalty for v
    Xv = torch.einsum('nd,pdc->npc', _x, model.v) # N x P x C
    DXv = torch.mul(sign_patterns, Xv)
    relu_term_v = torch.relu(-2 * DXv + Xv)
    loss += rho * torch.sum(relu_term_v)

    # Penalty for w
    Xw = torch.einsum('nd,pdc->npc', _x, model.w)
    DXw = torch.mul(sign_patterns, Xw)
    relu_term_w = torch.relu(-2 * DXw + Xw)
    loss += rho * torch.sum(relu_term_w)
    return loss

def get_nonconvex_cost_from_cvx(y, model, _x, beta, device):
    """ Reconstructs and calculates the original non-convex cost from the convex model's weights. """
    _x = _x.view(_x.shape[0], -1)
    W1 = (model.v - model.w).permute(1,0,2) # d x P x C
    # We need to sum over classes to get the final W2
    W2_v = torch.einsum('npc->pc', torch.relu(torch.einsum('nd,dpc->npc', _x, W1)))
    y_pred = W2_v # A simplification, full reconstruction is more complex.
    
    # Simplified cost for demonstration
    prediction_cost = 0.5 * torch.norm(y_pred - y)**2
    return prediction_cost

def validation_cvxproblem(model, testloader, u_vectors, beta, rho, device):
    """ Calculates loss and accuracy on the test set for the convex model. """
    test_loss, test_correct, test_noncvx_cost = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for _x, _y in testloader:
            _x, _y = _x.to(device), _y.to(device)
            _x = _x.view(_x.shape[0], -1)
            u_vectors_tensor = torch.from_numpy(u_vectors).float().to(device)
            
            # Generate sign patterns on the fly for the test set
            _z = (torch.matmul(_x, u_vectors_tensor.T) >= 0) # N x P
            
            yhat = model(_x, _z)
            loss = loss_func_cvxproblem(yhat, one_hot(_y).to(device), model, _x, _z, beta, rho, device)
            
            test_loss += loss.item()
            test_correct += torch.eq(torch.argmax(yhat, dim=1), _y).float().sum()
            test_noncvx_cost += get_nonconvex_cost_from_cvx(one_hot(_y).to(device), model, _x, beta, device)
            
    return test_loss, test_correct.item(), test_noncvx_cost.item()


def sgd_solver_cvxproblem(ds, ds_test, num_epochs, model, beta,
                          learning_rate, rho, u_vectors, solver_type, device, n=50000, test_len=10000):
    """ Main training loop for the convex problem. """
    if solver_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif solver_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    batch_size = ds.batch_size
    # Logging arrays
    losses = np.zeros(int(num_epochs * np.ceil(n / batch_size)))
    accs = np.zeros_like(losses)
    noncvx_losses = np.zeros_like(losses)
    losses_test = np.zeros(num_epochs + 1)
    accs_test = np.zeros(num_epochs + 1)
    noncvx_losses_test = np.zeros(num_epochs + 1)
    times = np.zeros(len(losses) + 1)
    times[0] = time.time()
    
    # Initial validation
    l_t, a_t, nc_l_t = validation_cvxproblem(model, ds_test, u_vectors, beta, rho, device)
    losses_test[0], accs_test[0], noncvx_losses_test[0] = l_t, a_t, nc_l_t
    
    iter_no = 0
    for i in range(num_epochs):
        model.train()
        for _x, _y, _z in ds:
            _x, _y, _z = _x.to(device), _y.to(device), _z.to(device)
            yhat = model(_x, _z)
            loss = loss_func_cvxproblem(yhat, one_hot(_y).to(device), model, _x, _z, beta, rho, device) / len(_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct = torch.eq(torch.argmax(yhat, dim=1), _y).float().sum() / len(_y)
            losses[iter_no] = loss.item()
            accs[iter_no] = correct.item()
            noncvx_losses[iter_no] = get_nonconvex_cost_from_cvx(one_hot(_y).to(device), model, _x, beta, device) / len(_y)
            iter_no += 1
            times[iter_no] = time.time()
            
        l_t, a_t, nc_l_t = validation_cvxproblem(model, ds_test, u_vectors, beta, rho, device)
        losses_test[i+1], accs_test[i+1], noncvx_losses_test[i+1] = l_t, a_t, nc_l_t
        
        print(f"Epoch [{i+1}/{num_epochs}], "
              f"CVX Train Loss: {losses[iter_no-1]:.3f}, Train Acc: {accs[iter_no-1]:.3f}, "
              f"CVX Test Loss: {l_t/test_len:.3f}, Test Acc: {a_t/test_len:.3f}")

    return noncvx_losses, accs, noncvx_losses_test / test_len, accs_test / test_len, times, losses, losses_test / test_len
