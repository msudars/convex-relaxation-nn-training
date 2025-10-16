import torch
import torch.nn as nn

class FCNetwork(nn.Module):
    """ Standard two-layer fully-connected network with ReLU activation. """
    def __init__(self, H, num_classes=10, input_dim=3072):
        self.num_classes = num_classes
        super(FCNetwork, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, H, bias=False), nn.ReLU())
        self.layer2 = nn.Linear(H, num_classes, bias=False)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        out = self.layer2(self.layer1(x))
        return out

class CustomCvxLayer(torch.nn.Module):
    """
    Custom PyTorch layer for the convex formulation of a two-layer network.
    The weights v and w are learned to solve the convex optimization problem.
    """
    def __init__(self, d, num_neurons, num_classes=10):
        super(CustomCvxLayer, self).__init__()
        # Parameters for the convex relaxation: v and w
        # P x d x C, where P is num_neurons, d is input_dim, C is num_classes
        self.v = torch.nn.Parameter(data=torch.zeros(num_neurons, d, num_classes), requires_grad=True)
        self.w = torch.nn.Parameter(data=torch.zeros(num_neurons, d, num_classes), requires_grad=True)

    def forward(self, x, sign_patterns):
        # sign_patterns: N x P
        # x: N x d
        sign_patterns = sign_patterns.unsqueeze(2) # N x P x 1
        x = x.view(x.shape[0], -1)  # N x d

        # Matmul: (N x d) x (P x d x C) -> needs permutation
        # Let's compute (v-w) first
        v_minus_w = self.v - self.w # P x d x C
        
        # We want to compute X * (v-w) for each neuron, which is tricky with batching.
        # A more direct way is to permute and then do batch matrix multiplication.
        x_perm = x.unsqueeze(0).permute(1,0,2) # N x 1 x d
        v_minus_w_perm = v_minus_w.permute(0,2,1) # P x C x d
        
        # This is still not quite right. Let's use einsum for clarity.
        # n: batch_size, d: input_dim, p: num_neurons, c: num_classes
        Xv_w = torch.einsum('nd,pdc->npc', x, v_minus_w) # N x P x C
        
        # Element-wise product with sign patterns
        DXv_w = torch.mul(sign_patterns, Xv_w) # N x P x C
        
        # Sum over the neuron dimension (P)
        y_pred = torch.sum(DXv_w, dim=1, keepdim=False)  # N x C
        return y_pred
