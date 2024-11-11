import torch
import torch.nn.functional as F


def Triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = F.pairwise_distance(anchor, positive, p=2)
    distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
    loss = F.relu(distance_positive - distance_negative + margin)
    return loss


def loss_cal(z_i, z_j, t=0.2):
    batch_size, _ = z_i.size()
    z_i_abs = z_i.norm(dim=1)
    z_j_abs = z_j.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', z_i, z_j) / torch.einsum('i,j->ij', z_i_abs, z_j_abs)
    sim_matrix = torch.exp(sim_matrix / t)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss

def focal_loss(pred, target, gamma=2, alpha=0.25):
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()

def info_nce_loss(z1, z2, temperature=0.07):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    sim_i_j = torch.diag(sim, N)
    sim_j_i = torch.diag(sim, -N)
    positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(N, 2)
    mask = torch.eye(2*N, dtype=torch.bool, device=z1.device)
    negative_samples = sim[~mask].reshape(2*N, -1)

    labels = torch.zeros(N, device=z1.device).long()
    logits = torch.cat([positive_samples, negative_samples[:N, :N-1]], dim=1)
    return F.cross_entropy(logits, labels)


def loss_func(adj, A_hat, attrs, X_hat):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)


    return structure_cost, attribute_cost
