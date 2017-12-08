import torch

def load_embeddings(num, dim, scale=0.1):
    embeddings = torch.Tensor(num, dim)
    embeddings.normal_(0, scale)

    # fill in embeddings

    # zero NULL token
    return embeddings
