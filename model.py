import torch
import torch.nn as nn
import util

class Model(nn.Module):

    def __init__(self, vocab_size, embedding_dim=10):
        super(Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.data.uniform_(-0.001, 0.001)

    def forward(self, u, v, negatives):
        u_emb = self.embeddings(u)
        v_emb = self.embeddings(v)
        negative_embeddings = self.embeddings(negatives)
        numer = torch.exp(-1 * util.hyperbolic_distance(u_emb, v_emb))
        denom = torch.exp(-1 * util.hyperbolic_distance(u_emb, negative_embeddings))
        denom = torch.sum(denom)
        # print('Numer: ', torch.log(numer))
        # print('Denom: ', torch.log(denom))
        # res = torch.log(numer) - torch.log(denom)
        res = numer / denom
        return res

    def fit(self, graph, alpha=0.10, iter=5, negative_samples=10, c=10):
        loss = 0
        for i in range(iter):
            loss = 0
            self.zero_grad()
            for u, v, negs in graph.get_examples(negative_samples):
                loss += self.forward(u, v, negs)
            print('Loss at iteration ', i, ' is ', loss.data[0])
            loss.backward()
            for theta in self.parameters():
                beta = -alpha
                if i < 10:
                    beta /= c
                tensor_vals = torch.pow(util.metric_tensor(theta), -1)
                multed = tensor_vals.mul(beta).repeat(1, self.embedding_dim)
                grads = theta.grad
                scaled_gradient = multed * grads
                theta.data.add_(scaled_gradient.data)
                theta.data = util.proj(theta.data)









