
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)

def to_var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return x.clone().detach().requires_grad_(requires_grad)

class SingleLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SingleLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class SLFNet(nn.Module):
  def __init__(self, in_dim, n_hidden):
      super(SLFNet, self).__init__()
      self.linear = SingleLinear(in_dim, n_hidden, bias=True)
      self.dropout = nn.Dropout(0)
      self.sig = nn.Sigmoid()

  def initialize(self):
      for m in self.modules():
          if isinstance(m, nn.Linear):
              nn.init.kaiming_normal_(m.weight.data)

  def forward_L(self, x):
      x = self.linear(x)
      x = self.dropout(x)
      x = self.sig(x)
      return x

  def set_masks(self, masks):
      self.linear.set_mask(masks[0])


  def train(self, features, targets,L=0.1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    H_train = self.forward_L(features)
    self.P = torch.pinverse(torch.matmul(H_train.T, H_train)+torch.eye(H_train.shape[1]).to(device) / L)
    # SVD
    # self.P = self.svd_PIV(torch.matmul(H_train.T, H_train)+torch.eye(H_train.shape[1]).to(device) / L)
    self.pinv = torch.matmul(self.P, H_train.T)
    self.beta = torch.matmul(self.pinv, targets)
    return self.beta


  def upgrade_beta(self, valid, target, L=0.1, device='cpu'):
      batch_size = valid.shape[0]
      H = self.forward_L(valid)
      I = (torch.eye(batch_size)/L).to(device)
      temp = torch.linalg.pinv(I + torch.matmul(torch.matmul(H, self.P), H.T))
      self.P -= torch.matmul(torch.matmul(torch.matmul(self.P, H.T), temp), torch.matmul(H, self.P))
      pHT = torch.matmul(self.P, H.T)
      Hbeta = torch.matmul(H, self.beta)
      self.beta += torch.matmul(pHT, target - Hbeta)
      return self.beta


  def predict(self, features, beta):
    H = self.forward_L(features)
    prediction = torch.matmul(H, beta)
    return prediction

  def svd_PIV(self, x):
      u, sigma, v = torch.svd(x, some=True)
      s_plus = torch.zeros_like(x).T
      for i in range(len(sigma)):
          s_plus[i, i] = 1/(sigma[i] + 1e-10)
      piv = torch.mm(v.t(), torch.mm(s_plus, u.t()))
      return piv


class SLFNetXAI(nn.Module):
      def __init__(self, in_dim, n_hidden):
          super(SLFNetXAI, self).__init__()
          self.linear = SingleLinear(in_dim, n_hidden, bias=True)
          self.dropout = nn.Dropout(0)
          self.sig = nn.Sigmoid()

      def initialize(self):
          for m in self.modules():
              if isinstance(m, nn.Linear):
                  nn.init.kaiming_normal_(m.weight.data)

      def forward(self, x):
          H = self.forward_L(x)
          prediction = torch.matmul(H, self.beta).view(-1, 1)
          return prediction

      def forward_L(self, x):
          x = self.linear(x)
          x = self.dropout(x)
          x = self.sig(x)
          return x

      def set_masks(self, masks):
          self.linear.set_mask(masks[0])

      def train(self, features, targets, L=0.1):
          device = 'cuda' if torch.cuda.is_available() else 'cpu'
          H_train = self.forward_L(features)
          self.P = torch.pinverse(torch.matmul(H_train.T, H_train) + torch.eye(H_train.shape[1]).to(device) / L)
          # SVD
          # self.P = self.svd_PIV(torch.matmul(H_train.T, H_train)+torch.eye(H_train.shape[1]).to(device) / L)
          self.pinv = torch.matmul(self.P, H_train.T)
          self.beta = torch.matmul(self.pinv, targets)
          return self.beta

      def upgrade_beta(self, valid, target, L=0.1, device='cpu'):
          batch_size = valid.shape[0]
          H = self.forward_L(valid)
          I = (torch.eye(batch_size) / L).to(device)
          temp = torch.linalg.pinv(I + torch.matmul(torch.matmul(H, self.P), H.T))
          self.P -= torch.matmul(torch.matmul(torch.matmul(self.P, H.T), temp), torch.matmul(H, self.P))
          pHT = torch.matmul(self.P, H.T)
          Hbeta = torch.matmul(H, self.beta)
          self.beta += torch.matmul(pHT, target - Hbeta)
          return self.beta

      def predict(self, features, beta):
          H = self.forward_L(features)
          prediction = torch.matmul(H, beta).view(-1,1)
          return prediction

      def svd_PIV(self, x):
          u, sigma, v = torch.svd(x, some=True)
          s_plus = torch.zeros_like(x).T
          for i in range(len(sigma)):
              s_plus[i, i] = 1 / (sigma[i] + 1e-10)
          piv = torch.mm(v.t(), torch.mm(s_plus, u.t()))
          return piv
