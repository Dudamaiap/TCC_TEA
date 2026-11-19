"""MLP e BrainNetCNN compacta para conectividade funcional ABIDE/CC200."""
import torch
from torch import nn

class MLPConectividade(nn.Module):
    """MLP sobre vetor de conexões Fisher Z, com dimensões inferidas do dataset."""
    def __init__(self, n_atributos, dropout1=0.50, dropout2=0.40):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_atributos,128), nn.ReLU(), nn.Dropout(dropout1),
                                 nn.Linear(128,64), nn.ReLU(), nn.Dropout(dropout2), nn.Linear(64,1))
    def forward(self,x): return self.net(x).squeeze(1)

class E2E(nn.Module):
    def __init__(self,n_rois,in_ch,out_ch):
        super().__init__(); self.row_weight=nn.Parameter(torch.empty(out_ch,in_ch,n_rois)); self.col_weight=nn.Parameter(torch.empty(out_ch,in_ch,n_rois))
        self.row_bias=nn.Parameter(torch.zeros(out_ch)); self.col_bias=nn.Parameter(torch.zeros(out_ch))
        nn.init.xavier_uniform_(self.row_weight); nn.init.xavier_uniform_(self.col_weight)
    def forward(self,x):
        row=torch.einsum('bcij,fcj->bfi',x,self.row_weight)+self.row_bias[None,:,None]
        col=torch.einsum('bcij,fci->bfj',x,self.col_weight)+self.col_bias[None,:,None]
        return row.unsqueeze(-1)+col.unsqueeze(-2)

class E2N(nn.Module):
    def __init__(self,n_rois,in_ch,out_ch):
        super().__init__(); self.row_weight=nn.Parameter(torch.empty(out_ch,in_ch,n_rois)); self.col_weight=nn.Parameter(torch.empty(out_ch,in_ch,n_rois))
        self.row_bias=nn.Parameter(torch.zeros(out_ch)); self.col_bias=nn.Parameter(torch.zeros(out_ch))
        nn.init.xavier_uniform_(self.row_weight); nn.init.xavier_uniform_(self.col_weight)
    def forward(self,x):
        row=torch.einsum('bcij,fcj->bfi',x,self.row_weight)+self.row_bias[None,:,None]
        col=torch.einsum('bcij,fci->bfj',x,self.col_weight)+self.col_bias[None,:,None]
        return (row+col).unsqueeze(-1)

class N2G(nn.Module):
    def __init__(self,n_rois,in_ch,out_ch):
        super().__init__(); self.weight=nn.Parameter(torch.empty(out_ch,in_ch,n_rois)); self.bias=nn.Parameter(torch.zeros(out_ch)); nn.init.xavier_uniform_(self.weight)
    def forward(self,x): return (torch.einsum('bcn,fcn->bf',x.squeeze(-1),self.weight)+self.bias[None,:])[:,:,None,None]

class BrainNetCNN(nn.Module):
    """E2E -> E2N -> N2G sobre matriz de conectividade simétrica."""
    def __init__(self,n_rois=200,e2e=8,e2n=16,n2g=16,dropout=0.5):
        super().__init__(); self.e2e=nn.Sequential(E2E(n_rois,1,e2e),nn.LeakyReLU(.33)); self.e2n=nn.Sequential(E2N(n_rois,e2e,e2n),nn.LeakyReLU(.33),nn.Dropout(dropout)); self.n2g=nn.Sequential(N2G(n_rois,e2n,n2g),nn.LeakyReLU(.33)); self.classifier=nn.Linear(n2g,1)
    def forward(self,x):
        if x.ndim!=4: raise ValueError('Esperado tensor 4D de matrizes de conectividade.')
        if x.shape[-1]==1 and x.shape[1]==x.shape[2]: x=x.permute(0,3,1,2)
        if x.shape[1]!=1 or x.shape[-1]!=x.shape[-2]: raise ValueError('Esperada matriz quadrada com um canal.')
        x=(x+x.transpose(-1,-2))*.5; x=self.e2e(x); x=self.e2n(x); x=self.n2g(x)
        return self.classifier(x.flatten(1)).squeeze(1)
