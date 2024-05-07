import random

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.distributions as td
from torch import nn, optim

class MIWAE():

    def __init__(self, n_neurons=128, batch_size=64, lr=0.001, n_epochs=100, dim_Z=20, K=20, random_state=None):
        self.n_neurons = n_neurons # number of hidden units in (same for all MLPs)
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.dim_Z = dim_Z # dimension of the latent space
        self.K = K # number of IS during training
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Define prior
        self.p_z = td.Independent(td.Normal(loc=torch.zeros(self.dim_Z).to(self.device),scale=torch.ones(self.dim_Z).to(self.device)),1)

        # Define encoder and decoder
        self.encoder = None
        self.decoder = None

        if random_state is not None:
            initialise_random_state(random_state)

    def fit(self, xmiss):
        if isinstance(xmiss, pd.DataFrame):
            xmiss = xmiss.to_numpy(dtype="float32")
        miwae_loss_train=np.array([])
        mse_train=np.array([])
        mse_train2=np.array([])
        bs = self.batch_size # batch size
        n_epochs = self.n_epochs

        # Create binary mask that indicates which values are missing
        mask = np.isfinite(xmiss)
        # Preimpute with zeros
        xhat_0 = np.copy(xmiss)
        xhat_0[np.isnan(xmiss)] = 0
        xhat = np.copy(xhat_0) # This will be out imputed data matrix

        # Get data dimensions
        n = xmiss.shape[0] # number of observations
        self.n_features = xmiss.shape[1] # number of features

        # Define encoder
        self.encoder = nn.Sequential(
            torch.nn.Linear(self.n_features, self.n_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_neurons, self.n_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_neurons, 2*self.dim_Z),  # the encoder will output both the mean and the diagonal covariance
        )

        # Define decoder
        self.decoder = nn.Sequential(
            torch.nn.Linear(self.dim_Z, self.n_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_neurons, self.n_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_neurons, 3*self.n_features),  # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*self.n_features)
        )

        # Send networks to device
        self.encoder.to(self.device) # we'll use the GPU
        self.decoder.to(self.device)

        # Initialize weights
        self.encoder.apply(self.weights_init)
        self.decoder.apply(self.weights_init)

        # Define the optimizer
        optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),lr=self.lr)

        for ep in range(1,n_epochs):
            perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
            batches_data = np.array_split(xhat_0[perm,], n/bs)
            batches_mask = np.array_split(mask[perm,], n/bs)
            for it in range(len(batches_data)):
                optimizer.zero_grad()
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                b_data = torch.from_numpy(batches_data[it]).float().to(self.device)
                b_mask = torch.from_numpy(batches_mask[it]).float().to(self.device)
                loss = self.miwae_loss(iota_x = b_data,mask = b_mask)
                loss.backward()
                optimizer.step()
            if ep % 100 == 1:
                print('Epoch %g' %ep)
                print('MIWAE likelihood bound  %g' %(-np.log(self.K) -
                                                      self.miwae_loss(iota_x = torch.from_numpy(xhat_0).float().to(self.device),
                                                                        mask = torch.from_numpy(mask).float().to(self.device)).cpu().data.numpy()))

    def transform(self, xmiss):
        mask = np.isfinite(xmiss)

        # print(mask[369,:])

        xhat_0 = np.copy(xmiss)
        xhat_0[np.isnan(xmiss)] = 0
        xhat = np.copy(xhat_0) # This will be out imputed data matrix

        # print('xhat')
        # print(xhat[369,:])
        #import pdb; pdb.set_trace()
        res1 = self.miwae_impute(iota_x = torch.from_numpy(xhat_0).float().to(self.device),
                                         mask = torch.from_numpy(mask).float().to(self.device),L=10).cpu().data.numpy()
        # print(res1[369,:])
        xhat[~mask] = res1[~mask]
        # print(xhat[369,:])
        
        return xhat

    def miwae_loss(self, iota_x, mask):
        batch_size = iota_x.shape[0]
        out_encoder = self.encoder(iota_x)
        # I am adding an epsilon 1e-5 for numerical stability
        q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :self.dim_Z],scale=1e-5 + torch.nn.Softplus()(out_encoder[..., self.dim_Z:(2*self.dim_Z)])),1)

        zgivenx = q_zgivenxobs.rsample([self.K])
        zgivenx_flat = zgivenx.reshape([self.K*batch_size,self.dim_Z])

        out_decoder = self.decoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :self.n_features]
        all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., self.n_features:(2*self.n_features)]) + 0.001
        all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*self.n_features):(3*self.n_features)]) + 3

        data_flat = torch.Tensor.repeat(iota_x,[self.K,1]).reshape([-1,1])
        tiledmask = torch.Tensor.repeat(mask,[self.K,1])

        all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([self.K*batch_size,self.n_features])

        logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([self.K,batch_size])
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)

        neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq,0))

        return neg_bound

    def miwae_impute(self, iota_x,mask,L):
        batch_size = iota_x.shape[0]
        out_encoder = self.encoder(iota_x)
        q_zgivenxobs = td.Independent(
            td.Normal(
                loc=out_encoder[..., :self.dim_Z],
                scale=torch.nn.Softplus()(out_encoder[..., self.dim_Z:(2*self.dim_Z)]) + 1e-8
            ),
            1
        )

        zgivenx = q_zgivenxobs.rsample([L])
        zgivenx_flat = zgivenx.reshape([L*batch_size,self.dim_Z])

        out_decoder = self.decoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :self.n_features]
        all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., self.n_features:(2*self.n_features)]) + 0.001
        all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*self.n_features):(3*self.n_features)]) + 3

        data_flat = torch.Tensor.repeat(iota_x,[L,1]).reshape([-1,1]).to(self.device)
        tiledmask = torch.Tensor.repeat(mask,[L,1]).to(self.device)

        all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L*batch_size,self.n_features])

        logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([L,batch_size])
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)

        xgivenz = td.Independent(td.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model),1)

        imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
        xms = xgivenz.sample().reshape([L,batch_size,self.n_features])
        xm=torch.einsum('ki,kij->ij', imp_weights, xms)

        return xm

    def mse(self, xhat,xtrue,mask): # MSE function for imputations
        xhat = np.array(xhat)
        xtrue = np.array(xtrue)
        return np.mean(np.power(xhat-xtrue,2)[~mask])

    def weights_init(self, layer):
        if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)


def initialise_random_state(seed):
    """Set random seed for `random`, `tf` and `numpy`

    Based on fastai.torch_core.
    """
    s = seed % (2**32-1)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
