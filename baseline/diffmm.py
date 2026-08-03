import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import scipy.sparse as sp


class GCNLayer(nn.Module):
    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds) if adj.is_sparse else adj @ embeds


class SpAdjDropEdge(nn.Module):
    def __init__(self, keep_rate=0.5):
        super().__init__()
        self.keep_rate = keep_rate

    def forward(self, adj):
        if not adj.is_sparse or self.keep_rate >= 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edge_num = vals.size(0)
        mask = ((torch.rand(edge_num, device=adj.device) + self.keep_rate).floor()).bool()
        new_vals = vals[mask] / self.keep_rate
        new_idxs = idxs[:, mask]
        return torch.sparse_coo_tensor(new_idxs, new_vals, adj.shape).coalesce()


class Denoise(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size=10, dropout=0.5, norm=False):
        super().__init__()
        self.time_emb_dim = emb_size
        self.norm = norm
        self.emb_layer = nn.Linear(emb_size, emb_size)
        in_dims_temp = [in_dims[0] + emb_size] + in_dims[1:]
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out)
                                        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out)
                                         for d_in, d_out in zip(out_dims[:-1], out_dims[1:])])
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for layer in self.in_layers:
            std = np.sqrt(2.0 / (layer.weight.size(0) + layer.weight.size(1)))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        for layer in self.out_layers:
            std = np.sqrt(2.0 / (layer.weight.size(0) + layer.weight.size(1)))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        std = np.sqrt(2.0 / (self.emb_layer.weight.size(0) + self.emb_layer.weight.size(1)))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps):
        device = x.device
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim // 2,
                                                           dtype=torch.float32, device=device) / (self.time_emb_dim // 2))
        temp = timesteps[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for layer in self.in_layers:
            h = layer(h)
            h = torch.tanh(h)
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        return h


class GaussianDiffusion(nn.Module):
    def __init__(self, noise_scale=0.1, noise_min=0.0001, noise_max=0.02, steps=5):
        super().__init__()
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        betas = torch.tensor(self._get_betas(), dtype=torch.float64)
        betas[0] = 0.0001
        self._calculate_for_diffusion(betas)

    def _get_betas(self):
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        betas = [1 - alpha_bar[0]]
        for i in range(1, self.steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], 0.999))
        return np.array(betas)

    def _calculate_for_diffusion(self, betas):
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
               self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def _extract(self, arr, t, shape):
        res = arr[t].float()
        while len(res.shape) < len(shape):
            res = res[..., None]
        return res.expand(shape)

    def p_mean_variance(self, model, x, t):
        model_output = model(x, t)
        var = self._extract(self.posterior_variance, t, x.shape)
        log_var = self._extract(self.posterior_log_variance_clipped, t, x.shape)
        mean = self._extract(self.posterior_mean_coef1, t, x.shape) * model_output + \
               self._extract(self.posterior_mean_coef2, t, x.shape) * x
        return mean, log_var

    def p_sample(self, model, x_start, steps, sampling_noise=False):
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps - 1] * x_start.shape[0], device=x_start.device)
            x_t = self.q_sample(x_start, t)
        for i in range(self.steps - 1, -1, -1):
            t = torch.tensor([i] * x_t.shape[0], device=x_t.device)
            mean, log_var = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                x_t = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise
            else:
                x_t = mean
        return x_t

    def training_losses(self, model, x_start, itm_embeds, batch_index, model_feats):
        batch_size = x_start.size(0)
        t = torch.randint(0, self.steps, (batch_size,), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        model_output = model(x_t, t)
        mse = ((x_start - model_output) ** 2).mean(dim=list(range(1, len(x_start.shape))))
        snr = self.alphas_cumprod.to(x_start.device)[t] / (1 - self.alphas_cumprod.to(x_start.device)[t])
        weight = torch.where(t == 0, torch.ones_like(snr), snr - self.alphas_cumprod.to(x_start.device)[t.clamp(min=1) - 1] /
                             (1 - self.alphas_cumprod.to(x_start.device)[t.clamp(min=1) - 1]))
        diff_loss = weight * mse
        usr_model = model_output @ model_feats
        usr_id = x_start @ itm_embeds
        gc_loss = ((usr_model - usr_id) ** 2).mean(dim=list(range(1, len(x_start.shape))))
        return diff_loss, gc_loss


class DiffMM(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, modality_dims, n_layers=1,
                 keep_rate=0.5, ssl_reg=0.01, temp=0.5, ris_lambda=0.5,
                 ris_adj_lambda=0.2, e_loss=0.1, reg=1e-5,
                 noise_scale=0.1, noise_min=0.0001, noise_max=0.02,
                 diff_steps=5, rebuild_k=1, dims=None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.keep_rate = keep_rate
        self.ssl_reg = ssl_reg
        self.temp = temp
        self.ris_lambda = ris_lambda
        self.ris_adj_lambda = ris_adj_lambda
        self.e_loss = e_loss
        self.reg = reg
        self.rebuild_k = rebuild_k

        self.uEmbeds = nn.Parameter(torch.empty(n_users, embed_dim))
        self.iEmbeds = nn.Parameter(torch.empty(n_items, embed_dim))
        nn.init.xavier_uniform_(self.uEmbeds)
        nn.init.xavier_uniform_(self.iEmbeds)

        self.gcnLayers = nn.Sequential(*[GCNLayer() for _ in range(n_layers)])
        self.edgeDropper = SpAdjDropEdge(keep_rate)

        self.modality_trans = nn.ModuleDict()
        for k, dim in modality_dims.items():
            self.modality_trans[k] = nn.Linear(dim, embed_dim)

        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5][:len(modality_dims)]))
        self.softmax = nn.Softmax(dim=0)
        self.leakyrelu = nn.LeakyReLU(0.2)

        self.diffusion = GaussianDiffusion(noise_scale, noise_min, noise_max, diff_steps)
        diff_dims = dims if dims else [1000]
        out_dims = diff_dims + [n_items]
        in_dims = out_dims[::-1]
        self.denoise_models = nn.ModuleDict()
        for k in modality_dims:
            self.denoise_models[k] = Denoise(in_dims, out_dims, emb_size=10, norm=False, dropout=0.5)

        self._modality_adjs = None
        self._ui_adj = None

    def set_precomputed_adj(self, adj_matrices):
        if adj_matrices is not None:
            image_adj = adj_matrices.get('image_adj', None)
            text_adj = adj_matrices.get('text_adj', None)
            if image_adj is not None:
                self._modality_adjs = {'visual': image_adj, 'textual': text_adj}

    def _normalize_adj(self, mat):
        if mat.is_sparse:
            mat = mat.to_dense()
        degree = mat.sum(dim=-1)
        d_inv_sqrt = torch.pow(degree.clamp_min(1e-8), -0.5)
        d_inv_sqrt = torch.where(torch.isinf(d_inv_sqrt), torch.zeros_like(d_inv_sqrt), d_inv_sqrt)
        return (mat * d_inv_sqrt.unsqueeze(1)) * d_inv_sqrt.unsqueeze(0)

    def _build_ui_matrix(self, u_list, i_list, edge_list, device):
        if len(u_list) == 0:
            return torch.eye(self.n_users + self.n_items, device=device).to_sparse()
        mat = torch.zeros(self.n_users, self.n_items, device=device)
        mat[u_list, i_list] = edge_list
        a = torch.zeros(self.n_users, self.n_users, device=device)
        b = torch.zeros(self.n_items, self.n_items, device=device)
        bipartite = torch.cat([torch.cat([a, mat], dim=1), torch.cat([mat.T, b], dim=1)], dim=0)
        bipartite = (bipartite != 0).float()
        bipartite = bipartite + torch.eye(bipartite.shape[0], device=device)
        return self._normalize_adj(bipartite).to_sparse().coalesce()

    def _get_modality_feats(self, modality_features):
        feats = {}
        for k in modality_features:
            feats[k] = self.leakyrelu(self.modality_trans[k](modality_features[k]))
        return feats

    def _forward_mm(self, adj, modality_adjs, modality_features):
        modality_feats = self._get_modality_feats(modality_features)
        weight = self.softmax(self.modal_weight)

        modality_embeds = []
        modality_keys = list(modality_features.keys())
        for k in modality_keys:
            feat = F.normalize(modality_feats[k])
            adj_mod = modality_adjs[k] if modality_adjs and k in modality_adjs else adj

            embeds = torch.cat([self.uEmbeds, feat])
            embeds = torch.spmm(adj, embeds) if adj.is_sparse else adj @ embeds
            embeds_ = torch.cat([embeds[:self.n_users], self.iEmbeds])
            embeds_ = torch.spmm(adj, embeds_) if adj.is_sparse else adj @ embeds_
            embeds = embeds + embeds_

            id_embeds = torch.cat([self.uEmbeds, self.iEmbeds])
            id_embeds = torch.spmm(adj_mod, id_embeds) if adj_mod.is_sparse else adj_mod @ id_embeds
            embeds = embeds + self.ris_adj_lambda * id_embeds
            modality_embeds.append(embeds)

        combined = weight[0] * modality_embeds[0]
        for i in range(1, len(modality_embeds)):
            combined = combined + weight[i] * modality_embeds[i]

        embeds = combined
        embeds_list = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        embeds = embeds + self.ris_lambda * F.normalize(combined)
        return embeds[:self.n_users], embeds[self.n_users:]

    def _forward_cl_mm(self, adj, modality_adjs, modality_features):
        modality_feats = self._get_modality_feats(modality_features)
        modality_keys = list(modality_features.keys())
        views = []
        for k in modality_keys:
            feat = F.normalize(modality_feats[k])
            adj_mod = modality_adjs[k] if modality_adjs and k in modality_adjs else adj
            embeds = torch.cat([self.uEmbeds, feat])
            embeds = torch.spmm(adj_mod, embeds) if adj_mod.is_sparse else adj_mod @ embeds
            embeds_list = [embeds]
            for gcn in self.gcnLayers:
                embeds = gcn(adj, embeds_list[-1])
                embeds_list.append(embeds)
            embeds = sum(embeds_list)
            views.append((embeds[:self.n_users], embeds[self.n_users:]))
        return views

    def _contrastive_loss(self, e1, e2, ids, temp):
        e1_norm = F.normalize(e1[ids])
        e2_norm = F.normalize(e2[ids])
        sim = e1_norm @ e2_norm.T / temp
        labels = torch.arange(sim.shape[0], device=sim.device)
        return F.cross_entropy(sim, labels)

    def _build_norm_adj(self, graph_norm):
        if graph_norm.is_sparse:
            return graph_norm
        return graph_norm

    def compute_loss(self, user_ids, pos_ids, neg_ids, graph_norm, modality_features):
        if self._modality_adjs is None:
            modality_adjs = {}
            for k in modality_features:
                modality_adjs[k] = graph_norm
        else:
            modality_adjs = {}
            for k in modality_features:
                modality_adjs[k] = graph_norm
        usr_embeds, itm_embeds = self._forward_mm(graph_norm, modality_adjs, modality_features)
        ancs = user_ids
        poss = pos_ids
        negs = neg_ids

        score_diff = (usr_embeds[ancs] * itm_embeds[poss]).sum(dim=1) - \
                     (usr_embeds[ancs] * itm_embeds[negs]).sum(dim=1)
        bpr_loss = -F.logsigmoid(score_diff).sum() / max(user_ids.shape[0], 1)

        reg_loss = (self.uEmbeds.norm(2).pow(2) + self.iEmbeds.norm(2).pow(2)) * self.reg

        views = self._forward_cl_mm(graph_norm, modality_adjs, modality_features)
        cl_loss = torch.tensor(0.0, device=user_ids.device)
        main_u, main_i = usr_embeds, itm_embeds
        for i in range(len(views)):
            v_u, v_i = views[i]
            cl_loss = cl_loss + self._contrastive_loss(main_u, v_u, ancs, self.temp) * self.ssl_reg
            cl_loss = cl_loss + self._contrastive_loss(main_i, v_i, poss, self.temp) * self.ssl_reg
            for j in range(i + 1, len(views)):
                v_u2, v_i2 = views[j]
                cl_loss = cl_loss + self._contrastive_loss(v_u, v_u2, ancs, self.temp) * self.ssl_reg
                cl_loss = cl_loss + self._contrastive_loss(v_i, v_i2, poss, self.temp) * self.ssl_reg

        return bpr_loss + reg_loss + cl_loss

    def get_embs(self, graph_norm, modality_features):
        if self._modality_adjs is None:
            modality_adjs = {}
            for k in modality_features:
                modality_adjs[k] = graph_norm
        else:
            modality_adjs = {}
            for k in modality_features:
                modality_adjs[k] = graph_norm
        return self._forward_mm(graph_norm, modality_adjs, modality_features)
