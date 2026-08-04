import os
import csv
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from collections import defaultdict


class DAMRSDataset:
    def __init__(self, dataset_name, data_dir='data/damrs/', embed_dim=64, split_ratio=(8, 1, 1)):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.embed_dim = embed_dim
        self.split_ratio = split_ratio
        self.n_users = 0
        self.n_items = 0
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_user_item_dict = defaultdict(set)
        self.val_user_item_dict = defaultdict(set)
        self.test_user_item_dict = defaultdict(set)
        self.user_item_dict = defaultdict(set)
        self.item_user_dict = defaultdict(set)
        self.visual_features = None
        self.textual_features = None
        self.image_adj = None
        self.text_adj = None
        self.interaction_matrix = None
        self.graph = None
        self.inter_norm_u = None
        self.inter_norm_v = None
        self._load_data()

    def _load_data(self):
        ds_dir = os.path.join(self.data_dir, self.dataset_name)

        inter_file = os.path.join(ds_dir, f'{self.dataset_name}.inter')
        if not os.path.exists(inter_file):
            raise FileNotFoundError(f'Inter file not found: {inter_file}')

        v_feat_file = os.path.join(ds_dir, 'image_feat.npy')
        t_feat_file = os.path.join(ds_dir, 'text_feat.npy')
        if not os.path.exists(v_feat_file) or not os.path.exists(t_feat_file):
            raise FileNotFoundError(f'Modality features not found in {ds_dir}')

        self.visual_features = np.load(v_feat_file, allow_pickle=True).astype(np.float32)
        self.textual_features = np.load(t_feat_file, allow_pickle=True).astype(np.float32)
        self.n_items = self.visual_features.shape[0]

        image_adj_file = os.path.join(ds_dir, 'image_adj_80.pt')
        text_adj_file = os.path.join(ds_dir, 'text_adj_80.pt')
        if os.path.exists(image_adj_file):
            self.image_adj = self._load_adj(image_adj_file)
        else:
            self.image_adj = None
        if os.path.exists(text_adj_file):
            self.text_adj = self._load_adj(text_adj_file)
        else:
            self.text_adj = None

        if not self._load_split_cache(ds_dir, inter_file):
            self._load_interactions_from_text(inter_file)
            self._save_split_cache(ds_dir, inter_file)

        max_inter_item_id = self._max_inter_item_id()
        feature_item_count = self.visual_features.shape[0]
        if self.textual_features.shape[0] != feature_item_count:
            raise ValueError(
                f'Feature item count mismatch: image={feature_item_count}, '
                f'text={self.textual_features.shape[0]}'
            )
        if max_inter_item_id >= feature_item_count:
            raise ValueError(
                f'Interaction item id {max_inter_item_id} exceeds feature item count '
                f'{feature_item_count} in {self.dataset_name}'
            )
        self.n_items = feature_item_count

        self._build_interaction_matrix()
        if not self._load_graph_cache(ds_dir):
            self._build_graph()
            self._save_graph_cache(ds_dir)

        print(f'{self.dataset_name}: n_users={self.n_users}, n_items={self.n_items}, '
              f'train={len(self.train_data)}, val={len(self.val_data)}, test={len(self.test_data)}, '
              f'total={len(self.train_data)+len(self.val_data)+len(self.test_data)}')
        print(f'  image_feat: {self.visual_features.shape}, text_feat: {self.textual_features.shape}')
        if self.image_adj is not None:
            print(f'  image_adj: {self.image_adj.shape}, text_adj: {self.text_adj.shape}')

    def _split_cache_path(self, ds_dir):
        return os.path.join(ds_dir, '_split_cache.pkl')

    def _inter_file_signature(self, inter_file):
        stat = os.stat(inter_file)
        return {
            'path': os.path.abspath(inter_file),
            'size': int(stat.st_size),
            'mtime_ns': int(stat.st_mtime_ns),
        }

    def _load_split_cache(self, ds_dir, inter_file):
        cache_path = self._split_cache_path(ds_dir)
        if not os.path.exists(cache_path):
            return False
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
        except Exception as exc:
            print(f'  split cache ignored ({exc}); reparsing')
            return False
        if cache.get('inter_file') != self._inter_file_signature(inter_file):
            print('  split cache metadata mismatch; reparsing')
            return False
        required_keys = {
            'train_data', 'val_data', 'test_data', 'n_users',
            'train_user_item_dict', 'val_user_item_dict', 'test_user_item_dict',
            'user_item_dict', 'item_user_dict'
        }
        if not required_keys.issubset(cache.keys()):
            print('  split cache schema mismatch; reparsing')
            return False

        self.train_data = cache['train_data']
        self.val_data = cache['val_data']
        self.test_data = cache['test_data']
        self.n_users = cache['n_users']
        self.train_user_item_dict = defaultdict(set, cache['train_user_item_dict'])
        self.val_user_item_dict = defaultdict(set, cache['val_user_item_dict'])
        self.test_user_item_dict = defaultdict(set, cache['test_user_item_dict'])
        self.user_item_dict = defaultdict(set, cache['user_item_dict'])
        self.item_user_dict = defaultdict(set, cache['item_user_dict'])
        return True

    def _save_split_cache(self, ds_dir, inter_file):
        cache = {
            'inter_file': self._inter_file_signature(inter_file),
            'train_data': self.train_data,
            'val_data': self.val_data,
            'test_data': self.test_data,
            'n_users': int(self.n_users),
            'train_user_item_dict': dict(self.train_user_item_dict),
            'val_user_item_dict': dict(self.val_user_item_dict),
            'test_user_item_dict': dict(self.test_user_item_dict),
            'user_item_dict': dict(self.user_item_dict),
            'item_user_dict': dict(self.item_user_dict),
        }
        with open(self._split_cache_path(ds_dir), 'wb') as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_interactions_from_text(self, inter_file):
        train_pairs = []
        val_pairs = []
        test_pairs = []
        all_u = set()

        with open(inter_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            x_label_idx = header.index('x_label')
            user_idx = header.index('userID')
            item_idx = header.index('itemID')
            for row in reader:
                u = int(row[user_idx])
                v = int(row[item_idx])
                label = int(row[x_label_idx])
                all_u.add(u)
                self.user_item_dict[u].add(v)
                self.item_user_dict[v].add(u)
                if label == 0:
                    train_pairs.append((u, v))
                    self.train_user_item_dict[u].add(v)
                elif label == 1:
                    val_pairs.append((u, v))
                    self.val_user_item_dict[u].add(v)
                elif label == 2:
                    test_pairs.append((u, v))
                    self.test_user_item_dict[u].add(v)

        self.train_data = np.array(train_pairs, dtype=np.int64).reshape(-1, 2)
        self.val_data = np.array(val_pairs, dtype=np.int64).reshape(-1, 2)
        self.test_data = np.array(test_pairs, dtype=np.int64).reshape(-1, 2)
        self.n_users = max(all_u) + 1 if all_u else 0

    def _max_inter_item_id(self):
        max_ids = []
        for data in (self.train_data, self.val_data, self.test_data):
            if data.size:
                max_ids.append(int(data[:, 1].max()))
        return max(max_ids) if max_ids else -1

    def _load_adj(self, path):
        adj = torch.load(path, map_location='cpu').float()
        if adj.is_sparse:
            return adj.coalesce()
        return adj.to_sparse().coalesce()

    def _build_interaction_matrix(self):
        rows = self.train_data[:, 0]
        cols = self.train_data[:, 1]
        data = np.ones(len(rows), dtype=np.float32)
        self.interaction_matrix = sp.coo_matrix(
            (data, (rows, cols)), shape=(self.n_users, self.n_items)
        ).tocsr()

    def _build_graph(self):
        inter_M = self.interaction_matrix.tocoo()
        inter_M_t = inter_M.transpose()

        # Symmetric normalized Laplacian for ID embeddings (LightGCN, Eq 2-3)
        A = sp.bmat(
            [[None, inter_M], [inter_M_t, None]],
            format='coo',
            dtype=np.float32
        )

        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D_sym = sp.diags(diag)
        L = D_sym * A * D_sym

        L = sp.coo_matrix(L)
        row, col = L.row, L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        self.graph = torch.sparse_coo_tensor(i, data, (self.n_users + self.n_items, self.n_users + self.n_items))

        # Asymmetric normalized matrices for modality propagation (Eq 1)
        # D_U^{-1} R and D_V^{-1} R^T stored as sparse tensors
        user_deg = np.array(inter_M.sum(axis=1)).flatten() + 1e-7
        D_U_inv = sp.diags(1.0 / user_deg)
        norm_u_sparse = (D_U_inv @ inter_M).tocoo()

        item_deg = np.array(inter_M.sum(axis=0)).flatten() + 1e-7
        D_V_inv = sp.diags(1.0 / item_deg)
        norm_v_sparse = (D_V_inv @ inter_M_t).tocoo()

        self.inter_norm_u = torch.sparse_coo_tensor(
            torch.LongTensor(np.array([norm_u_sparse.row, norm_u_sparse.col])),
            torch.FloatTensor(norm_u_sparse.data),
            (self.n_users, self.n_items)
        ).coalesce()

        self.inter_norm_v = torch.sparse_coo_tensor(
            torch.LongTensor(np.array([norm_v_sparse.row, norm_v_sparse.col])),
            torch.FloatTensor(norm_v_sparse.data),
            (self.n_items, self.n_users)
        ).coalesce()

    def _graph_cache_path(self, ds_dir):
        return os.path.join(ds_dir, '_graph_cache.pt')

    def _graph_cache_meta(self):
        return {
            'n_users': int(self.n_users),
            'n_items': int(self.n_items),
            'train_size': int(len(self.train_data)),
            'val_size': int(len(self.val_data)),
            'test_size': int(len(self.test_data)),
            'interaction_nnz': int(self.interaction_matrix.nnz),
            'visual_features_shape': tuple(self.visual_features.shape),
            'textual_features_shape': tuple(self.textual_features.shape),
        }

    def _load_graph_cache(self, ds_dir):
        cache_path = self._graph_cache_path(ds_dir)
        if not os.path.exists(cache_path):
            return False

        try:
            cache = torch.load(cache_path, map_location='cpu')
        except Exception as exc:
            print(f'  graph cache ignored ({exc}); rebuilding')
            return False

        if cache.get('meta') != self._graph_cache_meta():
            print('  graph cache metadata mismatch; rebuilding')
            return False

        self.graph = cache['graph'].coalesce()
        self.inter_norm_u = cache['inter_norm_u'].coalesce()
        self.inter_norm_v = cache['inter_norm_v'].coalesce()
        return True

    def _save_graph_cache(self, ds_dir):
        cache_path = self._graph_cache_path(ds_dir)
        cache = {
            'meta': self._graph_cache_meta(),
            'graph': self.graph.cpu().coalesce(),
            'inter_norm_u': self.inter_norm_u.cpu().coalesce(),
            'inter_norm_v': self.inter_norm_v.cpu().coalesce(),
        }
        torch.save(cache, cache_path)

    def get_modality_features(self):
        visual = torch.FloatTensor(self.visual_features)
        textual = torch.FloatTensor(self.textual_features)
        visual = F.normalize(visual, dim=1)
        textual = F.normalize(textual, dim=1)
        return {
            'visual': visual,
            'textual': textual,
        }

    def get_modality_features_dim(self):
        dims = {}
        if self.visual_features is not None:
            dims['visual'] = self.visual_features.shape[1]
        if self.textual_features is not None:
            dims['textual'] = self.textual_features.shape[1]
        return dims

    def get_norm_graph(self):
        return self.graph

    def get_modality_norm_matrices(self):
        return {
            'inter_norm_u': self.inter_norm_u,
            'inter_norm_v': self.inter_norm_v,
        }

    def get_adj_matrices(self):
        return {
            'image_adj': self.image_adj,
            'text_adj': self.text_adj,
        }

    def inter_matrix(self, form='coo'):
        if form == 'coo':
            return self.interaction_matrix.tocoo()
        elif form == 'csr':
            return self.interaction_matrix.tocsr()
        return self.interaction_matrix
