import torch
import numpy as np
from collections import defaultdict


class BPRDataLoader:
    def __init__(self, train_data, n_users, n_items, train_user_item_dict,
                 batch_size=4096, user_item_dict=None, context_max=4):
        self.train_data = train_data
        self.n_users = n_users
        self.n_items = n_items
        self.train_user_item_dict = train_user_item_dict
        self.neg_exclude_dict = user_item_dict if user_item_dict is not None else train_user_item_dict
        self.batch_size = batch_size
        self.context_max = context_max
        self.all_items = list(range(n_items))
        self._exclude_users = set(self.neg_exclude_dict.keys())
        self._train_item_lists = {}
        for u in self.train_user_item_dict:
            self._train_item_lists[u] = list(self.train_user_item_dict[u])

    def __len__(self):
        return max(1, int(np.ceil(len(self.train_data) / self.batch_size)))

    def get_batch(self):
        indices = np.random.randint(0, len(self.train_data), self.batch_size)
        batch = self.train_data[indices]
        user_ids = batch[:, 0].astype(np.int64, copy=False)
        pos_ids = batch[:, 1].astype(np.int64, copy=False)
        neg_ids = np.random.randint(0, self.n_items, size=self.batch_size, dtype=np.int64)

        invalid = self._invalid_negative_mask(user_ids, neg_ids)
        max_retries = 50
        retry_count = 0
        while invalid.any() and retry_count < max_retries:
            neg_ids[invalid] = np.random.randint(0, self.n_items, size=int(invalid.sum()), dtype=np.int64)
            invalid = self._invalid_negative_mask(user_ids, neg_ids, invalid)
            retry_count += 1

        context_ids = self._sample_contexts(user_ids, pos_ids)

        return (torch.from_numpy(user_ids.copy()).long(),
                torch.from_numpy(pos_ids.copy()).long(),
                torch.from_numpy(neg_ids).long(),
                context_ids)

    def _sample_contexts(self, user_ids, pos_ids):
        B = len(user_ids)
        C_max = self.context_max
        context_ids = np.full((B, C_max), -1, dtype=np.int64)
        for b in range(B):
            u = int(user_ids[b])
            pos = int(pos_ids[b])
            if u not in self._train_item_lists:
                continue
            candidates = [i for i in self._train_item_lists[u] if i != pos]
            n_ctx = min(C_max, len(candidates))
            if n_ctx > 0:
                chosen = np.random.choice(candidates, size=n_ctx, replace=False)
                context_ids[b, :n_ctx] = chosen
        return torch.from_numpy(context_ids).long()

    def _invalid_negative_mask(self, user_ids, neg_ids, candidate_mask=None):
        if candidate_mask is None:
            mask = np.zeros(len(user_ids), dtype=bool)
            candidate_indices = range(len(user_ids))
        else:
            mask = candidate_mask.copy()
            candidate_indices = np.flatnonzero(candidate_mask)

        for idx in candidate_indices:
            u = int(user_ids[idx])
            if u in self._exclude_users and int(neg_ids[idx]) in self.neg_exclude_dict[u]:
                mask[idx] = True
            elif candidate_mask is not None:
                mask[idx] = False
        return mask

    def shuffle(self):
        np.random.shuffle(self.train_data)


class EvalDataLoader:
    def __init__(self, dataset, batch_size=4096):
        self.dataset = dataset
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.batch_size = batch_size

        self.train_user_item_dict = dataset.train_user_item_dict
        self.val_user_item_dict = defaultdict(set)
        self.test_user_item_dict = defaultdict(set)

        for u, v in dataset.val_data:
            self.val_user_item_dict[u].add(v)
        for u, v in dataset.test_data:
            self.test_user_item_dict[u].add(v)

        self.val_users = sorted(self.val_user_item_dict.keys())
        self.test_users = sorted(self.test_user_item_dict.keys())

    def get_eval_users(self, mode='test'):
        users = self.test_users if mode == 'test' else self.val_users
        return torch.LongTensor(users)

    def get_pos_items_per_user(self, mode='test'):
        users = self.test_users if mode == 'test' else self.val_users
        target_dict = self.test_user_item_dict if mode == 'test' else self.val_user_item_dict
        result = []
        for u in users:
            result.append(list(target_dict[u]))
        return result

    def get_train_mask_items(self, users):
        u_ids = []
        i_ids = []
        for i, u in enumerate(users):
            items = list(self.train_user_item_dict.get(u, set()))
            u_ids.extend([i] * len(items))
            i_ids.extend(items)
        return torch.LongTensor(u_ids), torch.LongTensor(i_ids)
