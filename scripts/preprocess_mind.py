#!/usr/bin/env python3
"""
Preprocess MIND dataset into DA-MRS format for I2MD4Fair.

Steps (run individually or all):
  python scripts/preprocess_mind.py 1   # Step 1: build interactions
  python scripts/preprocess_mind.py 2   # Step 2: extract visual features
  python scripts/preprocess_mind.py 3   # Step 3: extract textual features
  python scripts/preprocess_mind.py 4   # Step 4: compute adj matrices
  python scripts/preprocess_mind.py 0   # Run all steps

Output: data/damrs/mind/
  mind.inter, image_feat.npy, text_feat.npy, image_adj_80.pt, text_adj_80.pt,
  u_id_mapping.csv, i_id_mapping.csv
"""

import os
import sys
import csv
import time
import zipfile
from collections import Counter
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image as PILImage
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIND_DIR = os.path.join(PROJECT_DIR, 'data', 'MIND')
OUT_DIR = os.path.join(PROJECT_DIR, 'data', 'damrs', 'mind')
TOP_K = 80


def _get_torch_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _ensure_extracted(split_name):
    split_dir = os.path.join(MIND_DIR, split_name)
    if os.path.isdir(split_dir):
        return split_dir
    zip_path = os.path.join(MIND_DIR, f'{split_name}.zip')
    if os.path.exists(zip_path):
        print(f"[Extract] {zip_path} -> {split_dir}")
        os.makedirs(split_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(split_dir)
        return split_dir
    raise FileNotFoundError(f"Missing {split_dir} or {zip_path}")


def step1_build_interactions():
    os.makedirs(OUT_DIR, exist_ok=True)
    inter_path = os.path.join(OUT_DIR, 'mind.inter')
    if os.path.exists(inter_path):
        print(f"[Step1] {inter_path} already exists, skipping.")
        return

    print("[Step1] Parsing behaviors.tsv...")
    t0 = time.time()

    train_dir = _ensure_extracted('MINDlarge_train')
    dev_dir = _ensure_extracted('MINDlarge_dev')

    train_news_path = os.path.join(train_dir, 'news.tsv')
    dev_news_path = os.path.join(dev_dir, 'news.tsv')

    all_news = {}
    for npath in [train_news_path, dev_news_path]:
        with open(npath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    nid = parts[0]
                    title = parts[3] if len(parts) >= 4 else ''
                    abstract = parts[4] if len(parts) >= 5 else ''
                    all_news[nid] = {'title': title, 'abstract': abstract}

    train_behav = os.path.join(train_dir, 'behaviors.tsv')
    dev_behav = os.path.join(dev_dir, 'behaviors.tsv')

    interactions = {}
    for bpath in [train_behav, dev_behav]:
        with open(bpath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                uid = parts[1]
                timestamp = parts[2]
                if parts[3]:
                    for nid in parts[3].split():
                        if nid in all_news:
                            interactions[(uid, nid)] = (uid, nid, 1, timestamp)
                if parts[4]:
                    for imp in parts[4].split():
                        nid, clicked = imp.rsplit('-', 1)
                        if clicked == '1' and nid in all_news:
                            interactions[(uid, nid)] = (uid, nid, 1, timestamp)

    interactions = list(interactions.values())

    print(f"  Total interactions: {len(interactions)}")

    user_set = sorted(set(i[0] for i in interactions))
    item_set = sorted(set(i[1] for i in interactions))
    user_map = {u: idx for idx, u in enumerate(user_set)}
    item_map = {i: idx for idx, i in enumerate(item_set)}
    n_users = len(user_map)
    n_items = len(item_map)
    print(f"  n_users={n_users}, n_items={n_items}")

    with open(os.path.join(OUT_DIR, 'u_id_mapping.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['original_id', 'mapped_id'])
        for orig, mapped in user_map.items():
            writer.writerow([orig, mapped])
    with open(os.path.join(OUT_DIR, 'i_id_mapping.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['original_id', 'mapped_id'])
        for orig, mapped in item_map.items():
            writer.writerow([orig, mapped])

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(interactions))
    n_total = len(interactions)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    labels = np.zeros(n_total, dtype=np.int32)
    labels[indices[:n_train]] = 0
    labels[indices[n_train:n_train + n_val]] = 1
    labels[indices[n_train + n_val:]] = 2

    with open(inter_path, 'w') as f:
        f.write('userID\titemID\trating\ttimestamp\tx_label\n')
        for idx, (uid, nid, rating, ts) in enumerate(interactions):
            f.write(f'{user_map[uid]}\t{item_map[nid]}\t{rating}\t{ts}\t{labels[idx]}\n')

    train_cnt = (labels == 0).sum()
    val_cnt = (labels == 1).sum()
    test_cnt = (labels == 2).sum()
    print(f"  Split: train={train_cnt}, val={val_cnt}, test={test_cnt}")
    sparsity = 1.0 - (len(interactions) / max(n_users * n_items, 1))
    print(f"  Local stats: users={n_users}, items={n_items}, interactions={len(interactions)}, sparsity={sparsity * 100:.2f}%")
    print(f"  Time: {time.time()-t0:.1f}s")

    np.save(os.path.join(OUT_DIR, '_item_map.npy'), item_map)
    np.save(os.path.join(OUT_DIR, '_news_data.npy'), all_news)


def print_local_stats():
    inter_path = os.path.join(OUT_DIR, 'mind.inter')
    if not os.path.exists(inter_path):
        print("[Stats] mind.inter not found. Run step 1 first.")
        return
    users = set()
    items = set()
    labels = Counter()
    n = 0
    with open(inter_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            users.add(int(row['userID']))
            items.add(int(row['itemID']))
            labels[int(row['x_label'])] += 1
            n += 1
    sparsity = 1.0 - (n / max(len(users) * len(items), 1))
    print(f"[Stats] users={len(users)}, items={len(items)}, interactions={n}, sparsity={sparsity * 100:.2f}%, split={dict(labels)}")


def step2_extract_visual_features():
    feat_path = os.path.join(OUT_DIR, 'image_feat.npy')
    done_path = os.path.join(OUT_DIR, '_image_feat_done.npy')
    if os.path.exists(feat_path) and os.path.exists(done_path) and np.load(done_path).all():
        print(f"[Step2] {feat_path} already exists, skipping.")
        return

    print("[Step2] Extracting visual features (ResNet-50 avgpool → 2048-dim)...")
    t0 = time.time()

    item_map = np.load(os.path.join(OUT_DIR, '_item_map.npy'), allow_pickle=True).item()
    n_items = len(item_map)

    device = _get_torch_device()
    print(f"  Device: {device}")

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
    feature_extractor.eval()

    img_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if os.path.exists(feat_path):
        feats = np.load(feat_path, mmap_mode='r+')
    else:
        feats = np.lib.format.open_memmap(feat_path, mode='w+', dtype=np.float32, shape=(n_items, 2048))
        feats[:] = 0

    if os.path.exists(done_path):
        done_mask = np.load(done_path)
        if done_mask.shape[0] != n_items:
            done_mask = np.zeros(n_items, dtype=bool)
    else:
        done_mask = np.zeros(n_items, dtype=bool)

    extracted = 0
    failed = 0
    item_list = sorted(item_map.keys(), key=lambda x: item_map[x])

    batch_size = 128 if device.type == 'cuda' else 32 if device.type == 'mps' else 64
    batch_imgs = []
    batch_ids = []
    rng = np.random.RandomState(42)

    def flush_batch():
        nonlocal extracted, batch_imgs, batch_ids
        if not batch_imgs:
            return
        batch = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            out = feature_extractor(batch).squeeze(-1).squeeze(-1).detach().cpu().numpy().astype(np.float32)
        for i, idx in enumerate(batch_ids):
            feats[idx] = out[i]
            done_mask[idx] = True
        extracted += len(batch_ids)
        batch_imgs = []
        batch_ids = []
        if extracted % 512 < batch_size:
            np.save(done_path, done_mask)
            feats.flush()
            print(f"  Extracted {int(done_mask.sum())}/{n_items} done, failed={failed}...")

    for nid in item_list:
        mapped_id = item_map[nid]
        if done_mask[mapped_id]:
            continue
        img_path = os.path.join(MIND_DIR, f'{nid}.jpg')
        if not os.path.exists(img_path):
            feats[mapped_id] = rng.randn(2048).astype(np.float32) * 0.01
            done_mask[mapped_id] = True
            failed += 1
            continue
        try:
            img = PILImage.open(img_path).convert('RGB')
            img_tensor = img_transform(img)
            batch_imgs.append(img_tensor)
            batch_ids.append(mapped_id)
        except Exception:
            feats[mapped_id] = rng.randn(2048).astype(np.float32) * 0.01
            done_mask[mapped_id] = True
            failed += 1
            continue

        if len(batch_imgs) >= batch_size:
            flush_batch()

    flush_batch()

    if not done_mask.all():
        missing = ~done_mask
        feats[missing] = rng.randn(missing.sum(), 2048).astype(np.float32) * 0.01
        done_mask[missing] = True

    feats.flush()
    np.save(done_path, done_mask)

    print(f"  Saved {feat_path}: shape={feats.shape}")
    print(f"  Newly extracted: {extracted}, Failed or missing images: {failed}, Done: {int(done_mask.sum())}/{n_items}")
    print(f"  Time: {time.time()-t0:.1f}s")


def step3_extract_textual_features():
    feat_path = os.path.join(OUT_DIR, 'text_feat.npy')
    if os.path.exists(feat_path):
        print(f"[Step3] {feat_path} already exists, skipping.")
        return

    print("[Step3] Extracting textual features (all-MiniLM-L6-v2 → 384-dim)...")
    t0 = time.time()

    item_map = np.load(os.path.join(OUT_DIR, '_item_map.npy'), allow_pickle=True).item()
    all_news = np.load(os.path.join(OUT_DIR, '_news_data.npy'), allow_pickle=True).item()
    n_items = len(item_map)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    item_list = sorted(item_map.keys(), key=lambda x: item_map[x])
    texts = []
    for nid in item_list:
        if nid in all_news:
            title = all_news[nid].get('title', '') or ''
            abstract = all_news[nid].get('abstract', '') or ''
            text = f"{title}. {abstract}" if title and abstract else (title or abstract or nid)
        else:
            text = nid
        texts.append(text)

    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True,
                              normalize_embeddings=True)

    feats = np.array(embeddings, dtype=np.float32)
    if feats.shape[1] != 384:
        if feats.shape[1] < 384:
            feats = np.pad(feats, ((0, 0), (0, 384 - feats.shape[1])))
        else:
            feats = feats[:, :384]

    np.save(feat_path, feats)
    print(f"  Saved {feat_path}: shape={feats.shape}")
    print(f"  Time: {time.time()-t0:.1f}s")


def step4_compute_adj_matrices():
    img_adj_path = os.path.join(OUT_DIR, 'image_adj_80.pt')
    txt_adj_path = os.path.join(OUT_DIR, 'text_adj_80.pt')

    if os.path.exists(img_adj_path) and os.path.exists(txt_adj_path):
        print(f"[Step4] Adj matrices already exist, skipping.")
        return

    print("[Step4] Computing top-80 similarity adjacency matrices...")
    t0 = time.time()

    image_feats = np.load(os.path.join(OUT_DIR, 'image_feat.npy'))
    text_feats = np.load(os.path.join(OUT_DIR, 'text_feat.npy'))
    n_items = image_feats.shape[0]

    for modality_name, feats, adj_path in [
        ('image', image_feats, img_adj_path),
        ('text', text_feats, txt_adj_path)
    ]:
        print(f"  Computing {modality_name} adj ({n_items} items)...")

        chunk_size = 2000
        n_chunks = (n_items + chunk_size - 1) // chunk_size

        adj_rows = []
        adj_cols = []
        adj_vals = []

        for i in range(n_chunks):
            start_i = i * chunk_size
            end_i = min(start_i + chunk_size, n_items)
            chunk_feats = feats[start_i:end_i]

            sim_chunk = cosine_similarity(chunk_feats, feats)

            for j in range(sim_chunk.shape[0]):
                row_idx = start_i + j
                sims = sim_chunk[j].copy()
                sims[row_idx] = -1.0
                k_eff = min(TOP_K, n_items - 1)
                top_k_indices = np.argpartition(sims, -k_eff)[-k_eff:]
                top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])[::-1]]
                for k_idx in top_k_indices:
                    val = sims[k_idx]
                    if val > 0:
                        adj_rows.append(row_idx)
                        adj_cols.append(k_idx)
                        adj_vals.append(val)

            if (i + 1) % 10 == 0 or i == n_chunks - 1:
                print(f"    Chunk {i+1}/{n_chunks} done")

        indices = torch.LongTensor([adj_rows, adj_cols])
        values = torch.FloatTensor(adj_vals)
        adj = torch.sparse_coo_tensor(indices, values, (n_items, n_items)).coalesce()
        row_sum = torch.sparse.sum(adj, dim=1).to_dense().clamp_min(1e-8)
        values = adj.values() / row_sum[adj.indices()[0]]
        adj = torch.sparse_coo_tensor(adj.indices(), values, adj.shape).coalesce()

        torch.save(adj, adj_path)
        print(f"  Saved {adj_path}")

    print(f"  Time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    os.makedirs(OUT_DIR, exist_ok=True)

    step_funcs = {1: step1_build_interactions, 2: step2_extract_visual_features,
                  3: step3_extract_textual_features, 4: step4_compute_adj_matrices,
                  5: print_local_stats}

    if step == 0:
        for s in [1, 2, 3, 4]:
            step_funcs[s]()
    elif step in step_funcs:
        step_funcs[step]()
    else:
        print(f"Unknown step: {step}. Use 0-5.")
        sys.exit(1)

    print("\nDone. Files in:", OUT_DIR)
