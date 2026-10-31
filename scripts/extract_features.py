import os
import sys
import gzip
import ast
import json
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image as PILImage
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_amazon_visual_features(dataset_name, data_dir='data/', embed_dim=64):
    amazon_dir = os.path.join(data_dir, 'Amazon')
    image_dir = os.path.join(amazon_dir, f'{dataset_name}_images')
    os.makedirs(image_dir, exist_ok=True)

    processed_dir = os.path.join(data_dir, 'processed')
    inter_path = os.path.join(processed_dir, f'{dataset_name}_inter.npy')
    split_path = os.path.join(processed_dir, f'{dataset_name}_split.npz')

    if not os.path.exists(inter_path) or not os.path.exists(split_path):
        print(f"Processed data for {dataset_name} not found. Run dataset loading first.")
        return

    split_data = np.load(split_path)
    n_items = int(split_data['n_items'])

    meta_candidates = []
    if dataset_name == 'Baby':
        meta_candidates = [
            os.path.join(amazon_dir, 'meta_Baby_old.json.gz'),
            os.path.join(amazon_dir, 'meta_Baby.json.gz'),
        ]
    elif dataset_name == 'Clothing':
        meta_candidates = [
            os.path.join(amazon_dir, 'meta_Clothing_Shoes_and_Jewelry_old.json.gz'),
            os.path.join(amazon_dir, 'meta_Clothing_Shoes_and_Jewelry.json.gz'),
        ]

    meta_file = None
    for candidate in meta_candidates:
        if os.path.exists(candidate) and os.path.getsize(candidate) > 10000:
            meta_file = candidate
            break

    if meta_file is None:
        print(f"Metadata file not found for {dataset_name}")
        return

    asin_to_idx = {}
    reviews_candidates = []
    if dataset_name == 'Baby':
        reviews_candidates = [os.path.join(amazon_dir, 'reviews_Baby_5.json.gz')]
    elif dataset_name == 'Clothing':
        reviews_candidates = [
            os.path.join(amazon_dir, 'reviews_Clothing_Shoes_and_Jewelry_5_old.json.gz'),
            os.path.join(amazon_dir, 'reviews_Clothing_Shoes_and_Jewelry_5.json.gz'),
        ]

    review_file = None
    for candidate in reviews_candidates:
        if os.path.exists(candidate) and os.path.getsize(candidate) > 10000:
            review_file = candidate
            break

    if review_file is None:
        print(f"Review file not found for {dataset_name}")
        return

    item_cnt = 0
    with gzip.open(review_file, 'rt') as f:
        for line in f:
            try:
                review = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            asin = review.get('asin', '')
            if not asin:
                continue
            if asin not in asin_to_idx:
                asin_to_idx[asin] = item_cnt
                item_cnt += 1

    image_urls = {}
    with gzip.open(meta_file, 'rt') as f:
        for line in f:
            try:
                meta = json.loads(line.strip())
            except json.JSONDecodeError:
                try:
                    meta = ast.literal_eval(line.strip())
                except:
                    continue
            asin = meta.get('asin', '')
            if asin not in asin_to_idx:
                continue
            url = ''
            if 'imUrl' in meta and meta['imUrl']:
                url = meta['imUrl']
            elif 'imageURL' in meta and meta['imageURL']:
                url = meta['imageURL']
            elif 'imageURLHighRes' in meta and meta['imageURLHighRes']:
                urls = meta['imageURLHighRes']
                if isinstance(urls, list) and len(urls) > 0:
                    url = urls[0]
            if url:
                image_urls[asin] = url

    print(f"{dataset_name}: {len(asin_to_idx)} items, {len(image_urls)} with image URLs")

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    img_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    downloaded = 0
    extracted = 0
    all_feats = []
    extracted_asins = []
    failed_downloads = 0

    for asin, url in image_urls.items():
        idx = asin_to_idx[asin]
        img_path = os.path.join(image_dir, f'{asin}.jpg')
        if not os.path.exists(img_path):
            try:
                import urllib.request
                urllib.request.urlretrieve(url, img_path)
                downloaded += 1
            except Exception:
                failed_downloads += 1
                continue
        if os.path.exists(img_path):
            try:
                img = PILImage.open(img_path).convert('RGB')
                img_tensor = img_transform(img).unsqueeze(0)
                with torch.no_grad():
                    feat = resnet(img_tensor).squeeze().numpy()
                all_feats.append(feat)
                extracted_asins.append(asin)
                extracted += 1
            except Exception:
                continue
        if extracted % 500 == 0:
            print(f"  Extracted {extracted}/{len(image_urls)} features...")

    print(f"Extracted {extracted} features, downloaded {downloaded} new, failed {failed_downloads}")

    visual_feats = np.zeros((n_items, embed_dim), dtype=np.float32)
    if extracted > 0:
        feat_matrix = np.array(all_feats)
        n_components = min(embed_dim, feat_matrix.shape[0], feat_matrix.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(feat_matrix)
        if reduced.shape[1] < embed_dim:
            reduced = np.pad(reduced, ((0, 0), (0, embed_dim - reduced.shape[1])))
        for i, asin in enumerate(extracted_asins):
            idx = asin_to_idx[asin]
            visual_feats[idx] = reduced[i].astype(np.float32)
        zero_mask = visual_feats.sum(axis=1) == 0
        if zero_mask.any():
            rng = np.random.RandomState(42)
            visual_feats[zero_mask] = rng.randn(zero_mask.sum(), embed_dim).astype(np.float32) * 0.01

    vis_feat_path = os.path.join(processed_dir, f'{dataset_name}_visual_features.npz')
    np.savez(vis_feat_path, features=visual_feats)
    vis_path = os.path.join(processed_dir, f'{dataset_name}_visual.npy')
    np.save(vis_path, visual_feats)
    print(f"Saved visual features: {visual_feats.shape} to {vis_feat_path}")
    print(f"Nonzero rows: {(visual_feats.sum(axis=1) != 0).sum()}/{n_items}")


def extract_mind_visual_features(data_dir='data/', embed_dim=64):
    mind_dir = os.path.join(data_dir, 'MIND')
    processed_dir = os.path.join(data_dir, 'processed')
    split_path = os.path.join(processed_dir, 'MIND_split.npz')

    if not os.path.exists(split_path):
        print("MIND processed data not found. Run dataset loading first.")
        return

    split_data = np.load(split_path)
    n_items = int(split_data['n_items'])

    image_list_path = os.path.join(data_dir, 'IMRec_repo', 'imageList.npy')
    if os.path.exists(image_list_path):
        image_list = np.load(image_list_path, allow_pickle=True)
        valid_news_ids = set(image_list)
    else:
        valid_news_ids = set()

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    img_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    item_map = {}
    train_news = os.path.join(mind_dir, 'MINDlarge_train', 'news.tsv')
    dev_news = os.path.join(mind_dir, 'MINDlarge_dev', 'news.tsv')
    for news_path in [train_news, dev_news]:
        if os.path.exists(news_path):
            with open(news_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 1:
                        nid = parts[0]
                        item_map[nid] = len(item_map)

    extracted = 0
    all_feats = []
    extracted_ids = []

    for nid in valid_news_ids:
        if nid not in item_map:
            continue
        img_path = os.path.join(mind_dir, f'{nid}.jpg')
        if os.path.exists(img_path):
            try:
                img = PILImage.open(img_path).convert('RGB')
                img_tensor = img_transform(img).unsqueeze(0)
                with torch.no_grad():
                    feat = resnet(img_tensor).squeeze().numpy()
                all_feats.append(feat)
                extracted_ids.append(nid)
                extracted += 1
            except Exception:
                continue
        if extracted % 1000 == 0:
            print(f"  Extracted {extracted} MIND features...")

    print(f"Extracted {extracted} MIND visual features")

    visual_feats = np.zeros((n_items, embed_dim), dtype=np.float32)
    if extracted > 0:
        feat_matrix = np.array(all_feats)
        n_components = min(embed_dim, feat_matrix.shape[0], feat_matrix.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(feat_matrix)
        if reduced.shape[1] < embed_dim:
            reduced = np.pad(reduced, ((0, 0), (0, embed_dim - reduced.shape[1])))
        for i, nid in enumerate(extracted_ids):
            idx = item_map[nid]
            if idx < n_items:
                visual_feats[idx] = reduced[i].astype(np.float32)
        zero_mask = visual_feats.sum(axis=1) == 0
        if zero_mask.any():
            rng = np.random.RandomState(42)
            visual_feats[zero_mask] = rng.randn(zero_mask.sum(), embed_dim).astype(np.float32) * 0.01

    vis_feat_path = os.path.join(processed_dir, 'MIND_visual_features.npz')
    np.savez(vis_feat_path, features=visual_feats)
    vis_path = os.path.join(processed_dir, 'MIND_visual.npy')
    np.save(vis_path, visual_feats)
    print(f"Saved MIND visual features: {visual_feats.shape}")
    print(f"Nonzero rows: {(visual_feats.sum(axis=1) != 0).sum()}/{n_items}")


if __name__ == '__main__':
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'Baby'
    data_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/'
    embed_dim = int(sys.argv[3]) if len(sys.argv) > 3 else 64

    if dataset == 'MIND':
        extract_mind_visual_features(data_dir, embed_dim)
    elif dataset in ['Baby', 'Clothing']:
        extract_amazon_visual_features(dataset, data_dir, embed_dim)
    else:
        print(f"Unknown dataset: {dataset}")
