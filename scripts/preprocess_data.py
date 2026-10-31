import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import DAMRSDataset


def download_amazon(dataset_name, data_dir):
    amazon_dir = os.path.join(data_dir, 'Amazon')
    os.makedirs(amazon_dir, exist_ok=True)

    if dataset_name == 'Baby':
        files = {
            'reviews_Baby_5.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Baby_5.json.gz',
            'meta_Baby_old.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Baby.json.gz',
        }
    elif dataset_name == 'Clothing':
        files = {
            'reviews_Clothing_Shoes_and_Jewelry_5_old.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz',
            'meta_Clothing_Shoes_and_Jewelry_old.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz',
        }
    else:
        print(f"Unknown Amazon dataset: {dataset_name}")
        return False

    import urllib.request
    for filename, url in files.items():
        filepath = os.path.join(amazon_dir, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
            print(f"Already downloaded: {filepath} ({os.path.getsize(filepath)} bytes)")
            continue
        print(f"Downloading {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded: {filepath} ({os.path.getsize(filepath)} bytes)")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            print(f"Please manually download from: {url}")
            return False
    return True


def download_mind(data_dir):
    mind_dir = os.path.join(data_dir, 'MIND')
    os.makedirs(mind_dir, exist_ok=True)

    im_mind_zip = os.path.join(mind_dir, 'im_mind_data.zip')
    if os.path.exists(im_mind_zip) and os.path.getsize(im_mind_zip) > 10000:
        print(f"Already downloaded: {im_mind_zip}")
        return True

    print("Downloading im-MIND dataset from Google Drive...")
    print("File ID: 1gx0OzN7qSuyRlvN0cfVUjB4tmoKvvQk1")
    try:
        import subprocess
        result = subprocess.run(
            ['gdown', '1gx0OzN7qSuyRlvN0cfVUjB4tmoKvvQk1',
             '-O', im_mind_zip, '--fuzzy'],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"gdown failed: {result.stderr}")
            print("Please manually download from:")
            print("https://drive.google.com/file/d/1gx0OzN7qSuyRlvN0cfVUjB4tmoKvvQk1/view")
            print(f"And place as: {im_mind_zip}")
            return False
        print(f"Downloaded: {im_mind_zip}")
    except FileNotFoundError:
        print("gdown not installed. Install: pip install gdown")
        print("Then run: gdown 1gx0OzN7qSuyRlvN0cfVUjB4tmoKvvQk1 -O <path> --fuzzy")
        return False

    import zipfile
    print(f"Extracting {im_mind_zip}...")
    with zipfile.ZipFile(im_mind_zip, 'r') as zf:
        zf.extractall(mind_dir)
    print(f"Extracted to {mind_dir}")
    return True


def process_data(dataset_name, data_dir, embed_dim):
    print(f"\nProcessing {dataset_name} dataset...")
    ds_key = dataset_name.lower()
    if ds_key == 'mind':
        mind_inter = os.path.join(data_dir, 'damrs', 'mind', 'mind.inter')
        if not os.path.exists(mind_inter):
            print("MIND is not processed locally. Run: python scripts/preprocess_mind.py 0")
            return
    ds = DAMRSDataset(ds_key, os.path.join(data_dir, 'damrs'), embed_dim)
    print(f"  Users: {ds.n_users}")
    print(f"  Items: {ds.n_items}")
    print(f"  Train interactions: {len(ds.train_data)}")
    print(f"  Val interactions: {len(ds.val_data)}")
    print(f"  Test interactions: {len(ds.test_data)}")
    print(f"  Visual features shape: {ds.visual_features.shape}")
    print(f"  Textual features shape: {ds.textual_features.shape}")
    print(f"  Graph shape: {ds.graph.shape}")
    print(f"Processing complete! Loaded from {os.path.join(data_dir, 'damrs', ds_key)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and preprocess datasets')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['all', 'Baby', 'Clothing', 'MIND'])
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--download_only', action='store_true')
    args = parser.parse_args()

    datasets = ['Baby', 'Clothing', 'MIND'] if args.dataset == 'all' else [args.dataset]

    for ds_name in datasets:
        if ds_name in ['Baby', 'Clothing']:
            success = download_amazon(ds_name, args.data_dir)
        else:
            success = download_mind(args.data_dir)

        if not args.download_only and success:
            process_data(ds_name, args.data_dir, args.embed_dim)
