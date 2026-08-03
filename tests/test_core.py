import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.i2md4fair import CLUBEstimator, InfoNCELoss, I2MD4Fair


def test_club_all_pairs():
    torch.manual_seed(42)
    B, D, X = 8, 4, 16
    club = CLUBEstimator(X, D)
    x = torch.randn(B, X)
    z = torch.randn(B, D, requires_grad=True)

    mi_chunked = club.mi_upper_bound(x, z)

    with torch.no_grad():
        mu, logvar = club.forward(x)
    log_pos = club.log_prob(z, mu, logvar).sum(dim=-1)
    log_neg_sum = torch.tensor(0.0, device=z.device)
    for i in range(B):
        for j in range(B):
            if i == j:
                continue
            lp = club.log_prob(z[j:j+1], mu[i:i+1], logvar[i:i+1]).sum(dim=-1)
            log_neg_sum = log_neg_sum + lp
    mi_brute = (log_pos.mean() - log_neg_sum / (B * (B - 1))).clamp(min=0)
    assert torch.allclose(mi_chunked, mi_brute, atol=1e-5), f"chunked={mi_chunked}, brute={mi_brute}"
    print("PASS test_club_all_pairs")


def test_club_gradient_separation():
    torch.manual_seed(42)
    B, D, X = 16, 8, 32
    club = CLUBEstimator(X, D)
    x = torch.randn(B, X)
    z = nn.Parameter(torch.randn(B, D))

    mi = club.mi_upper_bound(x, z)
    mi.backward()
    assert z.grad is not None and z.grad.abs().sum() > 0, "z must have nonzero gradient"
    for p in club.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, "CLUB params must have no gradient in recommender step"

    club.zero_grad()
    z2 = z.detach().requires_grad_(False)
    nll = club.nll_loss(x, z2)
    nll.backward()
    for p in club.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, "CLUB params must have gradient in CLUB step"
    print("PASS test_club_gradient_separation")


def test_nll_scale():
    torch.manual_seed(42)
    B, D, X = 8, 4, 16
    club = CLUBEstimator(X, D)
    x = torch.randn(B, X)
    z = torch.randn(B, D)
    nll = club.nll_loss(x, z)
    mu, logvar = club.forward(x)
    manual = -club.log_prob(z, mu, logvar).sum(dim=-1).mean()
    assert torch.allclose(nll, manual, atol=1e-6), f"nll={nll}, manual={manual}"
    print("PASS test_nll_scale")


def test_chunked_infonce():
    torch.manual_seed(42)
    B, D = 16, 8
    chunked = InfoNCELoss(tau=0.01, chunk_size=4)
    full = InfoNCELoss(tau=0.01, chunk_size=512)
    embs = {'visual': torch.randn(B, D), 'textual': torch.randn(B, D)}
    loss_chunked = chunked(embs)
    loss_full = full(embs)
    assert torch.allclose(loss_chunked, loss_full, atol=1e-5), f"chunked={loss_chunked}, full={loss_full}"
    print("PASS test_chunked_infonce")


def test_ablation_gcn_vs_hgcn():
    from models.i2md4fair import HypergraphConv, PairwiseGCN
    torch.manual_seed(42)
    B, D, T = 10, 8, 4
    Z = torch.randn(B, D)
    H = torch.rand(B, T)
    gcn = PairwiseGCN(D)
    hgcn = HypergraphConv(D)
    out_gcn = gcn(Z, H)
    out_hgcn = hgcn(Z, H)
    assert not torch.allclose(out_gcn, out_hgcn), "GCN and HGCN must produce different outputs"
    print("PASS test_ablation_gcn_vs_hgcn")


def test_lightgcn_m_uses_modality():
    from scripts.run_ablation import AblationModel
    model = AblationModel(10, 20, 64, {'visual': 32, 'textual': 32}, use_lightgcn_m=True)
    assert hasattr(model, 'modality_encoders'), "LightGCN+M must have modality encoders"
    assert 'visual' in model.modality_encoders and 'textual' in model.modality_encoders
    print("PASS test_lightgcn_m_uses_modality")


if __name__ == '__main__':
    test_club_all_pairs()
    test_club_gradient_separation()
    test_nll_scale()
    test_chunked_infonce()
    test_ablation_gcn_vs_hgcn()
    test_lightgcn_m_uses_modality()
    print("\nAll tests passed!")
