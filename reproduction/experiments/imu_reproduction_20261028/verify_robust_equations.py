"""Regression checks for the equation-audited RobustHAR fusion (not calibration)."""
import json
import torch
from robusthar import RobustHAR
from prepare import HERE, atomic_json, sha


def check(dataset, length, channels, classes):
    torch.manual_seed(42)
    model = RobustHAR(dataset, length, channels, classes).eval()
    raw = torch.randn(2, length, channels)
    with torch.no_grad():
        features = [branch.encode(model.layout(raw).flatten(2)) for branch in model.branches]
        expected = []
        for layers, (i, j) in zip(model.fusion, [(0, 1), (0, 2), (1, 2)]):
            a, b = features[i], features[j]
            # Independent direct transcription of Eqs.9-11.
            a_scaled = a / torch.linalg.vector_norm(b, dim=1, keepdim=True).clamp_min(1e-6)
            b_scaled = b / torch.linalg.vector_norm(a, dim=1, keepdim=True).clamp_min(1e-6)
            r_a = torch.bmm(a_scaled.transpose(1, 2), a_scaled)
            r_b = torch.bmm(b_scaled.transpose(1, 2), b_scaled)
            a_att = layers[0](r_a, a, a, need_weights=False)[0]
            b_att = layers[1](r_b, b, b, need_weights=False)[0]
            expected.append(torch.bmm(a_att, b_att).mean(dim=1))
        expected = model.classifier(sum(expected) / 3)
    seen = []
    handles = []
    for layers in model.fusion:
        assert layers[0] is not layers[1]
        for layer in layers:
            handles.append(layer.register_forward_pre_hook(
                lambda module, args: seen.append(tuple(tuple(x.shape) for x in args[:3]))))
    actual = model(raw)
    for h in handles:
        h.remove()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert seen == [((2, 128, 128), (2, length, 128), (2, length, 128))] * 6, seen
    loss = model.pretrain_loss(raw)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    model.freeze_encoders()
    model.train()
    assert all(not b.training for b in model.branches)
    torch.nn.functional.cross_entropy(model(raw), torch.tensor([0, classes - 1])).backward()
    assert all(p.grad is None for b in model.branches for p in b.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for layers in model.fusion for layer in layers for p in layer.parameters())
    assert model.classifier.weight.grad is not None
    return {'dataset': dataset, 'passed': True, 'attention_queries': 'correlation_B,128,128',
            'keys_values': 'same_branch_B,T,128', 'fusion': 'matrix_dot_product',
            'frozen_encoders_verified': True}


if __name__ == '__main__':
    torch.set_num_threads(1)
    cases = [('uci_har',128,9,6), ('wisdm',200,3,6), ('pamap2',200,27,12),
             ('hhar',100,6,6), ('opportunity_a',30,113,17)]
    result = {'model_sha256': sha(HERE/'robusthar.py'),
              'cases': [check(*case) for case in cases]}
    atomic_json(HERE/'robusthar_equation_verification.json', result)
    print(json.dumps(result, indent=2))
