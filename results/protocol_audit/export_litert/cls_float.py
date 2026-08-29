"""litert-torch float export of a plain classification .pt. No monkeypatches."""
import sys
from pathlib import Path
import torch, litert_torch
pt, size, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
m = torch.load(str(pt), map_location="cpu", weights_only=False)
m = (m.module if hasattr(m, "module") and hasattr(m.module, "parameters") else m).eval()
print(f"[load] {pt.name} {type(m).__name__} params={sum(p.numel() for p in m.parameters())}")
net = litert_torch.to_channel_last_io(m, args=[0]).eval()   # 2-D logits out -> no outputs=
litert_torch.convert(net, (torch.randn(1, size, size, 3),)).export(str(out))
print(f"[float] {out.name} {out.stat().st_size/2**20:.3f} MiB")
