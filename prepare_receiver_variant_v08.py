"""Prepare a distinct, architecture-compatible receiver checkpoint for v0.8 tests.

A controlled perturbation yields a different checkpoint identity, NOT an
independently trained or genuinely heterogeneous LLM. Preserve original.
"""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="vendor/LLM_SEM/model/model-gpu-v0.4.pt")
    p.add_argument("--output",default="vendor/LLM_SEM/model/model-gpu-v0.4-variant.pt")
    p.add_argument("--seed",type=int,default=20261021)
    p.add_argument("--sigma",type=float,default=0.0001)
    a=p.parse_args()
    if not 0 < a.sigma <= 0.001:
        p.error("sigma must be in (0,0.001]")
    src,dst=Path(a.source),Path(a.output)
    if not src.is_file(): p.error("source checkpoint not found")
    if dst.exists(): p.error("output already exists; refusing overwrite")
    if src.resolve()==dst.resolve(): p.error("source and output must differ")
    import torch
    from semantic_decoder import load_encoder, digest
    model, tokenizer=load_encoder(src,"vendor/LLM_SEM/model/tokenizer.json",torch.device("cpu"))
    # Use the checkpoint's standard serializer so load_checkpoint remains compatible.
    checkpoint=torch.load(src,map_location="cpu",weights_only=False)
    if not isinstance(checkpoint,dict):
        p.error("unsupported checkpoint structure")
    # Locate state dictionary in known checkpoint containers.
    key=next((k for k in ("model_state_dict","state_dict","model") if isinstance(checkpoint.get(k),dict)),None)
    state=checkpoint[key] if key else checkpoint
    candidates=[(k,v) for k,v in state.items() if isinstance(v,torch.Tensor) and v.is_floating_point() and v.numel()>0]
    if not candidates: p.error("no floating model tensors")
    name,tensor=candidates[0]
    gen=torch.Generator(device="cpu").manual_seed(a.seed)
    modified=tensor.detach().clone()
    modified.add_(torch.randn(modified.shape,generator=gen,dtype=modified.dtype)*a.sigma)
    state[name]=modified
    dst.parent.mkdir(parents=True,exist_ok=True)
    torch.save(checkpoint,dst)
    try:
        other,_=load_encoder(dst,"vendor/LLM_SEM/model/tokenizer.json",torch.device("cpu"))
    except Exception:
        dst.unlink(missing_ok=True)
        raise
    print(json.dumps({"status":"variant-created","purpose":"checkpoint-distinction-smoke-test-only",
      "changed_tensor":name,"source_sha256":digest(src),"variant_sha256":digest(dst),
      "source_d_model":model.d_model,"variant_d_model":other.d_model,
      "path":str(dst)},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
