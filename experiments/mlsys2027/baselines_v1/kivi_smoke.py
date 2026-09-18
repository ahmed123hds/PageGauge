"""Native INT2/INT4 KIVI GQA GEMV correctness against reconstructed operands.

No model-quality or timing claim. Run with kivi_sm120_env, on an idle GPU.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('/home/anonymous/pagegauge_baselines/KIVI')
EXPECTED = '876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6'
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base


def main():
    import fcntl
    if subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip() != EXPECTED:
        raise RuntimeError('Baseline revision drift')
    if subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise RuntimeError('Undisclosed native source changes')
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        sys.path.insert(0, str(SOURCE))
        import torch
        import kivi_gemv
        from quant.new_pack import triton_quantize_and_pack_along_last_dim, unpack_tensor
        from quant.matmul import cuda_bmm_fA_qB_outer
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(2026090812)
        out = ROOT/'results/mlsys2027_baselines_v1'/('kivi_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        paths = [Path(__file__), Path(kivi_gemv.__file__), SOURCE/'quant/new_pack.py', SOURCE/'quant/matmul.py',
                 SOURCE/'quant/csrc/gemv_cuda.cu', SOURCE/'quant/csrc/pybind.cpp']
        hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        base.atomic_json(out/'manifest.json', {'commit':EXPECTED,'source_sha256':hashes,'idle':idle,
            'cases':{'batch':[1,4], 'bits':[2,4], 'operation':['QK','PV'], 'hq':32,'hkv':8,'context':256,'head_dim':128,'group':32},
            'execution_numerics':{'maximum_relative_l2':.005,'maximum_absolute_error':.03},
            'scope':'Synthetic native GQA GEMV correctness only; not timing or quantization-quality evaluation'})
        print('KIVI native smoke: '+str(out), flush=True)
        rows = []
        for batch in (1,4):
            for bits in (2,4):
                for operation in ('QK','PV'):
                    inner, outer = (128,256) if operation == 'QK' else (256,128)
                    operand = torch.randn(batch,8,inner,outer,device='cuda',dtype=torch.float16)
                    left = torch.randn(batch,32,1,inner,device='cuda',dtype=torch.float16)
                    if operation == 'PV':left = left.float().softmax(-1).half()
                    codes,scales,mins = triton_quantize_and_pack_along_last_dim(operand,32,bits)
                    reconstructed = (unpack_tensor(codes,bits,3).float()*scales.float().repeat_interleave(32,-1)
                                     + mins.float().repeat_interleave(32,-1)).repeat_interleave(4,1)
                    reference = torch.matmul(left.float(),reconstructed)
                    actual = cuda_bmm_fA_qB_outer(32,left,codes,scales,mins,bits)
                    torch.cuda.synchronize()
                    diff = actual.float()-reference
                    l2 = float(diff.norm()/reference.norm().clamp_min(1e-12))
                    maximum = float(diff.abs().max())
                    finite = bool(torch.isfinite(actual).all())
                    row = {'batch':batch,'bits':bits,'operation':operation,'relative_l2':l2,'max_abs':maximum,
                           'finite':finite,'passed':finite and l2 <= .005 and maximum <= .03,
                           'operand_storage_bytes':sum(t.numel()*t.element_size() for t in (codes,scales,mins))}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha for path,sha in hashes.items()):
            raise RuntimeError('Source drift')
        result = {'passed':all(r['passed'] for r in rows), 'rows':rows,
            'kernel_sha256':hashes[kivi_gemv.__file__], 'gpu':torch.cuda.get_device_name(),
            'torch':torch.__version__, 'scope':'Native kernel execution numerics only; full-model compatibility not established'}
        base.atomic_json(out/'analysis.json',result)
        if not result['passed']:raise SystemExit(2)


if __name__ == '__main__':main()
