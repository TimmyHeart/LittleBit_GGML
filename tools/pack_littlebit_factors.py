#!/usr/bin/env python3
"""Pack one LittleBit primary path into DEV6.3.2 LB1_0/V-mask tensors.

Input NPZ arrays:
  V_sign: [R, K] real-valued latent factor (quantized by sign; zero -> +1)
  U_sign: [M, R] real-valued output factor (quantized by sign; zero -> +1)
  v2:     [K] input scale
  v1:     [R] first latent scale component
  u2:     [R] second latent scale component
  u1:     [M] output scale

Output NPZ stores the two GGML I32 mask buffers as contiguous rows:
  V_mask_i32: [R, K/32] (GGML logical tensor shape [K/32, R])
  U_mask_i32: [M, ceil(R/32)] (GGML logical tensor shape [ceil(R/32), M])
plus the inference copies g_T=v2, ell_T=round_T(v1*u2), h_T=u1. BF16 scales
are stored as raw uint16 words and described in the sidecar manifest. The source
checkpoint's serialized FP32 scale masters are not replaced: this bundle carries
only the configured inference-dtype copies. Training loaders must preserve the
original FP32 scale masters separately, as in A172.

This is a factor-bundle converter, not a GGUF model writer/loader. It never
modifies the source NPZ. One invocation packs one independent path; residual
paths, when enabled by a later runtime integration, must be packed separately.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def torch_dtype(name: str):
    return {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "f32": torch.float32,
    }[name]


def to_scale_array(t: torch.Tensor, name: str) -> np.ndarray:
    t = t.detach().cpu().contiguous()
    if name == "bf16":
        # Raw IEEE BF16 payload, not float16 numeric conversion.
        return t.view(torch.uint16).numpy().copy()
    if name == "f16":
        return t.to(torch.float16).numpy().copy()
    return t.to(torch.float32).numpy().copy()


def pack_rows(sign: np.ndarray, *, padded_cols: int) -> np.ndarray:
    if sign.ndim != 2:
        raise ValueError("factor sign source must be rank-2")
    rows, cols = sign.shape
    if padded_cols < cols or padded_cols % 32:
        raise ValueError("padded_cols must be >= logical columns and divisible by 32")
    if not np.isfinite(sign).all():
        raise ValueError("factor contains NaN/Inf")
    bits = np.zeros((rows, padded_cols), dtype=np.uint8)
    # Final STE/SmoothSign contract: exact zero is normalized to +1.
    bits[:, :cols] = (sign < 0).astype(np.uint8)
    packed_bytes = np.packbits(bits, axis=1, bitorder="little")
    words_u32 = np.ascontiguousarray(packed_bytes).view("<u4").reshape(rows, padded_cols // 32)
    return words_u32.view("<i4").copy()


def unpack_rows(words_i32: np.ndarray, logical_cols: int) -> np.ndarray:
    u = np.ascontiguousarray(words_i32).view("<u4")
    shifts = np.arange(32, dtype=np.uint32)
    bits = ((u[..., None] >> shifts) & 1).astype(np.int8)
    bits = bits.reshape(u.shape[0], -1)[:, :logical_cols]
    return (1 - 2 * bits).astype(np.int8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_npz", type=Path)
    ap.add_argument("output_npz", type=Path)
    ap.add_argument("--dtype", choices=("f16", "bf16", "f32"), required=True,
                    help="Source activation/scale dtype contract T")
    ap.add_argument("--allow-nonfinite", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    src = args.input_npz.resolve(strict=True)
    dst = args.output_npz.resolve()
    if src == dst:
        raise SystemExit("input and output paths must differ")

    with np.load(src, allow_pickle=False) as z:
        required = ("V_sign", "U_sign", "v2", "v1", "u2", "u1")
        missing = [k for k in required if k not in z]
        if missing:
            raise ValueError(f"missing NPZ arrays: {missing}")
        arrays = {k: np.asarray(z[k]) for k in required}

    V = arrays["V_sign"]
    U = arrays["U_sign"]
    if V.ndim != 2 or U.ndim != 2:
        raise ValueError("V_sign and U_sign must be rank-2")
    R, K = V.shape
    M, R_u = U.shape
    if not (K > 0 and K % 32 == 0 and R > 0 and M > 0 and R_u == R):
        raise ValueError(f"incompatible shapes V={V.shape}, U={U.shape}; require V=[R,K], U=[M,R], K%32=0")

    expected = {"v2": (K,), "v1": (R,), "u2": (R,), "u1": (M,)}
    for k, shape in expected.items():
        if arrays[k].shape != shape:
            raise ValueError(f"{k} shape {arrays[k].shape} != {shape}")
        if not np.isfinite(arrays[k]).all():
            raise ValueError(f"{k} contains NaN/Inf")

    words_v = K // 32
    words_u = (R + 31) // 32
    V_mask = pack_rows(V, padded_cols=K)
    U_mask = pack_rows(U, padded_cols=words_u * 32)

    # Emulate the official forward's dtype boundary before folding v1*u2.
    T = torch_dtype(args.dtype)
    def as_T(key: str) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arrays[key])).to(dtype=T).contiguous()
    g_T = as_T("v2")
    v1_T = as_T("v1")
    u2_T = as_T("u2")
    h_T = as_T("u1")
    ell_T = (v1_T * u2_T).to(dtype=T).contiguous()

    if not np.array_equal(unpack_rows(V_mask, K), np.where(V < 0, -1, 1).astype(np.int8)):
        raise AssertionError("V pack/unpack roundtrip failed")
    if not np.array_equal(unpack_rows(U_mask, R), np.where(U < 0, -1, 1).astype(np.int8)):
        raise AssertionError("U pack/unpack roundtrip failed")
    if R % 32 and np.any((U_mask[:, -1].view("<u4") >> np.uint32(R % 32)) != 0):
        raise AssertionError("U tail padding is not all positive-sign zero bits")

    out_arrays = {
        "V_mask_i32": np.ascontiguousarray(V_mask),
        "U_mask_i32": np.ascontiguousarray(U_mask),
        "g_T": to_scale_array(g_T, args.dtype),
        "ell_T": to_scale_array(ell_T, args.dtype),
        "h_T": to_scale_array(h_T, args.dtype),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out_arrays)
    manifest = {
        "schema": "DEV6_3_2_LITTLEBIT_FACTOR_BUNDLE_V1",
        "source_npz": str(src),
        "source_sha256": sha256(src),
        "output_npz": str(dst.resolve()),
        "output_sha256": sha256(dst),
        "activation_dtype_T": args.dtype,
        "factor_bit_order": "LSB_FIRST",
        "bit_semantics": {"0": "+1", "1": "-1", "exact_zero_factor": "+1"},
        "V_logical_shape_RK": [R, K],
        "V_mask_tensor_ne0_ne1": [words_v, R],
        "V_mask_storage_rows_words": [R, words_v],
        "U_logical_shape_MR": [M, R],
        "U_mask_tensor_ne0_ne1": [words_u, M],
        "U_mask_storage_rows_words": [M, words_u],
        "logical_rank": R,
        "physical_rank": words_u * 32,
        "U_tail_padding_bits": words_u * 32 - R,
        "U_tail_padding_bits_all_zero": True,
        "scale_contract": "inference copies: g_T=v2_T; ell_T=round_T(v1_T*u2_T); h_T=u1_T",
        "scale_storage": "raw uint16 BF16 words" if args.dtype == "bf16" else args.dtype,
        "scale_master_policy": "serialized FP32 masters are not included in this inference bundle; preserve them separately for training/rehydration",
        "raw_fp32_scale_masters_included": False,
        "primary_path_only": True,
        "gguf_model_writer_included": False,
        "cuda_device_validation": False,
    }
    sidecar = dst.with_suffix(dst.suffix + ".json")
    sidecar.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in (
        "schema", "output_npz", "output_sha256", "activation_dtype_T",
        "V_mask_tensor_ne0_ne1", "U_mask_tensor_ne0_ne1", "physical_rank",
        "U_tail_padding_bits_all_zero", "scale_contract", "gguf_model_writer_included"
    )}, indent=2))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)
