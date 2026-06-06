#!/usr/bin/env python3
"""Apply NumPy 2.0 compatibility patches to an installed MXNet 1.9.1 package.

MXNet 1.9.1 (last release 2022) references NumPy symbols removed in NumPy 2.0.
This script patches the four affected files inside the active venv in-place.
Run once after `pip install -r requirements.txt` (bootstrap_venv.sh does this).

Safe to re-run: each patch is idempotent.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _find_mxnet_root() -> Path:
    spec = importlib.util.find_spec("mxnet")
    if spec is None:
        print("[fix_mxnet] mxnet not installed — nothing to do")
        sys.exit(0)
    return Path(spec.origin).parent


def _patch(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        print(f"[fix_mxnet] {label}: already patched or not found — skip")
        return
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"[fix_mxnet] {label}: patched")


def main() -> None:
    root = _find_mxnet_root()
    print(f"[fix_mxnet] mxnet root: {root}")

    # ------------------------------------------------------------------ #
    # 1. numpy/utils.py — removed constants and np.bool                   #
    # ------------------------------------------------------------------ #
    utils = root / "numpy" / "utils.py"
    _patch(
        utils,
        "bool = onp.bool\n",
        "bool = onp.bool_  # np.bool removed in NumPy 2.0\n",
        "numpy/utils.py :: np.bool",
    )
    for old, new, label in [
        ("PZERO = onp.PZERO", "PZERO = 0.0        # np.PZERO removed in NumPy 2.0", "np.PZERO"),
        ("NZERO = onp.NZERO", "NZERO = -0.0       # np.NZERO removed in NumPy 2.0", "np.NZERO"),
        ("NINF = onp.NINF",   "NINF = float('-inf')  # np.NINF removed in NumPy 2.0",  "np.NINF"),
        ("PINF = onp.PINF",   "PINF = float('inf')   # np.PINF removed in NumPy 2.0",  "np.PINF"),
        ("NAN = onp.NAN",     "NAN = float('nan')    # np.NAN removed in NumPy 2.0",   "np.NAN"),
        ("NaN = onp.NaN",     "NaN = float('nan')    # np.NaN removed in NumPy 2.0",   "np.NaN"),
    ]:
        _patch(utils, old, new, f"numpy/utils.py :: {label}")

    # ------------------------------------------------------------------ #
    # 2. numpy/fallback.py — removed functions                            #
    # ------------------------------------------------------------------ #
    fallback = root / "numpy" / "fallback.py"
    _patch(fallback, "alltrue = onp.alltrue\n",
           "alltrue = onp.all  # np.alltrue removed in NumPy 2.0\n",
           "numpy/fallback.py :: alltrue")
    _patch(fallback, "in1d = onp.in1d\n",
           "in1d = onp.isin  # np.in1d removed in NumPy 2.0\n",
           "numpy/fallback.py :: in1d")
    _patch(fallback, "mirr = onp.mirr\n",
           "def mirr(*a, **k): raise NotImplementedError('np.mirr removed in NumPy 2.0')\n",
           "numpy/fallback.py :: mirr")
    _patch(fallback, "msort = onp.msort\n",
           "def msort(a): return onp.sort(a, axis=0)  # np.msort removed in NumPy 2.0\n",
           "numpy/fallback.py :: msort")
    _patch(fallback, "npv = onp.npv\n",
           "def npv(*a, **k): raise NotImplementedError('np.npv removed in NumPy 2.0')\n",
           "numpy/fallback.py :: npv")
    _patch(fallback, "pmt = onp.pmt\n",
           "def pmt(*a, **k): raise NotImplementedError('np.pmt removed in NumPy 2.0')\n",
           "numpy/fallback.py :: pmt")
    _patch(fallback, "ppmt = onp.ppmt\n",
           "def ppmt(*a, **k): raise NotImplementedError('np.ppmt removed in NumPy 2.0')\n",
           "numpy/fallback.py :: ppmt")
    _patch(fallback, "pv = onp.pv\n",
           "def pv(*a, **k): raise NotImplementedError('np.pv removed in NumPy 2.0')\n",
           "numpy/fallback.py :: pv")
    _patch(fallback, "rate = onp.rate\n",
           "def rate(*a, **k): raise NotImplementedError('np.rate removed in NumPy 2.0')\n",
           "numpy/fallback.py :: rate")
    _patch(fallback, "trapz = onp.trapz\n",
           "trapz = getattr(onp, 'trapezoid', None) or getattr(onp, 'trapz', None)"
           "  # renamed in NumPy 2.0\n",
           "numpy/fallback.py :: trapz")

    # ------------------------------------------------------------------ #
    # 3. gluon/contrib/estimator/event_handler.py — np.Inf               #
    # ------------------------------------------------------------------ #
    handler = root / "gluon" / "contrib" / "estimator" / "event_handler.py"
    text = handler.read_text(encoding="utf-8")
    if "np.Inf" in text:
        handler.write_text(text.replace("np.Inf", "np.inf"), encoding="utf-8")
        print("[fix_mxnet] event_handler.py :: np.Inf → np.inf")
    else:
        print("[fix_mxnet] event_handler.py :: already patched — skip")

    # ------------------------------------------------------------------ #
    # 4. numpy_dispatch_protocol.py — graceful skip for removed ops       #
    # ------------------------------------------------------------------ #
    proto = root / "numpy_dispatch_protocol.py"
    OLD = (
        "    for op_name in _NUMPY_ARRAY_FUNCTION_LIST:\n"
        "        strs = op_name.split('.')\n"
        "        if len(strs) == 1:\n"
        "            mx_np_op = getattr(mx_np, op_name)\n"
        "            onp_op = getattr(_np, op_name)\n"
        "            setattr(mx_np, op_name, _implements(onp_op)(mx_np_op))\n"
        "        elif len(strs) == 2:\n"
        "            mx_np_submodule = getattr(mx_np, strs[0])\n"
        "            mx_np_op = getattr(mx_np_submodule, strs[1])\n"
        "            onp_submodule = getattr(_np, strs[0])\n"
        "            onp_op = getattr(onp_submodule, strs[1])\n"
        "            setattr(mx_np_submodule, strs[1], _implements(onp_op)(mx_np_op))\n"
        "        else:\n"
        "            raise ValueError('Does not support registering __array_function__ protocol '\n"
        "                             'for operator {}'.format(op_name))\n"
    )
    NEW = (
        "    for op_name in _NUMPY_ARRAY_FUNCTION_LIST:\n"
        "        strs = op_name.split('.')\n"
        "        if len(strs) == 1:\n"
        "            mx_np_op = getattr(mx_np, op_name, None)\n"
        "            onp_op = getattr(_np, op_name, None)\n"
        "            if mx_np_op is None or onp_op is None:\n"
        "                continue  # skip ops removed in NumPy 2.0\n"
        "            setattr(mx_np, op_name, _implements(onp_op)(mx_np_op))\n"
        "        elif len(strs) == 2:\n"
        "            mx_np_submodule = getattr(mx_np, strs[0], None)\n"
        "            onp_submodule = getattr(_np, strs[0], None)\n"
        "            if mx_np_submodule is None or onp_submodule is None:\n"
        "                continue\n"
        "            mx_np_op = getattr(mx_np_submodule, strs[1], None)\n"
        "            onp_op = getattr(onp_submodule, strs[1], None)\n"
        "            if mx_np_op is None or onp_op is None:\n"
        "                continue\n"
        "            setattr(mx_np_submodule, strs[1], _implements(onp_op)(mx_np_op))\n"
        "        else:\n"
        "            raise ValueError('Does not support registering __array_function__ protocol '\n"
        "                             'for operator {}'.format(op_name))\n"
    )
    _patch(proto, OLD, NEW, "numpy_dispatch_protocol.py :: graceful skip")

    # ------------------------------------------------------------------ #
    # 5. Verify                                                            #
    # ------------------------------------------------------------------ #
    print("\n[fix_mxnet] verifying import...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c", "import mxnet; print('mxnet import OK')"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"[fix_mxnet] {result.stdout.strip()}")
    else:
        print(f"[fix_mxnet] FAILED:\n{result.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
