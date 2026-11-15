# SPDX-License-Identifier: Apache-2.0
"""Flip a kt KERNEL knob inside the RUNNING server, for paired A/B measurement.

Why this exists
---------------
kt reads its MXFP4 prefetch policy from the environment **once**, so two
policies would mean two server processes -- and on this host a cross-process
comparison of the same code has differed 15x (``measure/HANDOFF.md``). Every
number this project keeps comes from arms alternating inside one process, or at
least inside one card session with the order flipped.

The knob itself is settable at runtime (``kt_kernel_ext.set_nz_prefetch_groups``
/ ``set_nz_prefetch_legacy_stride``), so one server process can hold both arms
-- provided something inside the process applies them. This is that something: a
watcher thread that reads one small file and applies what it says.

🔴 This is measurement scaffolding, not a feature.
  * It does nothing unless ``VLLM_ASCEND_DSV41_KT_KNOB_FILE`` names a file.
  * It only ever calls kt's own setters; it cannot change weights, placement or
    routing, so a run with it enabled and never written to is the default arm.
  * Every change it applies is logged with the value and the file's mtime, so an
    arm's identity is in the server's own words rather than in the harness's.
  * It never raises into the server. A malformed file is logged and ignored: a
    measurement tool must not be able to kill the thing being measured.

File format -- one line, whitespace-separated ``key=value``::

    nz_prefetch_groups=8 nz_prefetch_groups_short=-1 nz_prefetch_hint=0
    nz_prefetch_legacy_stride=0

Both keys are optional; an absent key is left alone. Comments (``#``) and blank
lines are ignored.
"""

from __future__ import annotations

import os
import threading

from vllm.logger import logger

#: Env var naming the knob file. Unset (the default) means no thread is started.
ENV_KNOB_FILE = "VLLM_ASCEND_DSV41_KT_KNOB_FILE"

#: How often the file is stat()ed. The A/B harness writes the file and then
#: waits out one interval before timing, so this is a lower bound on its
#: per-arm dead time, not a per-step cost: nothing but stat() happens between
#: changes.
POLL_SECONDS = 0.5

#: key -> (kt setter name, value parser). Anything else in the file is a typo
#: and is reported as one.
_KNOBS: dict[str, tuple[str, str]] = {
    "nz_prefetch_groups": ("set_nz_prefetch_groups", "int"),
    # -1 = "same as nz_prefetch_groups". w2 strides 81,920 B per k-group against
    # w13's 18,432, so one distance need not fit both.
    "nz_prefetch_groups_short": ("set_nz_prefetch_groups_short", "int"),
    # 0 = T0 (today), 1 = T1, 2 = NTA.
    "nz_prefetch_hint": ("set_nz_prefetch_hint", "int"),
    "nz_prefetch_legacy_stride": ("set_nz_prefetch_legacy_stride", "bool"),
}

_thread: threading.Thread | None = None
_lock = threading.Lock()


def knob_file() -> str:
    """The configured path, or "" when the watcher is off."""
    return os.environ.get(ENV_KNOB_FILE, "").strip()


def parse(text: str) -> dict[str, object]:
    """``"a=1 b=0"`` -> ``{"a": 1, "b": False}``. Unknown keys raise."""
    out: dict[str, object] = {}
    for tok in text.replace("\n", " ").split():
        if tok.startswith("#"):
            break
        if "=" not in tok:
            raise ValueError(f"{tok!r} is not key=value")
        key, _, val = tok.partition("=")
        key = key.strip()
        spec = _KNOBS.get(key)
        if spec is None:
            raise ValueError(f"unknown knob {key!r}; known: {sorted(_KNOBS)}")
        _, kind = spec
        if kind == "int":
            out[key] = int(val)
        else:
            if val not in ("0", "1"):
                raise ValueError(f"{key}={val!r}: expected 0 or 1")
            out[key] = val == "1"
    return out


def apply(values: dict[str, object], ext=None) -> dict[str, object]:
    """Call kt's setters, then read the values BACK and return what kt reports.

    Reading back is the point: a setter that silently clamps (the distance is
    clamped to [0, 4096]) or a kt build without the knob would otherwise leave
    the log claiming an arm that is not running.
    """
    if ext is None:
        from kt_kernel import kt_kernel_ext as ext  # noqa: PLC0415  (lazy: kt is optional)
    got: dict[str, object] = {}
    for key, val in values.items():
        setter, _ = _KNOBS[key]
        fn = getattr(ext, setter, None)
        if fn is None:
            raise RuntimeError(
                f"this kt build has no {setter}: the runtime knob is not in the LIVE binary "
                "(measure/kt_rebuild.sh rebuilds and verifies it)")
        fn(val)
        reader = getattr(ext, key, None)
        got[key] = reader() if reader is not None else val
    return got


def _watch(path: str, interval: float, stop: threading.Event) -> None:
    last_sig: tuple | None = None
    while not stop.wait(interval):
        try:
            st = os.stat(path)
            sig = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            continue
        except OSError as exc:                      # pragma: no cover - defensive
            logger.warning("kt_offload knob watcher: stat(%s) failed: %s", path, exc)
            continue
        if sig == last_sig:
            continue
        last_sig = sig
        try:
            with open(path) as fh:
                text = fh.read()
            values = parse(text)
            if not values:
                continue
            got = apply(values)
        except Exception as exc:                    # noqa: BLE001 - must not kill the server
            logger.warning("kt_offload knob watcher: %s: %s: %s", path, type(exc).__name__, exc)
            continue
        logger.info("kt_offload knob watcher: applied %s (kt reports %s) from %s mtime_ns=%d",
                    values, got, path, sig[0])


def start_if_configured(*, interval: float = POLL_SECONDS) -> threading.Thread | None:
    """Start the watcher once per process. Returns the thread, or None if off."""
    global _thread
    path = knob_file()
    if not path:
        return None
    with _lock:
        if _thread is not None:
            return _thread
        stop = threading.Event()
        t = threading.Thread(target=_watch, args=(path, interval, stop),
                             name="kt-knob-ab", daemon=True)
        t.stop_event = stop                        # type: ignore[attr-defined]
        _thread = t
        t.start()
    logger.info("kt_offload knob watcher: 🔴 MEASUREMENT MODE -- watching %s every %.2f s for "
                "%s. A run with this set is an A/B run; its arms are in this log.",
                path, interval, sorted(_KNOBS))
    return t
