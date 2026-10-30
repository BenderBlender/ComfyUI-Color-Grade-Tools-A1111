"""Give ComfyUI the memory policy its own server gives it.

A script that imports ComfyUI as a library skips everything ``main.py`` sets
up, and two of those things matter a great deal.

**The headroom.** ``comfy/memory_management.py`` keeps ``RAM_CACHE_HEADROOM``
at zero until somebody sets it, and the server sets it per prompt from
``execution.py``. Until then ``ensure_pin_budget`` falls back to a **two
gigabyte** margin::

    shortfall = size + max(RAM_CACHE_HEADROOM / 2, 2048 * 1024 ** 2) - available

So a bare script lets ComfyUI pin host memory until two gigabytes are left,
on a machine where the editor would have kept ten or more. The machine does
not run out, but it spends its time paging, and everything else on it stops
responding. Setting the headroom here is the whole fix.

**The release callback.** ``pin_memory`` asks ``extra_ram_release`` to free
something before it pins; with no callback installed that call does nothing.
The server passes its node cache. A script has no node cache, so the
callback here only reports — but it must exist for the headroom to be
consulted at all.

Call :func:`before_comfy_imports` before anything imports
``comfy.model_management``, since it reads the flags once at import time,
and :func:`configure` once ComfyUI is importable.
"""

import logging

log = logging.getLogger(__name__)

DEFAULT_HEADROOM_GB = 24.0

_freed_total = 0


def before_comfy_imports(pinned=False, async_offload=False):
    """Set the flags ``model_management`` reads as it is imported.

    ``DISABLE_PINNED_MEMORY`` and the offload stream count are both captured
    at import time, so a flag set afterwards is simply ignored.
    """
    import comfy.cli_args

    args = comfy.cli_args.args
    if not pinned:
        args.disable_pinned_memory = True
    if not async_offload:
        args.disable_async_offload = True
    return args


def configure(headroom_gb=DEFAULT_HEADROOM_GB):
    """Install the headroom and a release callback, as the server does."""
    import comfy.memory_management as memory

    def release(target, free_active=False):     # noqa: ARG001
        # A script holds no node cache to drop, so there is nothing to free;
        # what matters is that the headroom above is consulted at all.
        global _freed_total
        _freed_total += 0
        return 0

    memory.set_ram_cache_release_state(release, int(headroom_gb * 1024 ** 3))
    log.info("ColorKit tools: keeping %.0f GB of headroom", headroom_gb)
    return memory.RAM_CACHE_HEADROOM


def report(tag=""):
    import torch

    bits = []
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        bits.append(f"VRAM {(total - free) / 2**30:.1f}/{total / 2**30:.1f}")
    try:
        import psutil
        vm = psutil.virtual_memory()
        bits.append(f"RAM {(vm.total - vm.available) / 2**30:.0f}/{vm.total / 2**30:.0f} GB")
        if vm.percent > 85:
            bits.append("** under pressure **")
    except ImportError:
        pass
    try:
        import comfy.model_management as mm
        pinned = getattr(mm, "TOTAL_PINNED_MEMORY", 0)
        if pinned:
            bits.append(f"pinned {pinned / 2**30:.1f} GB")
    except Exception:  # noqa: BLE001
        pass

    line = f"    [memory{' ' + tag if tag else ''}] " + "  ".join(bits)
    print(line, flush=True)
    return line


def free_everything(tag=""):
    """Unload, unpin, collect — in the order ComfyUI itself uses.

    ``unload_all_models`` only moves weights off the GPU. The host pages they
    were pinned into belong to a separate allocator with its own eviction,
    and the bookkeeping entries for models already gone are dropped by
    ``cleanup_models``. Missing any of the three leaves memory held.
    """
    import gc

    import torch

    import comfy.model_management as mm

    mm.unload_all_models()

    for name, call in (
        ("free_pins", lambda fn: fn(1 << 62, evict_active=True, loaded=True)),
        ("cleanup_models_gc", lambda fn: fn()),
        ("cleanup_models", lambda fn: fn()),
    ):
        fn = getattr(mm, name, None)
        if fn is None:
            continue
        try:
            call(fn)
        except Exception as exc:  # noqa: BLE001 - cleanup must not raise
            print(f"    [memory] {name}: {exc!r}")

    gc.collect()
    mm.soft_empty_cache(force=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report(f"after unload {tag}".strip())
