"""Process-pool map for embarrassingly parallel experiment loops.

Per-stock calibrations, per-stock GARCH/CAViaR fits, and independent config
runs are all independent — run them across cores instead of a serial loop.

macOS uses the spawn start method: workers re-import modules, so `fn` must
be a MODULE-LEVEL callable (functools.partial over one is fine; lambdas and
closures are not picklable). Keep the OpenMP isolation rule in mind: never
route work through a worker that imports both torch and lightgbm.
"""

import os
from concurrent.futures import ProcessPoolExecutor


def pmap(fn, items, workers: int | None = None) -> list:
    """Ordered map of fn over items using a process pool. Leaves two cores
    free by default so the machine stays usable."""
    items = list(items)
    if workers is None:
        workers = max(1, (os.cpu_count() or 4) - 2)
    if len(items) <= 1 or workers == 1:
        return [fn(x) for x in items]
    chunk = max(1, len(items) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items, chunksize=chunk))
