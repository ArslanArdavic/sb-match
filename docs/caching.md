# Coupling caching: how the reference does it, and how to port it to `train/bridge.py`

All reference line numbers are for `reference/dsbm-pytorch/` at the vendored revision:

- `bridge/data/cacheloader.py`
- `bridge/trainer_dbdsb.py`
- `bridge/runners/repeater.py`
- `conf/dataset/afhq_transfer.yaml`

---

## 1. The problem

Both codebases need, at the start of every drift-fitting stage, a coupling
`(x0, x1)` produced by simulating the *previous* drift with Euler–Maruyama. Simulation
is the expensive half of DSBM: it costs `num_pairs x num_steps` network forward passes
with no gradient reuse.

**What `bridge.py` does today.** `train/bridge.py:117-135` (forward) and
`train/bridge.py:169-187` (backward) sweep the *entire* class dataset through an `N`-step
EM loop once per outer iteration, per direction:

| | cats | steps `N` | NFE per direction | x 2 directions x `n_outer=15` |
|---|---|---|---|---|
| `bridge.py` | 5153 | 100 | 515,300 | **15.5M** |

Plus `train/bridge.py:32-35` materializes both full datasets as dense tensors in host RAM
before training even starts:

```python
X0 = torch.cat([x for (x,_) in x0_dataloader])
XT = torch.cat([x for (x,_) in xT_dataloader])
m = min(len(X0), len(XT))
coupling = TensorDataset(X0[:m], XT[:m])
```

At 64px that is 5153·3·64·64·4 B ≈ 253 MB per side — tolerable. At 512px it is
**≈ 16 GB per side**, which is the wall you hit when scaling the prototype up. Every
`coupling = TensorDataset(...)` at `train/bridge.py:134` and `:186` rebuilds another
full-size tensor of the same order.

**What the reference does.** It decouples *cache size* from *dataset size*. For AFHQ
(`conf/dataset/afhq_transfer.yaml:33-36`) it simulates **400 pairs** at a time and trains
on them for **1000 gradient steps**, then throws them away and simulates a fresh 400 drawn
from a different part of the dataset. The dataset is never fully simulated, and never fully
resident in RAM.

---

## 2. How the reference implements it

### 2.1 The unit of work is a gradient step, not an epoch

This is the precondition for everything else. `bridge.py` is epoch-structured
(`train/bridge.py:89` `for epoch in range(config["epochs_per_drift"])`, `:95` a `DataLoader`
pass over the whole coupling). The reference is iteration-structured
(`trainer_dbdsb.py:612`):

```python
for i in tqdm(range(step, num_iter + 1), mininterval=30):

    if (i == step) or ((i-1) % self.args.cache_refresh_stride == 0):
        new_dl = None
        torch.cuda.empty_cache()
        if not first_it:
            new_dl = self.new_cacheloader(*self.compute_prev_it(forward_or_backward, n), refresh_idx=(i-1) // self.args.cache_refresh_stride)
```

`trainer_dbdsb.py:612-618`. Every `cache_refresh_stride` iterations the old cache handle is
dropped (`new_dl = None`, so the memmap and its dataloader workers are released), CUDA
cache is cleared, and a new cache is built. Batches come from `next(new_dl)`
(`trainer_dbdsb.py:629-631`) — an *infinite* iterator, so nothing has to align with an
epoch boundary.

`compute_prev_it` (`trainer_dbdsb.py:714-721`) encodes which net generates the cache: to
train `b` at IPF step `n` you simulate with `f` from step `n-1`; to train `f` at step `n`
you simulate with `b` at step `n`. `first_it` (`trainer_dbdsb.py:604-610`) skips caching
entirely for the very first pass, where the coupling is independent/reference and can be
drawn directly from raw data (`trainer_dbdsb.py:626`).

### 2.2 The four sizing knobs

From `conf/dataset/afhq_transfer.yaml`:

```yaml
cache_npar: 400            # pairs held in one cache
cache_batch_size: 10       # pairs simulated per EM call (VRAM knob)
cache_refresh_stride: 1000 # training iterations served by one cache
num_repeat_data: 1         # noise draws reused per cached pair
batch_size: 4              # training batch
num_iter: 25000            # gradient steps per drift stage
num_steps: 100             # EM steps
```

`num_batches = cache_npar // cache_batch_size` (`trainer_dbdsb.py:307-309`, with an
assert that the division is exact). So one AFHQ cache costs 400·100 = 40k NFE and serves
1000 gradient steps; a full 25000-iteration stage costs 25 refreshes = **1M NFE**, against
5153·100 = 515k for *one* full-dataset sweep in `bridge.py`. The reference spends more
total NFE but spreads it — the peak memory and the latency of any single stall are what
change, and the cache is never larger than 400 pairs.

`trainer_dbdsb.py:97-104` records the two identities that tell you whether the knobs are
sane:

```python
self.npar = len(init_ds)
self.cache_npar = self.args.cache_npar if self.args.cache_npar is not None else self.batch_size * self.args.cache_refresh_stride // self.num_repeat_data
self.cache_epochs = (self.batch_size * self.args.cache_refresh_stride) / self.cache_npar             # passes over one cache
self.data_epochs = (self.num_iter * self.cache_npar) / (self.npar * self.args.cache_refresh_stride)  # dataset repeats across a stage
```

For AFHQ: `cache_epochs = 4·1000/400 = 10` (each cached pair is seen ~10 times before being
discarded) and `data_epochs = 25000·400/(5153·1000) ≈ 1.94` (across a whole stage the
cache draws ~2x the dataset, so coverage is good even though no single cache holds more
than 8% of it). The default when `cache_npar` is unset makes `cache_epochs == 1`.

**Coverage is the key idea.** A 400-pair cache is not a fixed 400-image subset. Each
refresh draws fresh batches from `cache_init_dl` / `cache_final_dl`
(`trainer_dbdsb.py:253`, `:268`), which are shuffled loaders wrapped in `repeater`
(`bridge/runners/repeater.py:3-5`):

```python
def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data
```

so consecutive refreshes walk different parts of the dataset and reshuffle on wraparound.
Note `cache_batch_size` is used *only* for these cache loaders; training reads from the
cache at the much smaller `batch_size` (`trainer_dbdsb.py:320`).

### 2.3 Cache construction, step by step

`DBDSB_CacheLoader` is `cacheloader.py:75-268`. Ignoring the distributed and conditional
(`cdsb`) branches, the flow is:

1. **Name the artifacts** (`cacheloader.py:79-83`): `cache/cache_{f|b}_{n:03}.npy` for the
   data and `cache_{f|b}_{n:03}.txt` for a completion marker. `cache_dir` is created at
   `trainer_dbdsb.py:124-127`.
2. **Temp dir per refresh** (`cacheloader.py:90-91`):
   `cache/temp_{direction}_{n:03}_{refresh_idx:03}/`.
3. **Skip if already done** (`cacheloader.py:97-105`): read the `.txt`, and if it contains
   the same `refresh_idx/refresh_tot`, set `use_existing_cache = True` and jump straight to
   loading the `.npy`. This is what makes a requeued HPC job cheap.
4. **Simulate batch by batch** (`cacheloader.py:112-131`):

```python
for b in range(num_batches):
    b_dist = b * ipf.accelerator.num_processes + ipf.accelerator.process_index
    try:
        batch_x0, batch_x1 = torch.load(os.path.join(temp_cache_dir, f"{b_dist}.pt"))
        ...
    except:
        ipf.set_seed(seed=ipf.compute_current_step(0, n+1)*num_batches_dist*refresh_tot + num_batches_dist*refresh_idx + b_dist)
        init_batch_x, init_batch_y, final_batch_x, _, _ = ipf.sample_batch(init_dl, final_dl)
        with torch.no_grad():
            batch_x0, batch_y, batch_x1 = langevin.generate_new_dataset(init_batch_x, init_batch_y, final_batch_x, sample_fn, sample_direction, sample=sample, num_steps=ipf.cache_num_steps)
            batch_x0, batch_x1 = batch_x0.contiguous(), batch_x1.contiguous()
            torch.save([batch_x0, batch_x1], os.path.join(temp_cache_dir, f"{b_dist}.pt"))
```

   Three things worth stealing: the `try`/`except` makes each *batch* independently
   resumable (a job killed at batch 27/40 resumes at 27); the seed is a deterministic
   function of `(ipf step, refresh_idx, batch index)` so a resumed batch is bit-identical
   to what it would have been; and `generate_new_dataset`
   (`bridge/sde/diffusion_bridge.py:102-112`) returns only the *endpoints* `(z0, z1)` of
   the trajectory — the intermediate states are discarded, which is why the on-disk record
   is `2 x C x H x W` per sample and not `num_steps x C x H x W`.

5. **Aggregate into one memmap** (`cacheloader.py:151`, `:191-193`):

```python
fp = open_memmap(cache_filepath_npy, dtype='float32', mode='w+', shape=(npar, 2, *batch_x0.shape[1:]))
...
batch = torch.stack([batch_x0, batch_x1], dim=1).float().cpu().numpy()
fp[b_dist*cache_batch_size_dist:(b_dist+1)*cache_batch_size_dist] = batch
fp.flush()
```

   `numpy.lib.format.open_memmap` writes a real `.npy` header and returns a writable
   memmap, so the array is written slice-by-slice and never exists in RAM in full. `del fp`
   at `cacheloader.py:200` closes it.

6. **Mark complete, then delete temps** (`cacheloader.py:204-208`) — in that order, so a
   crash between the two only costs disk, never correctness:

```python
f = open(cache_filepath_txt, 'w')
f.write(f'{refresh_idx}/{refresh_tot}')
f.close()
shutil.rmtree(temp_cache_dir)
```

7. **Reopen read-only** (`cacheloader.py:213-224`): `np.load(cache_filepath_npy, mmap_mode='r')`,
   with a retry loop for filesystem lag (relevant on Lustre).
8. **Prune old caches** (`cacheloader.py:249-254`): for each direction, `glob` all
   `cache_{fb}_**.npy`, keep the newest, delete the rest. Without this you accumulate one
   full cache per IPF iteration.
9. **Return a lazy dataset** (`cacheloader.py:268` → `cacheloader.py:10-23`):

```python
class MemMapTensorDataset(Dataset):
    def __init__(self, npy_file_list) -> None:
        self.npy_file_list = npy_file_list
        self.data_file_list = [np.load(npy_file, mmap_mode='r') for npy_file in self.npy_file_list]

    def __getitem__(self, index):
        out = []
        for data_file in self.data_file_list:
            data = torch.from_numpy(data_file[index])
            out = out + [d for d in data]
        return out

    def __len__(self):
        return len(self.data_file_list[0])
```

   `__getitem__` touches exactly one `(2, C, H, W)` record; the OS page cache handles the
   rest. Because it is a plain `Dataset`, `num_workers > 0` parallelizes the reads
   (`trainer_dbdsb.py:232-243`). The `[d for d in data]` unpacks the leading `2` into
   `(x0, x1)`, which is why training does `x0, x1 = next(new_dl)` at
   `trainer_dbdsb.py:631`.

### 2.4 Two more cost levers

- **`cache_num_steps`** (`trainer_dbdsb.py:106`, used at `cacheloader.py:127`): the EM step
  count used to *build the cache* can be lower than the one used at test time
  (`test_num_steps`). `record_langevin_seq` re-interpolates the timestep grid for an
  arbitrary `num_steps` (`bridge/sde/diffusion_bridge.py:62-67`). Halving it halves cache
  cost directly.
- **`num_repeat_data`** (`trainer_dbdsb.py:634`): each cached pair is expanded
  `repeat_interleave(num_repeat_data)` inside the training batch, so one simulated pair
  yields several independent bridge/noise draws per step. It buys `cache_epochs` without
  buying NFE. AFHQ leaves it at 1.

### 2.5 Pre-caching at stage end

`trainer_dbdsb.py:677-681` builds the *next* stage's cache immediately after the current
stage finishes training and checkpoints, with `build_dataloader=False`:

```python
self.save_ckpt(num_iter, n, forward_or_backward)
if not first_it_fn(*self.compute_next_it(forward_or_backward, n)):
    self.new_cacheloader(forward_or_backward, n, build_dataloader=False)
```

The result is discarded in memory but the `.npy` + `.txt` land on disk, so if the job dies
in the gap the next run's step 3 short-circuits.

### 2.6 What the older `CacheLoader` did (contrast)

`cacheloader.py:26-72` is the pre-DBDSB path: it concatenates every batch into one big
in-RAM `TensorDataset(all_x, all_out, all_steps)` and, crucially, stores *every timestep*
of every trajectory (`x.flatten(start_dim=0, end_dim=1)` at `:51`). That is the design the
DBDSB loader replaced, and it is structurally what `bridge.py` does today (minus the
timesteps). Useful as a reminder of what not to port.

---

## 3. Mapping to `bridge.py`

| reference concept | where it lives | `bridge.py` today |
|---|---|---|
| `npar` (dataset size) | `trainer_dbdsb.py:97` | `len(x0_dataloader.dataset)` = 5153 |
| `cache_npar` | `trainer_dbdsb.py:98` | **absent** — implicitly `npar` |
| `cache_batch_size` | `trainer_dbdsb.py:249` | `config["sample_batch_size"]` (`bridge.py:29-30`) |
| `cache_refresh_stride` | `trainer_dbdsb.py:614` | **absent** — refresh is once per outer, at `bridge.py:134` / `:186` |
| `num_steps` / `cache_num_steps` | `trainer_dbdsb.py:106` | `config["N"]` for both |
| `repeater` | `repeater.py:3` | **absent** — `bridge.py` re-iterates finite loaders |
| memmap dataset | `cacheloader.py:10` | `TensorDataset` in RAM (`bridge.py:35`, `:134`, `:186`) |
| resume marker | `cacheloader.py:97-105`, `:204-206` | **absent** |
| cache pruning | `cacheloader.py:249-254` | n/a |
| `first_it` | `trainer_dbdsb.py:604-610` | the independent coupling at `bridge.py:32-35` |

Note one structural difference to preserve: in `bridge.py` the variable `coupling`
(`bridge.py:35`, reassigned at `:134` and `:186`) is *shared* between the two directions —
the forward stage overwrites it and the backward stage trains on it. In the reference each
direction has its own cache file keyed by `(direction, n)`, which is what allows resume and
pruning to be per-direction. Keep the two caches separate when you port.

---

## 4. Proposed helpers under `utils/`

Put everything in a new **`utils/cache.py`**; the only thing that arguably belongs
elsewhere is the infinite-loader wrapper, which fits naturally next to the existing loaders
in `utils/process_image.py`.

**`utils/process_image.py`** — one addition:

```python
def repeater(dataloader):
    """Infinite reshuffling iterator over a finite DataLoader."""
```

Same three lines as `repeater.py:3-5`. Needed so `draw_pairs` can pull `cache_npar` samples
regardless of dataset length or epoch boundaries.

**`utils/cache.py`** — suggested surface:

```python
def cache_paths(cache_dir, direction, outer):
    """-> (npy_path, txt_path, temp_dir) for cache_{direction}_{outer:03}.*"""

def cache_is_complete(txt_path, refresh_idx, refresh_tot):
    """Read the marker; True iff it matches this refresh. Mirrors cacheloader.py:97-105."""

def mark_cache_complete(txt_path, refresh_idx, refresh_tot):
    """Write '{refresh_idx}/{refresh_tot}'. Mirrors cacheloader.py:204-206."""

def prune_old_caches(cache_dir, keep_path, directions=("forward", "backward")):
    """Keep only the newest cache_{d}_*.npy per direction. Mirrors cacheloader.py:249-254."""


class MemMapPairDataset(torch.utils.data.Dataset):
    """Lazy (x0, xT) reader over a (npar, 2, C, H, W) float32 .npy.
    __getitem__ -> (tensor, tensor); __len__ -> npar. Mirrors cacheloader.py:10-23."""


@torch.no_grad()
def simulate_euler_maruyama(net, x_start, N, sigma, device):
    """One EM rollout; returns only the endpoint.
    Lift of bridge.py:126-129 / :178-181 verbatim so the two stay in sync."""


def build_cache(net, source_loader_iter, *, cache_dir, direction, outer,
                cache_npar, cache_batch_size, N, sigma, device,
                refresh_idx=0, refresh_tot=1, seed_base=0, resume=True):
    """Simulate cache_npar pairs and return a MemMapPairDataset.

    - short-circuit on cache_is_complete
    - num_batches = cache_npar // cache_batch_size   (assert exact)
    - per batch: try torch.load(temp/{b}.pt) else seed, draw from
      source_loader_iter, simulate_euler_maruyama, torch.save
    - aggregate via numpy.lib.format.open_memmap((cache_npar, 2, C, H, W)),
      writing slice-by-slice with flush()
    - mark_cache_complete, shutil.rmtree(temp), prune_old_caches
    - return MemMapPairDataset(npy_path)
    """
```

`build_cache` needs to know pair *orientation*: the forward net maps `x0 -> xT` so the
record is `(x0, x_end)`, while the backward net maps `xT -> x0` so the record is
`(x_end, xT)`. Handle it inside `build_cache` with the `direction` argument, the way
`generate_new_dataset` does at `diffusion_bridge.py:103-112` — do not push that swap back
into the training loop, since `bridge.py:150-151` already has direction-specific argument
juggling and duplicating the convention in two places is how sign bugs get in.

---

## 5. Integration plan for `train/bridge.py`

Three stages, each independently useful. Stage 1 alone removes the 16 GB RAM wall.

### Stage 1 — bound the cache size (in-memory, no disk)

Smallest change that gets the main win. Keep the epoch structure.

- Delete `bridge.py:32-35`. Replace the full-dataset materialization with two persistent
  infinite iterators, `x0_iter = repeater(x0_dataloader)` and `xT_iter = repeater(xT_dataloader)`,
  built once outside the outer loop (they must persist so successive refreshes see fresh
  data — rebuilding them each outer iteration re-shuffles from scratch and partly defeats
  the coverage argument).
- Build the initial independent coupling by drawing `cache_npar` pairs from those
  iterators instead of the whole dataset.
- Replace the sampling blocks at `bridge.py:117-135` and `:169-187` with
  `build_cache(...)` calls limited to `cache_npar` pairs, still returning an in-RAM
  `TensorDataset`.
- Add `cache_npar` and `cache_batch_size` to `config`, and extend the `PRM` assert at
  `bridge.py:18-19`.
- Reinterpret `epochs_per_drift`: it is now passes over the *cache*, so the gradient steps
  per stage drop by `npar / cache_npar`. Compensate by raising `epochs_per_drift` (this is
  exactly the reference's `cache_epochs`), or move to Stage 2.

At this point, log the two identities from `trainer_dbdsb.py:99-100` at startup —
`cache_epochs` and `data_epochs` — so a misconfigured run is obvious from the first lines
of the job log rather than 6 hours in.

### Stage 2 — refresh on a stride (structural)

Convert the inner loop from epochs to iterations, mirroring `trainer_dbdsb.py:612-618`.
Replace `for epoch in ...: for x0, xT in DataLoader(coupling, ...)` at `bridge.py:89-95`
(and `:141-147`) with `for i in range(1, iters_per_drift + 1)`, a
`if (i - 1) % cache_refresh_stride == 0: cache_dl = repeater(DataLoader(build_cache(...)))`
guard at the top, and `x0, xT = next(cache_dl)` for the batch.

Consequences to plan for:

- `losses[...]["epoch"]` (`bridge.py:70-79`, `:112-114`) loses its meaning. Replace the
  epoch series with a windowed mean over the last `stride` steps, or key it by
  `refresh_idx`. The plotting in `tests/run_forward_backward_prototype.py:54-65` reads
  those keys, so update it in the same change.
- The stage's total gradient steps become an explicit knob (`iters_per_drift`) instead of
  `epochs_per_drift x ceil(npar / batch_size)`. Record the old effective value in
  `docs/experiments.md` before switching so the 64px prototype numbers stay comparable.
- Optionally add `num_repeat_data` via `repeat_interleave` on the drawn batch
  (`trainer_dbdsb.py:634`) — cheap, and it lets you push `cache_refresh_stride` up without
  overfitting the cache.

### Stage 3 — persist to disk and resume

Only worth it once you are back at 512px on a job-limited queue (the 6h limit noted in
`docs/experiments.md:5`). Switch `build_cache` to the memmap path and enable the marker
file, temp batches, and pruning. Then:

- Pass a `cache_dir` in `config`, on fast scratch (`/flash/project_465002822/...`), not on
  the project filesystem — the temp `.pt` files are written and deleted once per refresh.
- Add the retry-on-missing-file loop from `cacheloader.py:213-224` if the cache dir is on
  Lustre.
- Mirror `trainer_dbdsb.py:677-681`: build the next stage's cache right after saving the
  checkpoint at the end of a stage, so a resubmitted job skips it.
- Resume needs the outer index and direction persisted alongside the net weights;
  `bridge.py` currently returns nets only at `:189` and saves them in the test harness.
  Caching-with-resume is only half a resume story — the optimizer state and `outer` counter
  need checkpointing too, or the recovered cache is loaded and then discarded by a run that
  restarts at `outer = 0`.

### Disk budget

One 512px cache is `cache_npar x 2 x 3 x 512 x 512 x 4 B` = `cache_npar x 6.29 MB`
(400 pairs → 2.5 GB). Two directions live at once, plus temp `.pt` files during a build.
At 64px it is `cache_npar x 98 KB`, negligible.

---

## 6. Pitfalls

1. **`drop_last`.** `load_afhq_train` defaults to `drop_last=True`
   (`utils/process_image.py:20`); `bridge.py:29-30` overrides it to `False`. Under a
   `repeater` you want `drop_last=True` so every drawn batch is exactly
   `cache_batch_size` — the reference asserts this (`cacheloader.py:119`) and the memmap
   slice arithmetic at `cacheloader.py:192` depends on it.
2. **`cache_npar % cache_batch_size == 0`.** Assert it (`trainer_dbdsb.py:308`). The
   memmap is preallocated at `cache_npar` rows; a short final batch leaves zeros.
3. **Don't reuse `sample_batch_size` for both roles.** In `bridge.py` it currently sizes
   the loaders *and* the EM rollout. Once cached, the EM batch (`cache_batch_size`, VRAM
   bound) and the training batch (`batch_size`, optimization bound) are separate knobs;
   the reference keeps them at 10 and 4 respectively.
4. **`eval()`/`train()` around cache building.** `bridge.py:120` and `:172` already do
   this; keep it inside `build_cache` so the caller cannot forget, and keep the
   `torch.cuda.empty_cache()` from `bridge.py:135` / `:187` before a build.
5. **Stale cache across code changes.** The marker file only encodes
   `refresh_idx/refresh_tot`, not `sigma`, `N`, or the net weights. If you change the SDE
   or the model and rerun into the same `cache_dir`, you will silently train on stale
   pairs. Either wipe `cache_dir` per experiment or write the relevant config into the
   marker and compare it — the latter is a small deviation from the reference and worth it.
6. **`data_epochs < 1`.** If `cache_npar x num_iter / (npar x stride) < 1`, a whole stage
   never sees part of the dataset. That is a legitimate choice, but it should be a
   deliberate one — hence logging the identity at startup.
