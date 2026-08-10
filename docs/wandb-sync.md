# wandb on a batch system: offline sync, what the references do, and what we should do

Written while preparing `scripts/bridge_smoke_512.sh`. The smoke run goes out with
`wandb.mode=online`, which is a bet that LUMI compute nodes can reach `api.wandb.ai`.
This records the alternative, so that if the bet loses the fix is a config change and not
an afternoon of reading.

Reference line numbers are for `reference/dsbm-pytorch/` and `reference/DDSBM/` at the
vendored revision.

---

## 1. The problem

`wandb.init(mode="online")` opens a connection to `api.wandb.ai` at process start and
streams metrics as training runs. On a cluster whose compute nodes are firewalled off from
the internet, that connection cannot be made: `init` blocks until it times out (default
`init_timeout` is 90 s in wandb 0.26.1) and the job either dies or proceeds without
logging, having burned allocation on nothing.

The trade is narrow but real. Offline mode costs you **live monitoring only**. The
permanent record, the cross-run comparison, the curves against `global_step`, the config
and system metrics — all of that survives a later upload untouched.

## 2. The standard pattern: offline + `wandb sync`

W&B's own answer, from [Can I run wandb offline?](https://docs.wandb.ai/support/models/articles/can-i-run-wandb-offline):

> To save metrics locally without an internet connection, set the environment variable
> `WANDB_MODE=offline`. […] Use `wandb sync YOUR_RUN_DIRECTORY` to transfer metrics to the
> cloud service.

The run lands in `<WANDB_DIR>/wandb/offline-run-<timestamp>-<run_id>/`, and the upload is a
separate step run later from a machine that does have a route out — on a cluster, the login
node, which shares the filesystem with the compute nodes. That shared filesystem is the whole
trick: the job writes, the login node reads and uploads.

Useful flags from the [`wandb sync` CLI reference](https://docs.wandb.ai/models/ref/cli/wandb-sync):

| flag | behaviour |
|---|---|
| *(no args)* | prints a summary of synced and unsynced runs, uploads nothing |
| `--sync-all` | syncs every unsynced run in the local `wandb` directory |
| `--include-synced` | includes runs already uploaded; **off by default** |
| `--mark-synced` | marks runs as synced after upload; defaults to true |
| `--clean` | deletes local data for already-synced runs, older than `--clean-old-hours` (default 24) |

The `--mark-synced` / `--include-synced` pair is the part to be careful with. A synced run is
marked, and a later `wandb sync` skips it unless you force it. That matters if you ever try to
sync a run *while it is still being written* — the docs do not state what happens when a
partially-synced, still-growing run is re-synced, and I have not tested it. Treat mid-run
syncing as unverified until we try it on a smoke run's output.

If live plots from an offline job ever become genuinely necessary, the community solution is
[`wandb-osh`](https://github.com/klieret/wandb-offline-sync-hook), which assumes exactly our
topology:

> Your ML experiments run on compute nodes without internet access (for example, using a batch
> system) […] your head/login node (with internet) have access to a shared file system.

A hook in the training loop touches a file in a communication directory; a daemon on the login
node watches that directory and runs `wandb sync` when it appears.
[`wandb-offline-sync`](https://pypi.org/project/wandb-offline-sync/) solves the same problem
by polling. Both are extra moving parts, and neither is worth adding before we know online
mode is actually unavailable.

## 3. The environment variables that matter on a shared filesystem

Quoted from [W&B environment variables](https://docs.wandb.ai/models/track/environment-variables):

| variable | documented behaviour | why it matters here |
|---|---|---|
| `WANDB_MODE` | "If you set this to `offline` wandb will save your run metadata locally and not sync to the server. If you set this to `disabled` wandb will turn off completely." | **inert in this repo — see §4** |
| `WANDB_DIR` | "Where to store all generated files. If unset, defaults to the `wandb` directory relative to your training script." | `conf/config.yaml:40` sets `chdir: false`, so cwd stays the *submission* directory; without this the run dir lands wherever you happened to type `sbatch` |
| `WANDB_CACHE_DIR` | "This defaults to `~/.cache/wandb`, you can override this location with this environment variable" | LUMI home quotas are small; artifact staging belongs on flash |
| `WANDB_API_KEY` | "Must be set if `wandb login` hasn't been run on the remote machine." | we rely on `~/.netrc` instead, which is why `WANDB_CONFIG_DIR` is deliberately left alone |

`scripts/bridge_smoke_512.sh` sets `WANDB_DIR` and `WANDB_CACHE_DIR` under `${FLASH_BASE}`.
Singularity forwards the host environment into the container by default, so no
`SINGULARITYENV_` prefix is needed.

## 4. The trap: the `init` argument beats the environment

`setup_wandb()` passes the mode explicitly:

```python
# train/bridge_prototype.py:346-351
run = wandb.init(
    entity=config["wandb"]["entity"],
    project=config["wandb"]["project"],
    mode=config["wandb"]["mode"],
    config=config,
)
```

Because the keyword argument is always present, **`export WANDB_MODE=offline` in a batch
script does nothing at all** — the config value silently wins. Verified against wandb 0.26.1:

```
$ WANDB_MODE=online python3 -c "wandb.init(project='precedence-test', mode='offline'); ..."
$ ls wandb/
offline-run-20260810_150345-5ns8g2p4      # the init argument won
```

So the switch is the Hydra override, never the environment:

```bash
python3 -u main.py experiment=smoke wandb.mode=offline
```

and if it becomes permanent for the machine, pin it in `conf/env/lumi.yaml`, which is where
machine-shaped facts belong.

## 5. What the references do

### `reference/dsbm-pytorch` — a logger abstraction, no offline story

Logging is a swappable backend chosen by a single key, `conf/config.yaml:19`:

```yaml
LOGGER: Wandb  # CSV
```

`get_logger()` at `bridge/runners/config_getters.py:327-352` dispatches to three
implementations defined in `bridge/runners/logger.py`: `WandbLogger` (a subclass of
PyTorch Lightning's, `logger.py:20`), `CSVLogger` (`logger.py:15`), and a no-op `Logger`
base for `LOGGER: NONE`.

Two things stand out:

- The CSV path is the offline-proof one, and it is configured for it —
  `flush_logs_every_n_steps: 1` at `config_getters.py:331` means the file on disk is current
  after every step, so `tail -f` works. `LOGGER: CSV` is a complete escape hatch from wandb
  that costs nothing to use.
- There is **no mode setting anywhere**, no `WANDB_MODE`, and no `wandb sync` in the repo.
  The wandb path hard-asserts an entity from the environment (`config_getters.py:344`,
  `wandb_entity = os.environ['WANDB_ENTITY']`) and otherwise assumes a working connection.

Its Hydra job config is worth noting as prior art for what we just did to `main.py` —
`conf/job.yaml` attaches a file handler `run.log` with `root: handlers: [console, file]`,
i.e. the same log-to-both-places arrangement.

### `reference/DDSBM` — mode is a config key, and `chdir` does the placement

`configs/general/general_default.yaml:5`:

```yaml
wandb: "online" # online | offline | disabled
```

threaded into `wandb.init(mode=cfg.general.wandb)` at `src/ddsbm/utils.py:402`. So DDSBM
*can* go offline by config — but nothing in the repo syncs afterwards, and its Slurm script
`scripts/zinc_random_train.sh` sets no wandb environment at all: `conda activate ddsbm` and
straight into training. That is a cluster with internet on the compute nodes.

Its run-dir layout falls out of `configs/config_train.yaml:12`, `chdir: True`. With Hydra
chdir'ing into the output directory, wandb's default "`wandb` relative to the training
script" resolves *inside* the run's own directory, which is why the README tree shows

```
outputs/zinc/2025-04-12_SB_0.999/
├── forward_5
│   ├── checkpoints
│   └── wandb
└── main.log
```

for free. We chose `chdir: false`, so we buy the same colocation explicitly with `WANDB_DIR`.
Also of note: DDSBM opens **one wandb run per (direction, iteration)**, named
`f"{direction}_{iteration}"` and tied together with `group=cfg.general.name`
(`utils.py:389-405`) — a different choice from ours, which is one run for the whole outer
loop with `forward/` and `backward/` metric prefixes.

### The headline

**Neither reference implements offline-plus-sync.** Both assume compute nodes with internet;
one of them offers a CSV logger as the fallback. The offline/`wandb sync` dance is an
HPC-community practice, not something we can copy from either codebase — which is worth
knowing before treating it as the obviously-correct default. I pinned `mode: offline` in
`conf/env/lumi.yaml` earlier in this work on exactly that unexamined assumption, and it was
premature.

## 6. Does LUMI actually need any of this?

Probably not. Two pieces of evidence:

- LUMI's [network documentation](https://docs.lumi-supercomputer.eu/hardware/network/) states
  that the external-connection block `193.167.209.128/26` "contains the login nodes and also
  the **NAT gateways for the compute nodes**" — compute nodes are explicitly routed outbound.
- "Compute nodes do not have internet access" exists on LUMI's service-status page as an
  [*incident*](https://lumi-supercomputer.eu/lumi-service-status/compute-nodes-do-not-have-internet-access/),
  opened 2024-08-25 and closed 2024-08-31 with "The internet connectivity issue on the compute
  nodes is now fixed." A six-day outage that warranted a status page is strong evidence that
  connectivity is the normal state.

Neither is a promise about our project's firewall rules today, which is why the smoke run is
the test rather than the assumption.

## 7. What to do

Run the smoke online, and read the `.out`:

- **Training lines appear normally** → online works. Nothing to change, ever again.
- **A stall of ~90 s at `wandb.init`, then errors** → no route out. Rerun with
  `wandb.mode=offline`, pin it in `conf/env/lumi.yaml`, and upload afterwards from a login
  node:

  ```bash
  singularity exec -B /flash/project_465002822 \
    /project/project_465002822/containers/sb-match-20260627.sif \
    wandb sync /flash/project_465002822/sb-match/outputs/wandb/offline-run-*
  ```

  The run id in the path also appears in the startup log line, so a `.out` can always be
  matched to its wandb run.

On live monitoring, note that the question is now much less pressing than it looks: since
`train/bridge_prototype.py` logs through Hydra's root logger, the Slurm `.out` carries loss,
grad_norm, per-stage wall time and cache-refresh timings *live*, and a login node can
`tail -f` it with no network involved. Live diagnosis is the log's job; wandb's job is
comparing this run to the last five. Only the second one needs a server, and it can wait
until the job ends. `wandb-osh` is the answer only if live *plots* specifically become
necessary.

One thing offline mode does change materially: an unsynced run directory is the sole copy of
the metrics, and `WANDB_DIR` currently points at flash, which is purged. Online mode makes
that irrelevant; offline mode means syncing promptly, or moving `WANDB_DIR` to `projappl`.

---

## Sources

- W&B, [Can I run wandb offline?](https://docs.wandb.ai/support/models/articles/can-i-run-wandb-offline)
- W&B, [`wandb sync` CLI reference](https://docs.wandb.ai/models/ref/cli/wandb-sync)
- W&B, [Environment variables](https://docs.wandb.ai/models/track/environment-variables)
- klieret, [wandb-offline-sync-hook (`wandb-osh`)](https://github.com/klieret/wandb-offline-sync-hook) and its [readme](https://wandb-offline-sync-hook.readthedocs.io/en/stable/readme.html)
- PyPI, [`wandb-offline-sync`](https://pypi.org/project/wandb-offline-sync/)
- LUMI, [Network and interconnect](https://docs.lumi-supercomputer.eu/hardware/network/)
- LUMI, [service status: compute nodes do not have internet access](https://lumi-supercomputer.eu/lumi-service-status/compute-nodes-do-not-have-internet-access/) (resolved 2024-08-31)
- Digital Research Alliance of Canada, [Weights & Biases (wandb)](https://docs.alliancecan.ca/wiki/Weights_%26_Biases_(wandb)/en) — an HPC centre's own wandb page, listed as further reading; the site blocked automated retrieval, so nothing above is quoted from it.
