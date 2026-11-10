# Benchmarks

Two jobs, and they need different things from the same measurements:

1. **Absolute throughput** — how fast is the simulation loop, and where does
   the time go?
2. **A baseline** — did a change cost us throughput?

## Running

```bash
# The engine. Release only: a debug build of the execution kernel is ~50x
# slower and the numbers mean nothing.
cargo run --release --example bench -p alphax-sim

python benchmarks/compare.py                    # summary
python benchmarks/compare.py --check-baseline   # regression gate
python benchmarks/compare.py --write-baseline   # accept current numbers

# Instruction counts against main. Needs valgrind; on macOS see
# "Instruction counts" below.
python benchmarks/instructions.py
```

## Same work, or no number

The runner reads `workloads.tsv` and generates its inputs from a formula.
Every row reports a checksum of the arrays it produced and the fill count its
simulation returned, and `compare.py` **aborts** if rows of one workload
disagree, for instance a threaded run that did different work from the serial
one.

A "faster" number that turns out to be less work is worse than no number. The
gate is tested by corrupting a result file and confirming it fires, rather than
trusted because it exists.

The loop is timed on its own, with no language bindings in the measurement. The
Python layer (`alphax-py`) is timed separately by `bench_python.py`, which
covers what a caller pays around the loop — signal construction, the sweep
wrapper, `Portfolio` reads — rather than the loop itself.

`from_signals` has its own workloads in `workloads_signals.tsv`. They exist for
the regression gate: that loop is what the Python layer calls most.

## Results on the reference machine

4-core Xeon @ 2.10GHz. Millions of elements (bars × columns) per second, best
of seven, from the committed `baseline.tsv`:

| Workload | 1 thread | 2 threads | 4 threads |
|---|---:|---:|---:|
| `single_long` (200000 × 1, every bar fills) | 16.7M | 16.7M | 16.6M |
| `single_long_sparse` (200000 × 1) | 36.8M | 36.4M | 36.7M |
| `sweep_small` (2000 × 500, every bar fills) | 16.6M | 17.9M | 29.9M |
| `sweep_small_sparse` (2000 × 500) | 35.7M | 66.1M | 132.0M |
| `sweep_wide` (1000 × 2000) | 35.0M | 66.5M | 133.3M |
| `sweep_deep` (20000 × 100) | 35.6M | 67.9M | 131.9M |
| `target_percent` (20000 × 100, revalues every bar) | 15.1M | 17.0M | 28.8M |
| `with_costs` (20000 × 100) | 30.4M | 52.1M | 110.6M |

Three things show in the table:

- A single column cannot use more than one thread: columns are the parallel
  axis, so `single_long` is flat across thread counts.
- Sparse runs scale almost linearly with threads because they are
  compute-bound. Dense runs, where every bar writes a record, are memory-bound
  and scale far less (`sweep_small`, `target_percent`).
- Ratios between rows are steadier than the absolute numbers, which move with
  how busy the machine is. Quoting any single figure implies a precision this
  benchmark does not have.

Which parts of the design contribute what to the single-thread figure — the
`FlexArray` scalar path, column-major scans, stack-allocated state — is not
attributed here. What *is* measured is where the remaining time goes; see the
next section.

## Where the remaining time goes

Measured by building variants as separate binaries and interleaving their runs,
five rounds each, comparing medians. Cross-session A/B does **not** work here:
the baseline drifted 11% between two sittings and made a 12% "win" evaporate.

| Change | Dense workloads | Sparse workloads |
|---|---:|---:|
| Preallocate the serial record buffer | **+6% to +19%** | ~0% |
| Hoist the 7 column-constant parameter checks out of validation | +10% | **+20%** |
| Build the order request once per column, not once per bar | +11% | +8% |
| Remove record writing entirely (ceiling, not a change) | +36% to +66% | ~0% |
| Array-of-structs record buffer, transposed at the end | −5% to −27% | −22% |

Landed: the preallocation, and the two hoists, which ship together as
`ValidatedOrder` — one order request built per column, carrying its own
parameter check, with the four bar-varying fields rewritten each bar. Together
**+30% sparse and +22% dense**, measured twice on separate sittings (29.6% and
29.5% sparse) before being believed.

Validation is the larger half on sparse workloads and the smaller half on
dense ones, which is the shape to expect: a bar that does not fill used to pay
all fourteen checks and almost nothing else, so removing seven of them is most
of that bar's work. A bar that does fill has real arithmetic to amortize them
against.

**One row here reverses an earlier result.** Building the order once per column
was previously recorded at −5% to −36% — *worse*. That measurement held the
hoisted order in a `thread_local` `RefCell` and spent more on borrow
bookkeeping per bar than it saved; it measured the container, not the hoist.
A plain local rebuilt only when `params_vary_by_row()` says to is +8% to +11%.
The negative result was real for what it tested and wrong about the question
it was taken to answer.

Not landed, and why:

- **Record writing** has the largest ceiling and nothing to reach it with. The
  obvious approach — reserve exactly, then write through raw pointers — is
  blocked by the workspace's `unsafe_code = "deny"`. That deny costs nothing
  here: a *safe* stand-in that removes the same bookkeeping (pre-sized
  vectors written by index) measured 8–12% **slower** on every workload, since
  `Vec::push`'s capacity branch is already free and the pre-fill is an extra
  pass over memory the dense workloads are already saturating. The 36–66%
  figure is the ceiling for not writing records at all, which is not an
  optimization; it should not be read as available headroom.
- **Array-of-structs records** measured worse: six bounds checks in the hot
  loop traded for one, plus a full extra transpose pass.

## What the baseline gate can and cannot tell you

Running the suite unchanged, repeatedly, on the reference machine:

| Threads | Median spread | Worst |
|---:|---:|---:|
| 1, best-of-3 | 3.4% | 9.5% |
| 1, best-of-7 | **1.3%** | **3.7%** |
| 2 | 4.1% | 21.9% |
| 4 | 5.9% | **47.3%** |

The runner samples best-of-7 for this reason: at best-of-3 the noise floor was
wide enough to hide a 10% regression, and wide enough that three separate
optimization experiments produced numbers that did not survive a paired
re-measurement.

The gate therefore runs at a 10% threshold on **single-threaded rows only**. A gate whose
noise floor is wider than the regressions it is meant to catch does not gate
anything; it teaches you to ignore it. The threaded numbers stay in the results
for the summary, where a 2x difference survives the noise, but they
are not evidence of a regression on a shared machine. `--include-threaded`
exists for a quieter one.

Even at one thread, a result near the threshold deserves a re-run before it
deserves an investigation.

**A baseline is only as good as the quiet it was measured in.** One was taken
minutes after a package install, while the machine was still busy. Every
subsequent run then reported +35% to +55% across all ten workloads on a hot
path the diff had not touched. A uniform *improvement* is the same signal as a
uniform regression and a more dangerous one, because nobody investigates good
news — they refresh the baseline, and the busy machine's numbers get baked in
so the next honest run looks like a regression. The gate now says so in both
directions.

**A uniform slowdown is about the machine, not the code.** The gate says so
itself: if every workload regressed by close to the same factor, it prints a
note pointing at the load average. A code regression is selective — it hits the
workloads that exercise the changed path and leaves the rest alone. Observed
during this project: an unchanged build reported −40% to −43% across all ten
workloads, with a load average of 1.37 on four cores. The gate still exits
non-zero; the note explains, it does not excuse.

The threaded figures were found by trying to inject a 30% slowdown and watching
the same code report "+63%" between runs instead.

## Instruction counts

A feature that is switched off may cost at most 3%, and interleaved timings move by about 4% between rounds on a
shared machine. Timings alone cannot tell a 3.8% regression from a 2% one.
`instructions.py` counts instructions instead, which repeat exactly from run
to run:

```bash
python benchmarks/instructions.py                        # against main
python benchmarks/instructions.py --base origin/main
python benchmarks/instructions.py --base-dir ../checkout   # an existing tree
```

It builds the `bench` example from this tree and from the base (checked out
in a scratch worktree) into `target/instructions/`, runs each workload once
under callgrind with counting switched on only inside `run_once` (the `--once`
mode of the example, so input generation is not counted), and reports the
change per workload and as a geometric mean per loop. It aborts if the two
builds did different work, the same rule as above, and exits non-zero if a
loop's mean grew by more than 3%.

valgrind does not run on macOS, so there it runs in a container:

```bash
docker build -t alphax-callgrind -f benchmarks/callgrind.Dockerfile benchmarks
docker run --rm -v "$PWD:$PWD" -v alphax-cargo:/usr/local/cargo/registry \
    -w "$PWD" alphax-callgrind python3 benchmarks/instructions.py --base origin/main
```

Once, a build in the container missed an edit made just before it and the
check counted the previous binary. A result that did not move after an edit
deserves `CARGO_TARGET_DIR=target/instructions/head cargo clean --release -p
alphax-sim` and a second run before it is believed.

**A count is not a time.** The fix that brought `from_signals` back under the
limit showed this in both directions, measured against main on an Apple M4 machine,
with eight interleaved single-thread rounds per comparison (a range is two
sittings or more):

| Change | `from_orders` instructions | `from_orders` time | `from_signals` instructions | `from_signals` time |
|---|---:|---:|---:|---:|
| Before the fix | +4.2% | +3.2% to +4.0% | +1.1% | +4.5% to +4.6% |
| Two copies of each loop, with and without cash-flow code | +4.5% | −1.9% | −3.5% | −1.8% |
| Plus the kernel's rare branches out of line, `buy`/`sell` inlined | +0.5% | −3.7% to −4.1% | −4.5% | −2.0% to −2.4% |

Splitting the loops barely moved `from_orders`' instruction count and made it
about 5% faster: the branches it removed are on flags fixed for the run and
never taken, so they cost no work, and still cost time. The kernel change
alone, without the split, went the other way: it took `from_orders` back to
+0.2% instructions but its time down by only 0.4 to 2.7 points, and
`from_signals` got slower in two sittings out of four. This core retires about
six instructions a cycle on these loops, so where the code sits and what it
keeps in registers can matter more than how much of it runs.

So use the count to see a change, and the interleaved timings to decide
whether it matters. A count that grows means the loop does more work, which
is worth knowing before any timing. A flat count clears nothing, and the
acceptance is on time.

## Adding a workload

Add a row to `workloads.tsv`. The runner picks it up; nothing else needs
changing. Then regenerate the baseline, since the gate reports unknown entries rather than
silently skipping them.

Workloads are shaped to separate things that behave differently: fill rate
(dense runs are memory-bound on record writing, sparse ones are compute-bound),
aspect ratio (one long column versus many short ones), and per-bar work
(`TargetPercent` revalues every bar; `Amount` does not).
