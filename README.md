# PGB-AC: evaluation data and code

Processed experimental data, evaluation code and chain-measurement summaries for the manuscript
*Blockchain Access Control for Multi-Center Securities Operations: Trust Root Anchoring and Policy Governance*,
submitted to Scientific Reports.

## Contents

| Path | What it holds |
| --- | --- |
| `experiments/*.py` | Evaluation code. `pgbac_core.py` carries the admission, trust-aggregation and anchoring model; the other scripts produce the result tables. |
| `experiments/results/*.csv` | Processed results, one file per reported table. |
| `experiments/results/frozen_config.json` | Frozen parameter set used by every reported run. |
| `experiments/results/figures_sr39/` | The eight manuscript figures at 600 dpi, plus the manifest that records pixel size, print size and caption for each. |
| `experiments/results/figures_svg_src/` | Vector sources for the figures. All in-figure text lives here. |
| `experiments/results/figures_sr35/en/` | Five raster plates that the figure build reuses unchanged. |
| `_tools/make_figures_sr39.py` | Builds `figures_sr39` from the vector sources. |
| `_tools/render_svg.py` | SVG rasterisation helper used by the figure build. |

## Environment

Python 3.13, NumPy and Matplotlib:

```
python -m pip install -r requirements.txt
```

The reported runs used NumPy 2.5.3 and Matplotlib 3.11.2. Rebuilding the figure set also needs a local
Chrome or Chromium binary. Set `CHROME_EXE` to its path when it is not in a standard location.
`experiments/figstyle.py` resolves a CJK font for the plate legends and looks first in the Windows font
directory; on other platforms edit the candidate list at the top of that file.

## Running the evaluation

Each script writes its table into `experiments/results/` and prints a summary of the numbers it wrote.

```
python experiments/baseline_summary.py       # scheme-level comparison
python experiments/threshold_streams.py      # threshold selection across calibration streams
python experiments/decision_merge_check.py   # constraint-set merge and replay cases
python experiments/run_experiments.py E1 E2  # latency, throughput and interval checks
```

`run_experiments.py` takes experiment identifiers as arguments and accepts `--quick` for a reduced
sweep. The remaining scripts follow the same pattern: one script, one table, run from the repository root.

These four commands were rerun on a clean copy of this repository. The resulting tables match the
committed files byte for byte, with one exception noted below.

## Measurement conditions

Absolute wall-clock readings depend on the host. `E10_方案级基线对比.csv` was collected on the machine
used for the submission, and rerunning the script on a different host changes its latency columns while
leaving gas and throughput unchanged. The latency figures quoted in the manuscript come from the
final-build runs summarised in `E1_决策时延与吞吐.csv` and `E1_跨运行分位汇总.csv`, reported as
three-run medians over four offered request rates. Use `E10` for the relative comparison across schemes
on one host, and `E1` for the quoted latency values.

Chain costs in `E8_*.csv` and `E9_*.csv` are measured on a four-node consortium chain with the same
client, SDK and submission path for every route.

## Rebuilding the figures

```
python _tools/make_figures_sr39.py
```

The script writes eight PNG files and `manifest.json` into `experiments/results/figures_sr39/en/`.
It renders from the SVG sources and performs no string substitution at run time, so a change to figure
text belongs in the corresponding SVG.

## Data not included

Every reported evaluation uses constructed securities request and policy-tampering cases, generated inside
the evaluation code from the frozen configuration, together with numerical model checks and a benchmark on
a four-node consortium chain operated for this study. No customer records, employee records or production
ledger data were used, so none is distributed here.

## License

Code in `experiments/`, `_tools/` and any accompanying scripts is released under the MIT License; see
`LICENSE`. Result tables, figure sources and rendered figures are released under Creative Commons
Attribution 4.0; see `LICENSE-DATA`.

## Citation

Cite the manuscript while it is under review. `CITATION.cff` carries the metadata and is updated when the
article appears.
