"""Step 0: build the stock x manager ownership matrices from the raw WRDS files in data/.

For each 13F report date in config.yaml:
  stage 1  S&P 500 universe on that date (common shares only)
  stage 2  13F holdings -> ownership matrix W (stocks x managers); flags passive managers
  stage 4  stock controls: GICS sector, market equity, book-to-market

Writes caches to results/pipeline/cache/ and a log of every filter and check to results/pipeline/log.md.

Run: python prepare_data.py
"""
import warnings

from src import stage1_universe, stage2_holdings, stage4_controls
from src.common import load_config, start_log

warnings.filterwarnings("ignore", r".*encountered in (matmul|dot)", RuntimeWarning)


def main():
    cfg = load_config()
    start_log(cfg, "Data preparation: stages 1, 2, 4")
    for t in cfg["dates"]:
        u = stage1_universe.run(t, cfg)
        r = stage2_holdings.run(t, cfg)
        stage4_controls.run(t, cfg)
        print(f"{t}: {len(u)} stocks in universe; W = {r['W'].shape[0]} stocks x {r['W'].shape[1]} managers "
              f"({len(r['passive'])} passive, {r['W_nopassive'].shape[1]} active)")


if __name__ == "__main__":
    main()
