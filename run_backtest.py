"""Backtest 2010/2014/2018/2022 (feature model vs FIFA baseline) -> CSV."""

import logging

import worldcup_2026_sim as wc


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    harness = wc.BacktestHarness(iterations=100, workers=3, base_seed=2026, use_llm=False)
    df = harness.run([2010, 2014, 2018, 2022])
    df.to_csv("out_backtest.csv", index=False)
    print(df.to_string(index=False))
    print("saved -> out_backtest.csv")


if __name__ == "__main__":
    main()
