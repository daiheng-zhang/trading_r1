# Trading-R1 (Paper-Faithful Research Scaffold)

This package implements a paper-faithful Trading-R1 research pipeline in this repo:
- Multi-modal data pipeline (`price + news + fundamentals`)
- Volatility-adjusted 5-class labels
- Distillation artifacts (`SFTTarget`, `GRPOBatchItem`)
- Stage-wise SFT and GRPO interfaces
- Backtest-only runtime with CR/SR/HR/MDD

## Layout
- `configs/`: run configs
- `src/trading_r1/`: implementation
- `tests/`: unit + integration smoke tests
- `artifacts/`: generated outputs

## CLI
Run from repo root:

```bash
python -m trading_r1 collect-data --config trading_r1/configs/data.yaml
python -m trading_r1 make-labels --config trading_r1/configs/labels.yaml
python -m trading_r1 build-samples --config trading_r1/configs/data.yaml
python -m trading_r1 distill-sft --config trading_r1/configs/distill.yaml
python -m trading_r1 train-sft --config trading_r1/configs/train_stage1_sft.yaml
python -m trading_r1 train-grpo --config trading_r1/configs/train_stage1_grpo.yaml
python -m trading_r1 infer --config trading_r1/configs/infer.yaml --date 2024-07-15 --ticker AAPL
python -m trading_r1 backtest --config trading_r1/configs/backtest.yaml
```

## Split Policy
Fixed policy from plan:
- Train: `2024-01-02..2024-05-31` and `2024-09-03..2025-03-31`
- Validation: `2025-04-01..2025-05-31`
- Backtest holdout: `2024-06-03..2024-08-30`

## Notes
- GRPO defaults use paper values (`G=8`, `clip=0.2`, `kl_beta=0.03`).
- Stage reward weights are locked to:
  - Stage I: `(1.0, 0.0, 0.0)`
  - Stage II: `(0.4, 0.6, 0.0)`
  - Stage III: `(0.3, 0.5, 0.2)`
- Training modules provide `mock` mode for local smoke tests.
- Replace `mock` with real backends in configs for full multi-GPU runs.
