# implementer · EXP-007

- **Fecha:** 2026-10-13
- **Veredicto:** ok

add_har_features() en inversion/models/volatility.py: rv_d/rv_w/rv_m (log varianza realizada 1/5/22) + ratios. Integrado via _prepare_frame en train y predict. Retrain real 7 tickers mejora R2 en 7/7 (0.200->0.212). 2 tests nuevos. references/06 escrita. init.sh: 216 passed.
