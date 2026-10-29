# Claude's independent positions (written before reading Codex round A)

Scoping (scratchpad/scoping/scope.py, unreviewed; 5-fold StratifiedGroupKFold by uri, 3 repeats, empty row dropped):
- Full features (13 audio + year, month, day, date precision, days since epoch): RF 0.826, LGBM slow 0.827, LGBM default 0.822, HistGB 0.820, ET 0.820, XGB default 0.808, LR on log features 0.781. SE of the mean about 0.003.
- Audio only: 0.73 to 0.76. Date features add about +0.07 to +0.08 AUC: the largest lever.
- Ungrouped folds inflate AUC by +0.016 (RF and LGBM): grouping by uri is mandatory.
- Adversarial validation train vs test: AUC 0.529; year quantiles identical. Test looks iid with train: random grouped CV is the right proxy; no time split needed for the test, though a by-era diagnostic is useful for the report.
- Row order and test ID carry no signal (row-index AUC 0.506; |corr| < 0.07).
- 72% of test rows share an exact release date with some train row, but the OOF date target encoding alone (0.631) is below year alone (0.640): the date is mostly a time signal, not an album key. Medium priority as a feature.
- Public leaderboard noise: with 863 test rows and a public share of 30 to 50%, the Hanley-McNeil SE of AUC near 0.83 is about 0.022 to 0.029. Public-LB differences below about 0.03 are noise for model selection.

Positions:
1. Validation: StratifiedGroupKFold(uri), 5 folds x 3 repeats for screening, x 5 for finalists; every transform inside folds; OOF predictions banked for every model (one file per model, fixed fold ids) to feed ensembling. Paired comparison on identical folds; adopt only if the paired gain exceeds 2 SE and 0.002.
2. Strongest bets: (a) GBDT tuning with small leaves and strong regularization, CatBoost; (b) RF/ET, which already match GBDT here (small, noisy data); (c) foundation models: TabICLv2 (BSD, pip, CPU feasible at 3.9k rows) and TabPFN-2.5/3/3.5 (gated weights, Sam must accept the license; Colab GPU), which lead TabArena on small data; (d) hill-climbing ensemble on OOF, rank-averaged; (e) seed averaging and refit on full data.
3. Features: year and finer date parts, date precision flag, days since epoch, era bins; log transforms for skewed features (linear models only); key as cyclic or one-hot, mode x key; interactions only if trees miss them (EBM pairwise detection); counts of uri and date within train+test (transductive, allowed by the ruling); OOF target encoding of date (medium).
4. Duplicates and noise: dedupe by uri with the mean label as a soft target or sample weight vs keep as is; test both; conflicting labels treated as label noise.
5. Interpretability for the report: grouped permutation importance on OOF, SHAP with sign and dependence plots, EBM shape functions, fold stability of the top 10.
6. Gray areas to ask on Piazza: pretrained tabular foundation models (trained on synthetic data; RealTabPFN-2.5 also fine-tuned on real datasets) and LLM-generated features (CAAFE style). Default: allowed only after an instructor answer.
7. AutoGluon 1.6 best_quality on CPU as a strong reference and ensemble member; extreme preset needs a GPU (Colab).
8. Final two selections: (1) the best CV ensemble; (2) a decorrelated hedge, e.g. the best single strong model or an ensemble of a different family, both chosen on CV, not on the public LB.
