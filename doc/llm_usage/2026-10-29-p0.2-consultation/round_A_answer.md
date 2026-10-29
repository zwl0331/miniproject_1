**Status: complete consultation draft, with supplemental technical review passed. Required Claude approval remains pending because its CLI is not logged in. This is not yet an approved project work product. No files were modified.**

My recommended route is a carefully validated ensemble of regularized boosted trees, a smooth model, and at most one or two neural challengers. Release date deserves a serious ablation. Duplicate-aware validation is mandatory. Pretrained tabular models could be valuable at this sample size, but require a course ruling before use.

This is a plan, not a measured model comparison. I ran data audits, not a model tournament. Runtime budgets and AUC gains below are estimates. Research was checked on October 29, 2026. No project file was modified.

**1. Problem reading and verified evidence**

I read AGENTS.md, doc/review_policy.md, doc/README.md, doc/project_plan.md, doc/findings.md, and the assignment. The assignment rewards ROC AUC. For continuous scores, AUC measures ranking: the probability a randomly selected High row outranks a Low row, with half credit for ties. Calibration and selecting a classification threshold are separate concerns. Submit the probability of High, explicitly mapping High to 1 and Low to 0. Alphabetical class ordering can otherwise silently select Low.

The PDF's original audio-only wording is superseded by the documented October 28 Piazza ruling allowing all CSV columns. External song lookup, additional task data, and existing modeling code for this Spotify dataset remain forbidden. The public/private split mechanism and sizes are not documented in the files I read.

Verified data from [train.csv](/Users/zwl0331/Desktop/Mini_project_1/data/train.csv), [test.csv](/Users/zwl0331/Desktop/Mini_project_1/data/test.csv), and [sample_submission.csv](/Users/zwl0331/Desktop/Mini_project_1/data/sample_submission.csv):

| Check | Result |
|---|---|
| Dimensions | Train 3,864 × 19; test 863 × 19; sample submission 863 × 2 |
| Target | 1,359 High; 2,505 Low; 35.1708% High |
| Distinct train tracks | 3,642 nonmissing URIs, plus one row missing URI. Thus 3,643 groups if that row is a singleton |
| Train duplicates | 203 repeated URI groups, 424 rows, 221 excess copies, maximum multiplicity 3 |
| Conflicting labels | 32 URI groups, containing 66 rows |
| Within-URI features | All 13 audio features and the release-date string agree within each nonmissing URI |
| Test duplicates | 852 URIs; 11 groups of two rows |
| Train/test URI overlap | Zero |
| Other exact matches | Zero shared audio-plus-date signatures; two shared audio-only signatures, each involving one train and one test row, with different dates |
| Train signature counts | 3,638 distinct audio-plus-date signatures; 3,630 audio-only signatures |
| Missing data | One train row lacks all audio features and identifiers; its date is 2024-09-08 and its label Low. Test has no missing cells |
| Date formats | Train: 3,752 full dates, 111 year-only, one year-month. Test: 840 full dates, 23 year-only |
| Identifier redundancy | The terminal tokens in track_href and analysis_url equal the URI track token on all nonmissing rows |
| Constant feature | type is audio_features wherever present |
| Submission alignment | ID is unique; sample IDs exactly match test IDs in order. Values range from 2 to 967, so do not fabricate 0…862 IDs |

Univariate AUCs use increasing raw feature value as the score, exclude the single missing row for audio, and use all rows for year. These are descriptive full-training statistics, not cross-validated model performance.

| Feature | AUC | Marginal association |
|---|---:|---|
| loudness | 0.680513 | Higher tends toward High |
| instrumentalness | 0.325290 | Lower tends toward High |
| acousticness | 0.342188 | Lower tends toward High |
| release year | 0.359351 | Earlier tends toward High |
| energy | 0.628197 | Higher tends toward High |
| duration_ms | 0.607460 | Higher tends toward High |
| valence | 0.571335 | Higher tends toward High |
| danceability | 0.545023 | Weak positive |
| tempo | 0.543574 | Weak positive |
| key | 0.518868 | Numeric ordering is musically arbitrary |
| liveness | 0.513753 | Very weak positive |
| mode | 0.512269 | Very weak positive |
| time_signature | 0.503173 | Nearly null marginally |
| speechiness | 0.496031 | Nearly null marginally |

AUC below 0.5 is useful inverse signal, not evidence a feature is useless. Weak marginal signal can also conceal nonlinear effects and interactions.

| Release era | Train rows | Train High fraction | Test rows |
|---|---:|---:|---:|
| Before 1990 | 134 | 1.0000 | 31 |
| 1990s | 223 | 0.5022 | 38 |
| 2000s | 316 | 0.5190 | 58 |
| 2010s | 810 | 0.4222 | 183 |
| 2020s | 2,381 | 0.2549 | 553 |

My interpretation is that both audio properties and dataset selection by release era drive signal. The old-song pattern might reflect selection of enduring popular songs, but that mechanism is unverified. Do not infer a causal effect of age or hard-code every old song to probability 1.

The preliminary observations overstate evidence for train/test date drift. Median release year is 2022 in both sets. The year KS statistic is 0.0362, nominal p=0.3043; the largest of the 13 audio KS statistics is 0.0431 for acousticness, nominal p=0.1398. These dependent-row, univariate checks neither prove identical distributions nor exclude joint or conditional shift. A concrete difference is duplicate prevalence: 10.9731% of train rows versus 2.5492% of test rows belong to repeated URI groups.

Main risks are duplicate leakage, inconsistent labels, small effective sample size, collection-related feature associations, unknown private-set sampling, and repeated tuning to noisy CV or public scores. No artist IDs are supplied, so URI grouping cannot guarantee independence between songs by the same artist.

Commands used included:
~~~sh
cat AGENTS.md doc/review_policy.md doc/README.md
cat doc/project_plan.md doc/findings.md
pdftotext -layout Project_1.pdf -
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c '<in-memory pandas/scikit-learn audit>'
~~~

This compact executable reproduction covers the principal counts, AUCs, era tables, missingness, identifier checks, hashes, and KS results:
~~~sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c '
import hashlib, numpy as np, pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
tr=pd.read_csv("data/train.csv"); te=pd.read_csv("data/test.csv")
s=pd.read_csv("data/sample_submission.csv")
a=tr.select_dtypes("number").columns.tolist()
dt="track_album_release_date"; y=tr.Popularity_Type.eq("High")
print(tr.shape,te.shape,s.shape,tr.Popularity_Type.value_counts(),y.mean())
print("submission",te.ID.equals(s.ID),te.ID.nunique(),te.ID.min(),te.ID.max())
for name,d in [("train",tr),("test",te)]:
    g=d.groupby("uri"); n=g.size()
    print(name,"URI",d.uri.nunique(),"missing",d.uri.isna().sum(),
          "repeats",(n>1).sum(),n[n>1].sum(),"excess",(n-1).sum(),"max",n.max())
    print("varying groups",g[a+[dt]].nunique(dropna=False).gt(1).any(axis=1).sum())
    print("nulls",d.isna().sum().to_dict(),"date lengths",d[dt].str.len().value_counts().to_dict())
    print("type",d.type.dropna().unique(),"repeat fraction",d.uri.duplicated(False).mean())
    tok=d.loc[d.uri.notna(),"uri"].str.rsplit(":",n=1).str[-1]
    print("IDs",[d.loc[d.uri.notna(),c].str.rsplit("/",n=1).str[-1].eq(tok).all()
                 for c in ["track_href","analysis_url"]])
    era=pd.cut(d[dt].str[:4].astype(int),[-np.inf,1989,1999,2009,2019,np.inf])
    z=pd.DataFrame({"era":era,"high":d.Popularity_Type.eq("High") if name=="train" else np.nan})
    print(z.groupby("era",observed=True).high.agg(["size","mean"]))
c=tr.groupby("uri").Popularity_Type.nunique().gt(1)
print("conflicts",c.sum(),tr.uri.isin(c[c].index).sum())
print("empty",tr.loc[tr[a].isna().all(axis=1),[dt,"Popularity_Type"]])
print("URI overlap",len(set(tr.uri.dropna())&set(te.uri.dropna())))
for cols in [a,a+[dt]]:
    h=pd.util.hash_pandas_object(tr[cols],index=False)
    j=pd.util.hash_pandas_object(te[cols],index=False)
    print("signatures",len(cols),h.nunique(),len(set(h)&set(j)))
for col in a+["year"]:
    v=tr[dt].str[:4].astype(float) if col=="year" else tr[col]
    w=te[dt].str[:4].astype(float) if col=="year" else te[col]
    ok=v.notna()
    print(col,roc_auc_score(y[ok],v[ok]),v.median(),w.median(),ks_2samp(v[ok],w.dropna()))
for p in ["data/train.csv","data/test.csv","data/sample_submission.csv"]:
    print(p,hashlib.sha256(open(p,"rb").read()).hexdigest())
'
~~~

Input SHA-256:
~~~text
train.csv             d14b372deff47aa791242875ac5c134ce2497695bb190479b159b8ec49d68407
test.csv              70f1502ccae76396b45961a0e0c9f96c5ba213d5b99a101015c32ddc65d020e3
sample_submission.csv 35afd1abf23cb89279d1e10ef4d12d27b86d4406ae9b56984c1725b674835ca8
~~~

**2. Method catalog and priorities**

Costs below are planning estimates for this small table, not measured timings. “High priority” means test early, not guaranteed superiority. All ordinary methods below are permissible when fitted only to the provided training data, with fold-local preprocessing and no dataset-specific borrowed code. Exceptions and gray areas are stated explicitly.

| Preparation or feature family | Why consider it; priority and cost | Risks and implementation |
|---|---|---|
| Schema, missingness, constants | Essential; minutes | Explicit target mapping; drop constant type; preserve ID for output. Retain the incomplete row initially with native missing handling or fold-fit median imputation plus missing flags; test dropping it only as a training ablation |
| Dates | High value; cheap | Year first; then year-month ordinal, month, date precision, day-of-year or weekday for full dates only. Missing date components stay missing; do not pretend year-only songs were released January 1 |
| Age and era | High value; cheap | Fixed-reference age is a reparameterization of year, not independent information. Use coarse eras or smooth year splines; audit very recent dates separately. Avoid tuning a pre-1990 rule on the full labels |
| Audio transforms | Medium to high for smooth/distance models; cheap | log(duration), log1p transforms of strongly skewed audio values, zero indicators for instrumentalness, clipped logit for bounded features. Compare standard, robust, and quantile scaling. Fit learned transformations inside folds |
| Musical categories | Medium; cheap | Treat key, mode, and time signature as categories for linear/distance models. Compare key one-hot to cyclic encodings and key×mode. Pitch-class and circle-of-fifths embeddings impose different assumptions |
| Interactions | Medium; cheap if curated | Year×loudness, year×acousticness, energy×loudness, energy×danceability, acousticness×instrumentalness, tempo×duration. Approximate beat count = tempo×duration_ms/60000. Add safe ratios sparingly; tree models already learn interactions |
| Era-relative features | Medium exploratory | Fold-fit residuals such as loudness minus a smoothed era mean may separate production era from sonic characteristics. Sparse-era estimates need shrinkage |
| Target encodings | Low to medium | Smoothed year/era/date-category encodings can help, but require group-aware inner cross-fitting for training rows and mappings fitted only on outer training rows for validation. Do not use global label averages |
| Frequency/density encodings | Low to medium | Counts of dates, bins, clusters, or neighborhoods might represent collection structure. Define train and inference semantics consistently; unseen categories need a fallback |
| Raw identifiers | Low expected value | Use URI for grouping. The other identifier columns duplicate it. Exact-URI target encoding cannot transfer to unseen test URIs. Prefix/character/hash features are a small falsification experiment, not presumed metadata |
| Duplicate-group features | Low; easy to misuse | Multiplicity differs between train and test, and a held-out URI has zero training occurrences. Do not give training rows full-corpus counts while validation rows get unseen-category counts |
| PCA, clustering, local neighbors | Medium diversity experiment; minutes | Fit scales, PCA and clusters on training folds only. Compare distance-to-centroid and neighbor features. Supervised neighbor target means must exclude the entire row's group |
| Automated feature search | Low after curated ablations; hours | Polynomial screening, symbolic regression, feature selection, or a bounded LLM-proposed transformation list. Select within training folds; cap proposals. No song identities, external metadata, or LLM guesses about songs |
| External/audio/text augmentation | Excluded | No Spotify API, web lookup, lyrics, artist metadata, embeddings from retrieved audio, or extra Spotify datasets |

A practical feature ladder is F0 = 13 audio features; F1 = F0 + year; F2 = F1 + date precision and partial-date-aware calendar features; F3 = F2 + a small documented transform/interaction set. Evaluate blocks, then remove unhelpful additions. Avoid searching hundreds of individual columns against the same validation labels.

| Model family | Potential value here | Cost, risks, and priority |
|---|---|---|
| Logistic regression, ridge/elastic-net classifier | Stable global baseline and ensemble component | Seconds to minutes. Scale numeric features; one-hot categories; tune regularization. Include transformed features and spline variants |
| GAM, spline logistic regression, EBM | Smooth nonlinear response with explainable shapes | Minutes to tens of minutes. High priority. Limit spline degrees and EBM interactions; excellent report candidate |
| kNN, kernel smoothing | Tests whether local audio neighborhoods carry signal | Minutes. Tune k around 15–150, distance weighting, scaling and feature subsets. Duplicates and irrelevant dimensions can dominate |
| Linear/RBF SVM, kernel logistic, approximate kernels | Smooth nonlinear boundary for a few thousand rows | Minutes to tens of minutes. RBF SVM is a worthwhile challenger. Tune C and gamma on log scales; calibrate using group-safe inner splits if probabilities are needed |
| Gaussian naive Bayes, LDA/QDA | Cheap diagnostic diversity | Seconds. Correlated bounded audio features violate assumptions. Shrink covariance for QDA/LDA; low expectation, minimal budget |
| Random forest and Extra Trees | Robust interactions and different errors from boosting | Minutes to tens of minutes. Try 500–1,000 trees, leaf sizes 2–30, feature subsampling; avoid selecting by ordinary OOB score because duplicates can leak |
| LightGBM | Fast, flexible principal contender | High priority. Small leaves or weak regularization can overfit this sample |
| XGBoost, including a linear booster baseline | Strong regularized interactions; useful diversity | High priority for histogram tree booster. CPU suffices |
| CatBoost | Strong small-table contender even without rich categories | High priority; justify installation. Compare Plain and Ordered boosting, depths 3–7, numerical versus selected categorical date/music inputs |
| sklearn HistGradientBoosting | Low dependency cost, strong controlled baseline | High priority early, lower tuning priority if dominated. Disable automatic random validation and supply a group-safe stopping strategy |
| AdaBoost, classical gradient boosting, rule lists/RuleFit, BART or Gaussian-process classification | Further structural diversity or interpretable alternatives | Low priority. AdaBoost is noise-sensitive; exact GPs can be expensive at this n; BART adds dependency and tuning costs. Small diagnostic budgets only |
| MLP variants | Smooth complementary predictions | Medium priority. Try small 2–3-layer networks, 64–256 units, AdamW, weight decay, early stopping and 3 seeds. Compare quantile versus standard scaling |
| ResNet, FT-Transformer, numerical embeddings | More expressive tabular neural baselines | Lower priority than strong MLPs at n≈3,600. CPU experiments can take hours; use Colab if needed. Large attention models are easy to overfit |
| RealMLP | Strong designed defaults may save tuning effort | One early neural challenger; minutes to hours on CPU, benchmark a fold. Use implementation with clear preprocessing and early-stopping controls |
| TabM / TabM-mini | Parameter-efficient ensembles may improve stability and blend diversity | Preferred second neural challenger. Small numerical embeddings are worth an ablation; use a short CPU pilot or Colab |
| ModernNCA / TabR | Learned neighborhood geometry may exploit audio similarity | Medium exploratory priority after RBF SVM. More training/inference overhead and leakage-sensitive neighbor construction |
| TabNet, NODE, SAINT, DCNv2, AutoInt, DANets, tree/neural hybrids | Broader architectural alternatives | Low priority for this deadline. Keep only if cheap defaults materially improve the established ensemble |
| Tabular foundation models and ICL | Potentially strong with a few thousand labeled rows | Course approval required. Start with TabPFN-3.5/Fast and TabICLv2, then one diverse compact model. CPU minutes to hours; Colab likely more practical |
| AutoML | Broad search and automatic ensembles | AutoGluon is worthwhile for a bounded 1–2-hour challenge run. FLAML is a lighter CPU alternative; TPOT/auto-sklearn/PyCaret add installation or search overhead. Must enforce group splits and disable unapproved pretrained models |

Suggested initial boosting searches, expanded only for promising families:

| Model | Search region |
|---|---|
| LightGBM | num_leaves 7–31 initially, up to 63 only if justified; min_child_samples 20–150; learning_rate 0.02–0.08; L1 0–5, L2 0.1–30; feature_fraction 0.7–1; bagging_fraction 0.7–1 with bagging_freq > 0 |
| XGBoost | max_depth 2–5; min_child_weight 2–30; learning_rate 0.02–0.08; subsample/colsample 0.7–1; reg_lambda 1–50; reg_alpha 0–5; gamma 0–2; tree_method=hist |
| CatBoost | depth 3–7; learning_rate 0.02–0.08; l2_leaf_reg 3–30; random_strength 0–2; compatible bootstrap settings; loss Logloss and selection metric AUC |
| HistGB | max_leaf_nodes 7–31; min_samples_leaf 20–100; l2_regularization 0–20; learning_rate 0.03–0.1 |

Use generous iteration caps, around 2,000–4,000, with group-safe inner early stopping rather than assuming the cap is optimal. CatBoost Ordered processing does not make duplicate leakage disappear. LightGBM's documented bagging settings require both a fraction below one and nonzero frequency. [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html), [CatBoost parameters](https://catboost.ai/docs/en/references/training-parameters/common).

Use random search first, then Optuna/TPE if worthwhile. Allocate roughly 20–30 three-fold trials to each of the two leading tree families, followed by 10–20 local refinements total. Hold validation splits fixed across configurations. Halving/pruning is optional, but premature pruning can penalize slower-learning configurations. Record failed trials and preprocessing, not just the winning hyperparameters.

**Duplicates, noise, imbalance, and ensembles**

Keep contradictory labels initially. They may reflect popularity changing over time or a construction artifact; neither cause is established. Never relabel by majority vote or delete conflicts simply because a model dislikes them.

Compare ordinary row weighting with weights 1/group_size. Score both using unweighted validation rows because Kaggle scores rows. Also report a group-weighted sensitivity result. Collapsing identical inputs to their mean label is equivalent for additive cross-entropy only when using the appropriate multiplicity weight and a learner that supports fractional targets. Standard classifiers do not uniformly support this. Retaining weighted original rows is the safer implementation.

Use cross-fitted disagreement or a label-quality tool only as a diagnostic. Any removal/downweighting policy must be learned within training folds, with original validation labels untouched. Label smoothing or robust losses are optional low-priority ablations.

At 35% positives, start without class balancing. Test modest positive weights only if repeated AUC improves. SMOTE, downsampling, focal loss, and pairwise AUC surrogates are lower priority: they can amplify noise, distort neighborhoods, or add variance. Optimize log loss for fitting and use AUC for model selection. Calibration is useful for probability quality and blending, but a strictly increasing calibration transformation alone cannot improve a single score vector's AUC.

Start ensembling with equal probability averages of two to four complementary models. Then try coarse nonnegative weights, constrained to sum to one. Test rank averaging separately; rank outputs in [0,1] are ranking scores, not calibrated probabilities. If retaining a rank ensemble, document this distinction and, if probability calibration is desired, assess a group-safe calibrated version.

Hill-climbing ensemble selection is worth a short trial on a small diverse OOF library. Reject hundreds of tiny weight tweaks or negative-weight searches on this sample. A regularized logistic level-2 model on clipped probabilities/logits is the first stacker; a shallow tree stacker is a later, higher-risk option. Training a stacker on OOF predictions and scoring it on those same rows is in-sample evaluation of the stacker. Use nested group cross-fitting or an untouched selection set to evaluate the whole stack.

**Course-rule gray areas**

| Technique | Recommended treatment |
|---|---|
| Pretrained TabPFN, TabICL, Mitra, other TFMs | Ask whether pretrained weights count as external data, distinguishing synthetic-only pretraining from real-data pretraining or fine-tuning. Do not assume either is permitted |
| AutoML presets | Use an explicit model whitelist. Modern presets may invoke TFMs or other pretrained components automatically |
| Generic LLM help with code/formulas | Permitted subject to disclosure; retain only transformations of supplied columns. Inferring track identity, retrieving facts, or injecting memorized song popularity is prohibited |
| Unlabeled-test normalization, PCA, frequency fitting, density weighting | Ask separately. “All columns may be used” does not settle whether test rows may train preprocessing |
| Pseudo-labeling, self-training, label propagation, graph learning, test-set autoencoders | Ask explicitly because the assignment says training/validation use the provided training set. If approved, simulate transduction inside CV by hiding validation labels; never use those labels for pseudo-label generation |
| Training-only self-supervision or augmentation | Generally fits the provided-data rule; low value here unless a controlled ablation supports it. Do not generate synthetic labels using external song knowledge |

The exact Piazza question I would propose is: “Are pretrained tabular predictors permitted, including weights pretrained only on synthetic tables and weights pretrained/fine-tuned on real tables? Separately, may supplied unlabeled test features be used to fit unsupervised preprocessing or pseudo-label training?” This is a proposed question; I have not posted it.

**3. Current research and what it changes**

These are live-checked primary sources, not recalled rankings. Research comparisons below are the authors' reported results. I have not reproduced them on this dataset. The live TabArena interface did not expose its full table to the browsing tool, so I do not claim an independently verified October 29 overall leaderboard order.

| Source and date/venue | Verified contribution; implication |
|---|---|
| [TabArena](https://arxiv.org/abs/2506.16791), June 2025, NeurIPS 2025 Datasets and Benchmarks | A living benchmark emphasizing validation, tuning budgets and ensembles. Its study supports considering trees, modern neural models and foundation models rather than declaring a universal winner |
| [TALENT](https://jmlr.org/papers/v26/25-0512.html), JMLR 26(226), 2025 | Unified toolbox with more than 35 deep methods. Useful for consistent implementations; a benchmark average is not a recommendation for every small table |
| [RealMLP](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2ee1c87245956e3eaa71aaba5f5753eb-Abstract-Conference.html), NeurIPS 2024 | Strong meta-tuned defaults. Supports trying a well-designed MLP before a large transformer |
| [TabM](https://arxiv.org/abs/2410.24210), ICLR 2025 | Efficient MLP ensembling. Supports a compact neural challenger, not an assumption of superiority over trees |
| [ModernNCA](https://proceedings.iclr.cc/paper_files/paper/2025/file/ecd294194d5155432aab533f7ce0f6fe-Paper-Conference.pdf), ICLR 2025 | Modern learned neighborhood classification. Relevant to continuous audio features and a manageable training context |
| [TabPFN-2.5](https://arxiv.org/abs/2511.08667), November 2025 preprint | Expanded scale and strong reported comparisons. It is no longer the latest release |
| [TabICLv2](https://arxiv.org/abs/2602.11139), February 2026; official repository identifies ICML 2026 | Synthetic-prior, architectural and training improvements. [Released implementation](https://github.com/soda-inria/tabicl) supports classification, regression and fine-tuning; a practical approved-TFM candidate |
| [TabPFN-3.5](https://arxiv.org/abs/2609.17895), September 15, 2026, revised September 22 | New flagship plus Fast and higher-compute variants; reports broader benchmark improvements. [Official code](https://github.com/PriorLabs/TabPFN) documents CPU limits of 5,000 rows for current models and Apple Silicon support. Pin checkpoint and package rather than relying on mutable defaults |
| [Mitra-v2](https://arxiv.org/abs/2609.04540), September 3, 2026 | Reports a compact 77M-parameter synthetic-only model and strong TabArena/TALENT results. An alternative if permitted, with its own runtime pilot |
| [EXAONE Tabular](https://arxiv.org/abs/2608.25774), August 26, 2026 | Compact synthetic-pretrained ICL with strong reported efficiency. Worth screening if installation and course approval are straightforward |
| [LimiX-2](https://arxiv.org/abs/2609.17488), September 15; [Causilo](https://arxiv.org/abs/2609.22866), September 19; [TabFM](https://arxiv.org/abs/2609.37959), September 29, 2026 | Further current foundation-model candidates. Verify released weights, license, training provenance and memory needs before investing. Do not turn every new preprint into a deadline-critical dependency |
| [Test-Time Compute for TFMs](https://arxiv.org/abs/2610.12005), October 8, 2026 preprint | Reports benefits from selective aggregation/adaptation, but diminishing returns and inconsistent context-expansion gains. Supports modest diverse ensembles before expensive fine-tuning |
| [Kaggle S5E10 first-place account](https://www.kaggle.com/c/playground-series-s5e10/writeups/1st-place-i-think-it-was-genetic-programming), October 31, 2025 | Author explicitly says TabM helped an ensemble. This is an anecdote from a different regression competition, not an expected AUC gain here |
| [S5E10 14th-place account](https://www.kaggle.com/competitions/playground-series-s5e10/writeups/s5e10-14th-place-solution), October 31, 2025 | Documents diverse OOF models, hill climbing and ridge stacking, but also overfitting from aggressive optimization. Transfer the validation discipline, not its original-data augmentation or large model count |

Older or additional TFM families such as TabDPT, Real-TabPFN, ConTextTab and retrieval-augmented ICL belong on the longlist. I would not spend deadline time evaluating every lineage. Models involving real-data pretraining have an especially clear need for an instructor ruling. Likewise, benchmark-tuned defaults are distinct from importing learned pretrained weights, but disclose unusual meta-learning dependencies if in doubt.

For any approved TFM, first try raw numerical audio plus parsed dates, with identifiers removed. Compare audio-only and date-augmented variants; do not mechanically apply the linear-model scaler or one-hot pipeline. Evaluate 1–2 inference ensemble settings, then blend. Fine-tuning, adapters, feature-view ensembles and neighbor retrieval come only after a reproducible default beats or complements trees. At this n, full training context is feasible for some models; retrieval is not automatically necessary.

AutoGluon's current documentation makes an important distinction: its “groups” column denotes leave-one-group-out bagging folds. Supply precomputed fold IDs, not thousands of URIs, and verify every internal split. Use positive_class=1 after explicit target encoding. [AutoGluon TabularPredictor documentation, checked October 29, 2026](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html).

**4. Validation aimed at the private leaderboard**

No validation protocol can guarantee the private score without knowing the sampling scheme. My default assumes unseen tracks drawn from approximately the same collection, with release year as a predictor rather than the time at which labels were observed.

1. Define groups before fitting. The main grouping should connect identical URIs and exact audio-plus-date signatures across train rows. Give missing-ID rows a safe singleton unless their complete observed signatures establish a match. Add a stricter sensitivity split connecting exact audio-only matches, since two such signatures cross the supplied train/test boundary. Do not join approximate neighbors automatically.
2. Use five stratified group folds, with shuffle and seed 155. Check row counts, class counts and coarse era composition. Mixed-label groups must remain intact; stratification will be approximate. Reuse precisely these folds for every paired comparison.
3. Use three predeclared group folds for initial screening, then promote only a small shortlist to five folds. Repeat the finalist evaluation with two additional seeds, 2026 and 1029. Repeats measure sensitivity; they do not create independent data.
4. All imputation, scaling, quantiles, feature selection, target encoding, cluster fits and neighbor statistics are fitted inside the appropriate training fold. Outer validation labels never participate in these transformations.
5. Early stopping also needs separation. For confirmatory results, use a group-safe inner training/validation split to choose iteration count, refit on all outer-training rows at that count, and score the untouched outer fold. Automatic random internal splits in HistGB, MLP, SVM calibration and AutoML need inspection or replacement.
6. Report mean fold AUC as the primary comparison and pooled OOF AUC as a secondary view. Cross-fold differences in score scales can affect pooled AUC. Also report per-era performance where both classes occur, row-weighted versus group-weighted sensitivity, and prediction behavior on incomplete rows. Pre-1990 AUC alone is undefined because there are no Low rows.
7. Compare paired OOF predictions using a cluster bootstrap over track/signature groups, retaining all sampled group members. Preserve each repeat's fold assignments and recompute the mean-fold AUC difference in every bootstrap replicate. Reuse the same sampled group multiplicities for both models and all repeated split seeds. Label any pooled-OOF interval separately. Around 2,000 replicates is adequate for a practical interval. Skip replicates with single-class folds. Treat this as conditional uncertainty for the fitted predictions, not full uncertainty from model search or retraining.
8. Do not use an ordinary unpaired test or treat 15 overlapping folds as 15 independent experiments. For the top two or three frozen recipes, a small nested evaluation, such as three outer group folds with limited inner selection, can measure selection optimism. If time prevents this, state the limitation.
9. Run a chronological stress test, for example training on releases through 2022 and validating 2023–2024. Remove any cross-boundary identity/signature overlap. This evaluates extrapolation to newer releases, not necessarily the Kaggle test distribution. Prefer grouped random CV unless collection-time evidence supports a temporal target.
10. Consider a train-versus-test domain classifier only as a distribution diagnostic, with duplicate grouping and permutation checks. Any use of its output to weight popularity training enters the transductive gray area and needs approval. Broad univariate similarity is insufficient evidence for or against such weighting.

A practical promotion rule is an average improvement around 0.003–0.005 AUC with a consistent paired advantage across repeats, or a repeatable contribution to a small ensemble. This is a budget rule, not a significance threshold. A 0.001 win on one split does not justify another day of tuning. If evidence is inconclusive, favor simplicity, stability, or demonstrably complementary errors.

Use public scores to catch target inversion, row-order mistakes, catastrophic distribution mismatch, and broad confirmation of a preselected change. Avoid choosing features or weights from tiny public differences. Its sample size and duplicate structure are unknown; the 863-row full test already limits precision. Log the purpose of every submission before inspecting its score. A disagreement between CV and public LB triggers an audit, not automatic replacement of the validation protocol.

**5. Time-boxed experiments and deliverables**

Use roughly 60% of modeling compute for tree/smooth baselines and tuning, 20% for controlled diversity, and 20% for confirmation and ensembling. Reserve substantial human time for interpretation, review and writing. If a one-fold timing pilot predicts that an experiment will consume its entire time box, shrink it or drop it. Limit thread counts so concurrent libraries do not oversubscribe the Mac.

| Day | Concrete sequence and time boxes | Submission purpose and exit criterion |
|---|---|---|
| Thu Oct 29 | 1–2 h: reviewed audit and split harness. 2–3 h: date-only baseline, audio logistic, audio HistGB/LightGBM, then year-added versions. 1 h: CatBoost default and feature-block comparison. Start report skeleton and reproducibility manifest | Usually two submissions: a valid conservative baseline, then the best validated date-aware contender. Verify orientation, ID alignment and broad CV/LB consistency |
| Fri Oct 30 | 3–5 h compute: CatBoost, LightGBM and XGBoost screening, then tune two leaders. 1–2 h: spline logistic/EBM, RBF SVM and Extra Trees. Compare row versus group training weights. Proposed Piazza clarifications should be resolved early | Two or three justified submissions: leading tuned model, a materially different family, possibly first simple blend. Stop families consistently worse by roughly 0.01 with no blend benefit |
| Sat Oct 31 | 2–4 h: RealMLP and TabM pilots, then one chosen neural recipe. Optional 1–2 h AutoML with explicit folds and whitelist. If approved, give TFMs a separate 2–4 h cap, starting TabPFN-3.5/Fast and TabICLv2. Run duplicate and date stress tests | Two or three submissions covering genuinely different hypotheses. Keep models that win repeated CV or improve a validated ensemble; abandon installation work after about 30–45 minutes without a viable path |
| Sun Nov 1 | 3–5 h: freeze 2–3 candidate recipes, repeat group CV, evaluate simple blends and at most one stacker properly. 2–3 h: importance/direction plots, draft report and end-to-end Colab run | Two or three submissions: robust blend, strongest single or simpler alternative, optional approved-TFM blend. Freeze feature search by evening |
| Mon Nov 2 | Morning: resolve only concrete validation defects; refit frozen recipes, compare full-data refit versus fold-model averaging using prior validation evidence. Finish output checks and review. Target submissions by 2 PM PST and selections by 3 PM PST | Submit at most two final candidates or corrected equivalents. Keep spare capacity for genuine file problems; do not consume all 10 slots gratuitously. Sam chooses two before 5 PM PST |
| Tue Nov 3 | Final report edits, clean Colab rerun, public read-only sharing and required Piazza note, private score/rank if available, final review and Gradescope upload before 11:59 PM PST | Treat writing, reproducibility and explanation as the main graded deliverables |

No external posts, installations or submissions were performed in this consultation. Future tasks should record findings in doc/findings.md before downstream dependence, preserve exact folds/configurations/OOF predictions, and obtain the project's required approval before submission. The suggested daily counts are ceilings, not quotas.

The two final selections should be:

- A: the best stable, validated ensemble of a few complementary models, trained with a frozen recipe.
- B: the strongest credible alternative with meaningfully different failure modes and nearly comparable validation, for example a regularized tree/GAM blend versus an approved TFM blend.

If no diverse alternative is competitive, use the strongest single family or simpler validated blend. Do not select a clearly inferior model merely because its predictions differ. Prefer recipes within roughly 0.005–0.01 AUC, interpreted alongside uncertainty and stress tests. Do not assume Kaggle averages the two selected files; they are separate selected submissions. Sam makes the final choice.

Before each upload, verify exactly 863 rows, the sample's column names and ID order, unique IDs, finite probabilities in [0,1], and the positive-class mapping. Identical supplied features should produce identical predictions unless a documented allowed feature distinguishes them. Hash the prediction file and record the recipe, software versions, seeds, review and submission description.

For the report, keep a single experiment table with feature set, model, tuning budget, split, fold AUCs, pooled OOF AUC, runtime, public score and selection reason. Include an audio-only versus date-added ablation, duplicate treatment, failed models, and ensemble evidence. Follow the template page limits: Introduction ≤1 page, Overview preferably half a page and ≤1, Approach ≤1, Model Selection ≤1, Conclusion ≤1, optional insights separately. Include team identity, division of labor, final private score/rank when known, Colab and Piazza links, and exact LLM usage/integration.

For the top 10 features, compute held-out source-feature permutation importance using AUC. Permute an original source feature and recompute every dependent transform and interaction through the frozen fold-specific feature pipeline; do not permute engineered columns independently. Supplement this ranking with fold-model SHAP where supported. Aggregating SHAP values is a separate attribution analysis; explain how multi-source interactions are assigned rather than double-counting them. Use repeat/fold variability, discuss correlated loudness/energy/acousticness, and supplement with SHAP dependence or ALE. PDP can evaluate unrealistic combinations under correlation. SHAP sign is relative to a baseline, not a universal monotonic effect or a causal claim.

The observed marginal directions suggest loudness, energy, duration and valence positive, acousticness, instrumentalness and release year negative. These are hypotheses for the fitted model's interpretation. Do not predeclare them as the final top seven or invent signs for the remaining three. “Nonmonotonic,” “interaction-dependent,” and “uncertain” are acceptable directions when supported.

The Colab should install pinned dependencies, ingest only the authorized CSVs, verify schema, build the exact features and splits, reproduce the selected model or ensemble, produce several report figures, and generate a checked submission. Keep the full search out of the demo. If a GPU-dependent model wins, demonstrate that actual model and provide a clearly labeled CPU alternative; do not silently substitute it. Record package/checkpoint licenses and instructor approval for any pretrained components.

**6. Ten highest-value ideas, ranked**

These are subjective planning ranges in absolute AUC, not experimental results or confidence intervals. They are not additive. Feature and model improvements are relative to the stated comparator, with validation improvements expressed as risk reduction rather than fictitious score gains.

| Rank | Idea | Plausible impact | Confidence |
|---|---|---|---|
| 1 | Duplicate-aware grouped validation and leakage-safe preprocessing/stopping | No direct score gain. Prevents an unknown amount of optimistic model selection | High necessity; magnitude unmeasured |
| 2 | Add release year and a restrained date representation | About +0.01 to +0.06 versus the same audio-only learner; zero or negative remains possible | Medium to high |
| 3 | Tune regularized CatBoost/LightGBM/XGBoost | About +0.01 to +0.05 versus basic audio/date linear or untuned baseline | Medium |
| 4 | Blend 2–4 complementary validated models | About 0 to +0.015 versus the best member; can lose if weights overfit | Medium |
| 5 | Test an approved current TFM | About −0.02 to +0.03 versus strong tuned trees | Medium on relevance, low on direction for these data |
| 6 | Add spline logistic/GAM/EBM or RBF SVM | About 0 to +0.01 as a selected ensemble contribution; a single model may perform worse | Medium |
| 7 | Compare row weighting with one-track-one-weight and preserve conflicts | About −0.005 to +0.01 versus default row weighting | Low to medium |
| 8 | Use curated audio transforms and date/audio interactions | About −0.005 to +0.015 versus the corresponding plain feature set; likely smaller for trees | Medium |
| 9 | Try RealMLP or TabM for diversity | About −0.01 to +0.015 as a standalone challenger to a strong baseline; blend gain usually smaller | Low to medium |
| 10 | Stop broad search early and spend the last day on confirmation, reproducibility and explanation | AUC effect unquantified; protects model choice and the dominant graded deliverables | High process value |

I would spend almost no early budget on raw URI encodings, deep transformer architecture searches, pseudo-labeling, aggressive label correction, or hundreds of ensemble weights. Their upside is less well supported here than dates, regularized tabular models, and disciplined comparisons.

---

Review record retained in this conversation: two fresh supplemental reviewers independently checked the data and sources. Their requested bootstrap, permutation-importance, and citation corrections were incorporated. The latest reviewer’s exact technical outcome was **“SUBSTANTIVE VERDICT: PASS”**, covering revision `743c3d7ea72ca2853ac8e2f3cdf0e250052af76176acf644de30aa1ea46f4471`.

Required cross-family approval remains pending. [Review policy §8.4](/Users/zwl0331/Desktop/Mini_project_1/doc/review_policy.md) requires Claude Code review; the CLI returned “Not logged in · Please run /login.” No `VERDICT: APPROVED` is claimed. Your requested one-task reviewer exception remains unanswered.