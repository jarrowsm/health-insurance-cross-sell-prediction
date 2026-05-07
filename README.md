# Health Insurance Cross-Sell Prediction

This repository implements and benchmarks several algorithms for the [Kaggle Insurance Cross-Selling task](https://www.kaggle.com/competitions/playground-series-s4e7/data), including a from-scratch AdaBoost implementation, classical ML models, and AWS SageMaker automated pipelines.

`00_adaboost_impl.ipynb`
- Visual AdaBoost demonstrated with synthetic 2D data and selected sales data pair
- Examine margins (CDF)
- Optimise decision tree depth
- Focus on hard samples: modify weight updates + remove easy samples
- Optimise threshold for recall
- Ensemble pruning: rank-, search-, and cluster-based methods

`01_adaboost_opt.ipynb`
- AdaBoost optimisation: test ROC AUC 0.845 → 0.874
- Early stopping (patience + tolerance)
- Data balancing: undersampling vs SMOTE vs hybrid
- Feature engineering: interactions + derived features; prune via mean CV α
- Feature encoding + selection (manual + automated)
- Stratified k-fold CV tuning: tree depth, criterion (Gini/entropy/log loss), η, rounds, weight factor, threshold, easy + hard sample removal

`02_model_bench.ipynb`
- Benchmark 13 models (linear, instance-based, tree, ensemble)
- Benchmark data balancing methods
- Feature selection/engineering: LightGBM importance + CFS pruning
- CatBoost (best): SMOTE, undersampling, full vs LightGBM top-5 vs CFS top-5, Optuna tuning → ROC/AUC 0.876

`03_sagemaker_catboost_autopilot.ipynb` 
- S3 + SageMaker: Kaggle API data import → SKLearnProcessor preprocessing
- HyperparameterTuning:
    - Job 1: major AUC gain (analyse AUC, parameter effects)
    - Job 2: marginal gain (increase η + early stopping)
- Final CatBoost: train job → endpoint → inference → AUC = 0.876
- Autopilot benchmark: raw data → Boto3 monitoring → batch inference
