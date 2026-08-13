# labs — 可复现的正确性证明

「我怎么知道这些算法可靠？」——本目录就是答案。它把 duckdb-ml 每个
手写 Rust 算法与业界参考实现（scikit-learn 等）在**固定种子的合成数据**上
做对照，输出带明确容差的 PASS/FAIL 报告，一条命令可复现。

## 可靠的三层定义

| 层 | 含义 | 检查方式 |
|---|---|---|
| 1. 确定性 | 同数据同参数，两次训练结果逐位一致 | `run_all.py` 里的 determinism 检查 |
| 2. 对照参考实现 | 预测/系数与 sklearn 同参同数据的输出一致到指定容差 | 每个 verify_*.py |
| 3. 容差显式化 | 每项检查打印 ours / ref / tol，不达标即 FAIL 且退出码非零 | Report 表格 |

容差是**诚实**的：线性模型（闭式解）要求 1e-6 级系数一致；树/SVM/MLP 与
sklearn 的分裂搜索、SMO 内部、SGD 初始化本就不同，按预测质量（R²/accuracy）
设阈值；k-NN 是精确算法，要求与 sklearn 逐位一致；metrics/assoc_rules 是
确定性公式，要求 1e-9 级一致。

## 运行

前置：`make release` 已产出 `build/release/ml.duckdb_extension`（本目录
直接引用它，不重新编译）。

```bash
# 方式一：uv（推荐，隔离环境）
uv run --with duckdb --with numpy --with scipy --with scikit-learn python labs/run_all.py

# 方式二：pip
pip install -r labs/requirements.txt
python labs/run_all.py
```

退出码 0 = 全部通过；非 0 = 有 FAIL（CI 可直接挂这条）。

## 覆盖矩阵

| 文件 | 算法 | 参考实现 | 关键容差 |
|---|---|---|---|
| verify_01_linear.py | ml_ols / linear / ridge / lasso / elastic_net / robust | sklearn LinearRegression/Ridge/Lasso/ElasticNet/HuberRegressor | 系数 1e-6；预测 R²≥0.9~0.99 |
| verify_02_trees.py | decision_tree / random_forest / rf_classifier / xgboost_regression / xgboost_binary | sklearn Tree/Forest/GradientBoosting | R²≥0.95 / acc≥0.97；字符串标签解码 |
| verify_03_kernel.py | svm (linear/rbf) / svr (linear/rbf) | sklearn SVC/SVR | acc≥0.95 / R²≥0.95；预测相关≥0.99 |
| verify_04_classify.py | logistic / multilogistic / ordinal / naive_bayes / knn | sklearn Logistic/GNB/KNN | acc≥0.85~0.95；k-NN 与 sklearn 逐位一致 |
| verify_05_unsup.py | kmeans / dbscan / pca | sklearn KMeans/DBSCAN/PCA | 簇对齐≥0.85~0.95；PC1 相关≥0.999 |
| verify_06_misc.py | mlp / cox / arima / ml_metrics / ml_cross_validate / ml_assoc_rules | sklearn MLP/metrics/CV + 闭式 AR + 暴力 Apriori | metrics 1e-9；assoc_rules 1e-9 |

注：ordinal/cox 无 sklearn 等价物，用「预测类别与真实有序类一致 +
与潜变量秩相关」/「风险分与真实 log-hazard 秩相关」证明；pca 的
`ml_predict` 只暴露 PC1 得分（MlModel 是标量输出），按 PC1 对照。

## 数据与随机性

所有数据由 `np.random.default_rng(42)`（run_all 的确定性检查用 seed 7）
生成——两次运行输出完全相同。sklearn 侧一律传 `random_state=42`。
每类数据都是「干净、参数已知」的合成问题（如 y = 2 + 3x₀ − 1.5x₁ + 0.4x₂ + N(0,0.05)），
这样参考实现本身无可争辩，算法差在哪一目了然。

## 扩展新算法

1. 在 `common.py` 加数据生成器（固定 seed）
2. 新写 `verify_XX_*.py`：train → predict → 对照 → `report.add(...)`
3. `run_all.py` 的 SCRIPTS 列表加一行
4. 跑 `python labs/run_all.py` 确认全绿

## 已知边界

- **svr-rbf 不设正确性断言（KNOWN LIMITATION，2026-08-14 labs 实测）**：
  手写 working-set SMO 在 RBF 目标上**不稳定**——同一 sin(x) 配置一次
  R²=0.98、另一次 R²=-0.12；x²[0,3] / sin(2x) 全范围评估 R²≈0。
  skill 旧记录的「pred(1.5)=2.25024」单点 e2e 是碰巧命中，不代表正确拟合。
  SVR 家族的正确性由**线性核**证明（R²=1.0、与 sklearn 预测相关 1.0）；
  RBF 的 SMO 工作集质量是跟踪中的修复项，修好前 labs 只做「能跑且有限」探针。
- ml_embed / ml_load_onnx 需 `--features onnx` 构建，默认 `make release`
  未含，故本套件不覆盖（onnx 桥的验证见仓库内单测与 openspec 记录）
- 容差以本机 duckdb 1.5.x + sklearn 1.9 标定；升级 sklearn 大版本后
  若个别阈值波动，先看「ours/ref」两列再决定是否调整阈值
