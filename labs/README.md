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
| verify_01_linear.py | ml_ols / linear / ridge / lasso / elastic_net / robust / polynomial_regression | sklearn LinearRegression/Ridge/Lasso/ElasticNet/HuberRegressor/PolynomialFeatures | 系数 1e-6；预测 R²≥0.9~0.99；多项式单特征逐位一致 |
| verify_02_trees.py | decision_tree / random_forest / rf_classifier / xgboost_regression / xgboost_binary（num_class>2 → 多分类 softmax multi:softprob） | sklearn Tree/Forest/GradientBoosting | R²≥0.95 / acc≥0.97；字符串标签解码；softmax label agreement 1.0 |
| verify_03_kernel.py | svm (linear/rbf) / svr (linear/rbf) / lda | sklearn SVC/SVR/LDA | acc≥0.95 / R²≥0.95；预测相关≥0.99；lda 判别方向|corr|≥0.99 |
| verify_04_classify.py | logistic / multilogistic / ordinal / naive_bayes / knn / adaboost | sklearn Logistic/GNB/KNN/AdaBoost(SAMME) | acc≥0.85~0.95；k-NN 逐位一致；adaboost acc≥0.8 + 标签一致≥0.85 |
| verify_05_unsup.py | kmeans / fuzzy_cmeans / agglomerative / tsne / dbscan / pca | sklearn KMeans/DBSCAN/PCA/AgglomerativeClustering | 簇对齐≥0.85~0.95；PC1 相关≥0.999；fcm/tsne 重训逐位一致；agglomerative 三 linkage 对齐≥0.9；tsne within/between<0.5 |
| verify_06_misc.py | mlp / cox / arima / kaplan_meier / smote / voting / ml_metrics / ml_cross_validate / ml_assoc_rules | sklearn MLP/metrics/CV + 闭式 AR + 闭式乘积限 + 暴力 Apriori | metrics 1e-9；km 中位逐位一致；smote 数量/局部性/重训一致；voting hard==手动 bincount 100% |

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

- **svr-rbf 已修复（2026-08-14，working-set SMO 重写）**：此前不稳定
  （同配置 R² 0.98 ↔ -0.12 波动、x² 全范围 ≈-0.5）根因 = 对级 violation 全对扫描
  在单调目标（e 全同号）上选到相邻点 → eta 极小 → β 撞 box 边界 ±C → 伪收敛。
  修复 = libsvm 式**单点方向梯度工作集**（argmin d / argmax d，d = g + ε·sign(β)）+
  **无偏梯度**（迭代中 bias 恒 0，梯度增量维护 O(n)/轮，终值 bias 由自由 SV 平均）+
  更新步补 **ε·sign(β) 项**。现 sin(x)/x²/sin(2x) R²≥0.99998（sklearn 1.0/0.999997）。
  verify_03 已加回 R²≥0.9 + 与 sklearn 预测相关≥0.99 断言。
- ml_embed / ml_load_onnx 需 `--features onnx` 构建，默认 `make release`
  未含，故本套件不覆盖（onnx 桥的验证见仓库内单测与 openspec 记录）
- 容差以本机 duckdb 1.5.x + sklearn 1.9 标定；升级 sklearn 大版本后
  若个别阈值波动，先看「ours/ref」两列再决定是否调整阈值
