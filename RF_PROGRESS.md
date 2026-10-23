# Random Forest 开发进度：阶段 0–3、5 及 16 组网格搜索已完成

更新日期：2026-10-23。

依据 [Random Forest Guide](/Users/sylphil/Documents/ChatGPT/Programming/RANDOM_FOREST_VIBE_CODING_GUIDE.md)，已完成小规模训练，并按用户要求跳过阶段 4，直接完成阶段 5 的全量候选训练、验证选参和测试评估。团队 Benchmark 接入尚未实施。

**阶段 5 结果见第 6 节，首次精简见第 7 节，扩展网格搜索见第 8 节，固定参数改造见第 9 节，当前无命令行参数的最简主程序见第 10 节；第 1–5 节保留阶段 0–3 完成时的历史记录。**

## 1. 本轮文件变化

- 新增 `main_rf.py`：独立入口，目前仅支持 `--mode smoke`，默认也是 smoke。
- 新增 `tests/test_rf.py`：5 项小型合成数据测试。
- 修改 `tests/test_main.py` 两行：加载入口从旧 `main.py` 改为当前 `main_ridge.py`；按现有函数接口解包 `select_alpha()` 返回的 `(model, records)`。
- 新增本进度文件及 `results/rf_smoke_seed42/` 内的三个运行产物。

开始工作时 Git 状态为空。指南生成后 Ridge 入口已改名为 `main_ridge.py`，已有测试尚未同步，首次基线测试因此在收集阶段报 `FileNotFoundError`。上述测试修正后，原有 17 项测试全部通过。Ridge 主程序本身未修改。

现有读取、清洗、编码、划分、Adapter、README、依赖配置和原始数据均保留。未复制数据模块，未新增 RF Adapter、manifest、哈希样本 ID 或 JSON 配置。

## 2. 已完成的流程

1. 使用项目现有 `.venv`，直接导入 `ridge` 和 `RandomForestRegressor`，未安装或升级依赖。
2. 调用 `load_data()` / `clean_data()`，然后对**完整清洗数据**调用 `create_splits(seed=42)`。
3. 核对完整划分的 `source_row` 覆盖和唯一性、`utr` 唯一性，确保集合无重叠。
4. 分别用 `numpy.default_rng(split_seed).permutation()` 为训练池和验证池生成固定排列，取前 5,000 / 1,000 条。模型 seed 不影响样本抽取。
5. 复用 `one_hot_encode()` 和 `SklearnAdapter`：输入 `(N,50,4)`，Adapter 内部按 C 顺序展平到 `(N,200)`。
6. 只使用序列特征预测 `rl`，标签转为 float32，保持原始 MRL 单位；无缩放、训练权重或标签变换。
7. 用固定单个森林训练、验证、保存、加载，并核对全部 1,000 条验证预测。
8. 输出模型、验证预测 CSV 和简短运行说明。测试集仅检查样本隔离，未编码、预测或评分。

`select_random_forest()` 提供 train-only 候选拟合与验证 RMSE 选优接口，返回最佳已拟合模型和记录列表；并列保留较早候选。该逻辑用合成测试验证，本轮真实运行仅传入一个固定候选，没有实际调参。

## 3. 实际运行与结果

在 `/Users/sylphil/optimus5prime-pytorch-repro` 中执行：

```bash
.venv/bin/python -B -u main_rf.py --mode smoke --split-seed 42 --model-seed 42 --n-jobs 2 --output-dir results/rf_smoke_seed42
```

命令退出码为 0。再次运行需指定新的输出目录；已有目录不会被覆盖。

| 项目 | 本次实测 |
|---|---|
| 原始 / 清洗后数据量 | 326,033 / 326,033 |
| 完整 train / val / test | 260,826 / 32,603 / 32,604 |
| 实际拟合 / 验证样本量 | 5,000 / 1,000 |
| split seed / model seed | 42 / 42 |
| 树数 / 最大深度 | 20 / 12 |
| `min_samples_leaf` / `max_leaf_nodes` | 5 / 10,000 |
| `max_features` / `max_samples` | 0.3 / 0.8 |
| `bootstrap` / `criterion` | True / squared_error |
| 执行设备 / `n_jobs` | CPU（arm64）/ 2 |
| 验证 Pearson r | 0.154785831 |
| 验证 Pearson r² | 0.023958653 |
| 验证 RMSE | 1.722662443 |
| 训练样本均值基线的验证 RMSE | 1.742170397 |
| 单次模型拟合耗时 | 0.101 秒 |
| 主程序记录的总运行耗时 | 4.036 秒 |
| 模型文件大小 | 1,068,129 bytes |
| 保存 / 加载预测一致性 | 通过 |

所有清洗步骤的删除数量均为 0；重复序列组与标签冲突组均为 0。

这些是 **smoke 流程验证结果**，不是全量性能或正式测试结果。没有调整参数提高分数，也没有与不同样本划分或规模的旧 Ridge 分数直接比较。

运行环境：Python 3.14.7、NumPy 2.5.4、pandas 3.0.6、scikit-learn 1.9.1、joblib 1.6.0。

## 4. 输出与核对

输出目录：[results/rf_smoke_seed42](/Users/sylphil/optimus5prime-pytorch-repro/results/rf_smoke_seed42)。目录恰好包含：

- [random_forest.joblib](/Users/sylphil/optimus5prime-pytorch-repro/results/rf_smoke_seed42/random_forest.joblib)：森林及现有 Adapter 保存的特征约定、配置和版本。
- [validation_predictions.csv](/Users/sylphil/optimus5prime-pytorch-repro/results/rf_smoke_seed42/validation_predictions.csv)：1,000 行，列为 `source_row,utr,rl,prediction`。
- [summary.txt](/Users/sylphil/optimus5prime-pytorch-repro/results/rf_smoke_seed42/summary.txt)：数据、配置、形状、指标、计时、版本和检查结果。

训练后独立重新读取数据并执行同一划分及抽样，确认 CSV 的来源行、序列、标签和顺序一致；验证样本与完整训练集及测试集无重叠。再次加载模型的预测与 CSV 一致，预测全部有限。

使用 `scipy.stats.pearsonr` 和 NumPy 重新计算验证指标，与程序结果一致。复算时将 CSV 的 `rl` 转为 float32 后再计算，与训练入口使用的标签精度保持一致。

原始 CSV 大小仍为 175,824,030 bytes，修改时间纳秒值仍为 1792592722955066130。核对 Git diff，原有数据/Adapter 模块、Ridge 主程序、README、依赖配置和脚本均未修改。

## 5. 检查记录

```text
首次已有测试基线：1 个收集错误，tests/test_main.py 引用了已不存在的 main.py。

修正测试引用和返回值解包后：
.venv/bin/python -B -m pytest tests/test_main.py tests/test_data.py tests/test_ridge.py -q -p no:cacheprovider --basetemp=/private/tmp/rf-stage03-baseline-fixed
17 passed in 0.65s

新增 RF 测试：
.venv/bin/python -m pytest tests/test_rf.py -q
5 passed in 0.68s

完整测试集：
.venv/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/private/tmp/rf-stage03-final
22 passed in 0.72s

.venv/bin/python -B main_rf.py --help
exit code 0

git diff --check
exit code 0
```

5 项新增测试分别覆盖手动展平的一致性、保存加载、验证标签不参与固定候选拟合、非首项候选选优且仅使用训练标签，以及 Pearson/RMSE/常数情况。测试收集通过 `runpy` 导入入口，没有触发主程序训练。

另用临时合成 CSV 检查数据不足时在拟合前明确报错，并确认已有输出目录拒绝覆盖、`--mode final` 在当前阶段明确拒绝。三项预期失败均返回退出码 2。

阶段 0–3 完成时，尚未实现 `validate`、`final` 模式，原计划下一步做有限候选调参与资源评估。用户随后要求跳过阶段 4，直接实施下述阶段 5。

## 6. 阶段 5：全量训练与正式测试（2026-10-23）

### 6.1 范围与实现

按用户“跳过阶段 4，直接完成阶段 5”的要求，未开展 20k/50k 训练子集预筛，直接在完整训练集上比较指南中预先定义的 4 个候选。

- `main_rf.py` 新增 `--mode final`，保留原 `smoke` 模式和默认行为。
- 正式模式使用完整 train、完整 val；不提供 `--train-size`，不合并 train+val 重训。
- 四个候选均使用 100 棵树、`max_leaf_nodes=10000`、`bootstrap=True`、`max_samples=0.8`、`criterion="squared_error"`、`model_seed=42`、`n_jobs=2`。
- 复用已有选参函数，以验证集 RMSE 最小者选优，并列保留较早候选。
- 保留最佳已拟合模型，保存后加载，用验证集前 256 条确认预测一致；之后才编码测试集、生成最终预测并评分。
- 测试结果产生后，没有更改候选、种子、特征或样本，也没有根据测试分数重新选参。
- `tests/test_rf.py` 增加 smoke/final 参数化集成测试，检查数据范围、拟合次数、保存加载及测试评估顺序。

### 6.2 正式运行

可复现的训练入口（输出目录必须尚不存在）：

```bash
cd /Users/sylphil/optimus5prime-pytorch-repro
.venv/bin/python -B -u main_rf.py --mode final --split-seed 42 --model-seed 42 --n-jobs 2 --output-dir results/rf_full_seed42
```

本次实际执行在上述 Python 命令前加了 `/usr/bin/time -l`。训练程序完成了 4 个候选、保存加载核对、测试预测和三个结果文件的写出，并打印最终输出路径；随后外层计时工具因 `sysctl kern.clockrate: Operation not permitted` 返回退出码 1。

这是附加系统资源统计失败；没有为获取该统计重跑训练或提升权限。下文耗时采用主程序自身记录的 `perf_counter` 数据，未报告无法获得的峰值内存。模型和结果文件另行独立核对。

实际数据：原始/清洗后均为 326,033 条；train=260,826，val=32,603，test=32,604。四项清洗删除数、重复序列组和标签冲突组均为 0。两个 seed 均为 42，编码及标签单位与 smoke 阶段一致。

### 6.3 四个候选的验证结果

| 候选 | 最大深度 | 叶节点最少样本 | 特征比例 | 验证 RMSE | 验证 Pearson r² | 拟合耗时（秒） |
|---|---:|---:|---:|---:|---:|---:|
| 1 / A | 16 | 5 | 0.3 | 1.624663 | 0.160323 | 34.396 |
| 2 / B | 24 | 5 | 0.3 | 1.623922 | 0.158038 | 36.949 |
| 3 / C | 24 | 20 | 0.3 | 1.623125 | 0.169947 | 34.243 |
| **4 / D（选中）** | **24** | **5** | **0.6** | **1.607374** | **0.174020** | **70.203** |

候选 4 在这四组参数中具有最低验证 RMSE；它是本次有限候选比较的最佳配置，不代表所有可能参数中的全局最优。

### 6.4 选中模型的验证和测试结果

| 集合 | 样本量 | Pearson r | Pearson r² | RMSE | 训练均值基线 RMSE |
|---|---:|---:|---:|---:|---:|
| validation | 32,603 | 0.417157180 | 0.174020113 | 1.607373612 | 1.739888 |
| test | 32,604 | 0.421204958 | 0.177413617 | 1.619936740 | 1.755532 |

主要报告指标为 Pearson r²，选参指标仍是验证 RMSE。测试结果来自已经选定并重载的单一模型；没有使用测试集比较四个候选。

四次拟合合计 175.790 秒；候选搜索含验证预测等共 176.440 秒；主程序记录总运行 181.560 秒，约 3 分钟。测试集预测计时 0.164 秒，不包含编码或模型加载。

模型大小为 144,025,777 bytes。运行环境保持 Python 3.14.7、NumPy 2.5.4、pandas 3.0.6、scikit-learn 1.9.1、joblib 1.6.0，CPU arm64、2 个并行 worker。

正式输出位于 [results/rf_full_seed42](/Users/sylphil/optimus5prime-pytorch-repro/results/rf_full_seed42)，共三个文件：

- [random_forest.joblib](/Users/sylphil/optimus5prime-pytorch-repro/results/rf_full_seed42/random_forest.joblib)：候选 4 的模型、参数、特征约定及软件版本。
- [test_predictions.csv](/Users/sylphil/optimus5prime-pytorch-repro/results/rf_full_seed42/test_predictions.csv)：32,604 行，列为 `source_row,utr,rl,prediction`。
- [summary.txt](/Users/sylphil/optimus5prime-pytorch-repro/results/rf_full_seed42/summary.txt)：四个候选的记录、最终参数、划分、验证/测试指标、均值基线及计时。

### 6.5 检查记录

```text
.venv/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/private/tmp/rf-final-suite
24 passed in 0.74s

.venv/bin/python -B main_rf.py --help
exit code 0

git diff --check
exit code 0
```

新增集成测试使用临时合成数据和小型真实 RF，验证 final 使用完整 train/val、四次拟合仅接收 train、不合并验证集重训、先完成选参及重载核对再编码测试集、最终模型仅预测测试集一次，并检查输出文件和来源行顺序。smoke 仍不编码或评估测试集。

另确认正式模式拒绝 `--train-size`，已有结果目录拒绝覆盖；两项预期拒绝均返回退出码 2。没有执行阶段 4 的 `validate` 模式，也没有更改原始数据、Ridge 主程序、公共数据模块或 Adapter。

正式运行后，另行只读重建全量划分，确认三个集合的 `source_row` 和 `utr` 两两无交叉且完整覆盖清洗数据；测试 CSV 的 32,604 条来源行、序列及顺序精确对应测试集，标签也一致。使用 SciPy Pearson 和手工 RMSE 按 float32 标签精度复算测试指标，与摘要在 1e-12 容差内一致。

重新加载 joblib 后，核对参数与摘要完全一致，且候选 4 确实具有最低验证 RMSE；对完整 32,603 条验证数据重新预测，指标与保存的验证记录一致。该核对未重新训练、未修改任何参数或产物，也未触发新的候选选择。

## 7. 精简主程序和 summary（2026-10-23）

按用户“和 Ridge 一样简化主程序和 summary，只保留最主要结果”的要求，`main_rf.py` 从 227 行精简为 **139 行**。

保留 smoke/final 两种模式、原有四组候选及公共参数、固定划分与抽样、验证 RMSE 选参、train-only 拟合、模型保存和最终预测 CSV。`select_random_forest()` 现在只返回最佳已拟合模型，不再返回搜索记录。

删除逐候选输出及记录、计时、均值基线、详细 QC/环境/模型配置打印，以及主程序中的重复划分核对和保存重载断言。保存加载能力仍由原有 Adapter 及对应测试覆盖；模型包内的完整配置和版本信息仍保留。

summary 当前仅保留：

- 运行模式、数据路径、split/model seed 和实际使用的样本数。
- 最佳模型的树数、最大深度、叶节点最少样本和特征比例。
- 验证 RMSE 选参及仅训练集拟合的说明。
- 验证集/测试集的 Pearson r²、Pearson r、RMSE 表。smoke 仅有验证集结果，标明测试集未评估。
- 仅当相关系数未定义时附一行常数目标/预测说明。

已将正式结果 [summary.txt](/Users/sylphil/optimus5prime-pytorch-repro/results/rf_full_seed42/summary.txt) 从 **3,893 bytes 精简为 501 bytes**。更新基于已保存的真实运行结果，并从测试预测 CSV 独立复算核对了指标，没有重新训练或重新选择参数。

原有 `random_forest.joblib` 和 `test_predictions.csv` 的大小、修改时间均未变化；正式输出目录仍恰好包含三个文件。旧 smoke 运行产物保持不变。第 6 节保留正式训练时的候选比较及计时记录。

测试仅调整已有选参和入口集成断言，不增加测试函数；继续验证保存加载、train-only 拟合、选参完成及模型保存后才编码测试集、测试集只预测一次及 smoke 隔离测试集。

```text
.venv/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/private/tmp/rf-simplified-suite
24 passed in 0.76s

.venv/bin/python -B main_rf.py --help
exit code 0

git diff --check
exit code 0
```

本轮修改 `main_rf.py`、`tests/test_rf.py`、本进度文件和正式结果 summary；未修改 Ridge 主程序、公共数据模块、Adapter、依赖或原始数据。

## 8. 扩大参数范围的网格搜索（2026-10-23）

### 8.1 搜索范围和方法

新增独立入口 `grid_search_rf.py`，保留精简后的 `main_rf.py`。复用现有数据读取、清洗、划分、one-hot 编码和 Adapter，不改变原始数据、特征或标签单位。

| 参数 | 搜索值 |
|---|---|
| `n_estimators` | 50、100 |
| `max_depth` | 24、None（不直接限制深度） |
| `min_samples_leaf` | 5、20 |
| `max_features` | 0.6、1.0 |

共 **2 × 2 × 2 × 2 = 16** 组，包含之前最佳的候选 D，并探索更多特征、更深树及较少树数。固定 `max_leaf_nodes=10000`、`bootstrap=True`、`max_samples=0.8`、`criterion="squared_error"`、`random_state=42`；因此 `max_depth=None` 时仍有叶节点数量限制。

最初启动了 36 组搜索（100/200 棵树、深度 16/24/None、叶样本 5/20、特征比例 0.3/0.6/1.0）。完成 17 组后，用户要求缩短时间，随后明确改为约 16 组，因此中断原任务并采用上述范围。原实现仅在全部搜索结束后保存结果，原 17 组没有保存验证分数，不据此声明最佳参数。`results/rf_grid_seed42/summary.txt` 仅记录中断状态；最终结果另存新目录。

使用 scikit-learn 的 [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 穷举参数组合，以 `neg_root_mean_squared_error` 评分，分数最大即验证 RMSE 最小。采用 [PredefinedSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.PredefinedSplit.html) 保持原 train/val：train 标为 -1，val 标为 0。这是**固定验证集上的网格搜索，不是 K 折交叉验证**。

搜索直接使用与 Adapter 相同的 C-order 展平特征，设置 `refit=False`，避免默认在传入的 train+val 上重新拟合。选参后将最佳森林交给现有 Adapter，仅在原训练集上重新拟合；核对验证 RMSE 与搜索记录一致后保存。搜索组合顺序执行，森林内部使用 6 个线程。

评分函数包装官方负 RMSE scorer，每完成一组就原子更新 `cv_results.csv` 中的参数、验证 RMSE 和当前排名；全部搜索完成后补充拟合耗时。中断时已完成的分数可保留，当前不提供自动恢复或跳过已完成组合的功能。

训练和验证使用完整划分，均不抽样；种子均为 42。此前测试集已经评估过，本轮仅报告验证结果，不编码、预测或评分测试集，也不使用旧测试分数决定参数。

### 8.2 运行命令

```bash
cd /Users/sylphil/optimus5prime-pytorch-repro
.venv/bin/python -B -u grid_search_rf.py --split-seed 42 --model-seed 42 --n-jobs 6 --baseline-model results/rf_full_seed42/random_forest.joblib --output-dir results/rf_grid16_seed42
```

`--baseline-model` 可选；本次传入之前保存的同一数据、同一划分模型，重新计算其验证指标作为比较。已有输出目录会被拒绝覆盖。未来可在入口顶部修改 `PARAM_GRID` 扩展范围，再指定新输出目录运行。

### 8.3 完成结果

16 组全部完成，命令退出码为 0。最佳配置为：

```python
{"n_estimators": 100, "max_depth": 24, "min_samples_leaf": 20, "max_features": 1.0}
```

| 模型 | 验证 RMSE | 验证 Pearson r | 验证 Pearson r² |
|---|---:|---:|---:|
| 上一轮候选 D | 1.607373612 | 0.417157180 | 0.174020113 |
| 本轮网格最佳 | 1.594968191 | 0.443954820 | 0.197095882 |

验证 RMSE 降低 0.012405421，约 **0.7718%**；Pearson r² 增加约 0.023076。这是本次 16 组、固定验证划分下的最佳结果，不是全局最优或新的测试集成绩。

排名第二的配置仅将 `max_depth` 改为 None，验证 RMSE 为 1.594968449，与第一名几乎持平。因此本轮没有证据表明取消深度限制带来有效改善；选出的模型仍使用深度 24。

耗时与初始估计不符：搜索记录的 16 次拟合墙钟时间合计 **21,483.834 秒（约 5.97 小时）**，不包含最后一次最佳模型拟合。多数单组约 13–47 秒，但其中一组约 202 秒、最后三组分别约 140/48/162 分钟。该记录不能区分实际 CPU 工作与进程停顿，异常长耗时原因尚未核实；不据此比较参数的训练速度，也不把最初预计的 6–10 分钟当作实测结果。

### 8.4 产物和核对

本轮输出位于 [results/rf_grid16_seed42](/Users/sylphil/optimus5prime-pytorch-repro/results/rf_grid16_seed42)，共四个文件：

- [summary.txt](/Users/sylphil/optimus5prime-pytorch-repro/results/rf_grid16_seed42/summary.txt)：搜索范围、最佳参数和新旧模型主要验证指标。
- [cv_results.csv](/Users/sylphil/optimus5prime-pytorch-repro/results/rf_grid16_seed42/cv_results.csv)：16 组完整排名、验证 RMSE 和原始拟合耗时记录。
- [random_forest.joblib](/Users/sylphil/optimus5prime-pytorch-repro/results/rf_grid16_seed42/random_forest.joblib)：仅使用原训练集拟合的最佳模型，79,846,249 bytes。
- [validation_predictions.csv](/Users/sylphil/optimus5prime-pytorch-repro/results/rf_grid16_seed42/validation_predictions.csv)：全部 32,603 条验证预测，含来源行、序列、真实标签和预测值。

独立审计重新读取原数据、重建划分，确认 16 个组合完整且唯一、排名方向正确；验证 CSV 的来源行、序列、标签和顺序准确，train/val/test 按来源行和序列均无交叉。加载新模型后确认参数与排名第一的记录一致，完整验证集预测与 CSV 一致。

使用 NumPy RMSE 和 SciPy Pearson 独立复算上述指标。旧模型验证 RMSE 在新网格中精确复现；原始 CSV、旧模型及旧测试预测文件的大小和修改时间均保持不变。审计没有重新训练或编码、预测测试集。

新增两项测试覆盖固定验证划分、负 RMSE 选优、关闭自动 refit、每次仅拟合原 train、候选间及时保存结果、最终模型保存加载及不编码测试集。现有 `main_rf.py` 仍为 139 行；公共数据模块、Adapter、Ridge 入口和依赖配置未修改。

```text
.venv/bin/python -B -m pytest -q
26 passed in 1.43s

独立产物审计
audit_passed: true

git diff --check
exit code 0
```

运行过程中仅对最终排名表的写入方式做了额外健壮性修正：也先写临时文件，再原子替换；本次已启动的进程仍按原最终写入方式执行，已正常保存并通过审计。参数、评分及训练逻辑未因此变化，无需重跑训练。

## 9. 主程序固定使用网格最佳参数（2026-10-23）

本节记录固定参数改造时的行为；其中命令行参数和模式切换已在第 10 节进一步移除。

按用户要求，`main_rf.py` 删除原四组候选和 `select_random_forest()`，直接使用 `RF_PARAMS` 构建模型并拟合一次：

```python
RF_PARAMS = {
    "n_estimators": 100, "max_depth": 24, "min_samples_leaf": 20, "max_features": 1.0,
    "max_leaf_nodes": 10000, "bootstrap": True, "max_samples": 0.8,
    "criterion": "squared_error",
}
```

上述 8 项已与保存的网格最佳模型逐项核对一致。`smoke` 和 `final` 使用相同参数；smoke 只取原有 5,000/1,000 条训练/验证样本，final 使用完整训练集并在保存模型后评估测试集。默认模式仍为 smoke；种子和并行度继续由原命令行参数控制。

验证集仅用于报告指标，不再参与本次运行的参数选择。Summary 将 `best_params` 改为 `model_params`，说明参数固定来自此前网格搜索，保留主要验证/测试指标。

正式运行示例（输出目录须尚不存在）：

```bash
cd /Users/sylphil/optimus5prime-pytorch-repro
.venv/bin/python -B -u main_rf.py --mode final --split-seed 42 --model-seed 42 --n-jobs 6 --output-dir results/rf_fixed_seed42
```

本轮修改程序和测试，未再次执行真实数据的全量训练或测试集评分。集成测试用临时合成 CSV、真实 100 棵树配置验证两种模式均仅拟合一次，且只使用 train；同时验证模型保存加载、输出指标及测试评估顺序。删除原主程序候选选优测试，该行为仍由独立网格搜索测试覆盖。

```text
.venv/bin/python -B -m pytest -q
25 passed in 0.89s

.venv/bin/python -B main_rf.py --help
exit code 0

git diff --check
exit code 0
```

## 10. 无命令行参数的最简主程序（2026-10-23）

按用户要求，将 `main_rf.py` 简化为 **84 行**，移除 `argparse`、全部命令行选项、smoke/final 模式、子集抽样及相关分支。顶部直接设置 `DATA_PATH`、`OUTPUT_DIR`、`SEED`、`N_JOBS` 和固定的 `RF_PARAMS`；一个 `SEED` 同时控制数据划分和模型随机性，默认仍为 42。

直接运行，无需传入参数，也不再提供 `--help`：

```bash
/Users/sylphil/optimus5prime-pytorch-repro/.venv/bin/python /Users/sylphil/optimus5prime-pytorch-repro/main_rf.py
```

流程固定为：读取清洗数据 → 完整划分 → 仅在 train 上训练一次 → 保存模型 → 评估 validation/test → 保存测试预测和简短 summary。原有数据处理、Adapter、最佳模型参数和标签精度保持一致。

默认输出目录为 `results/rf_fixed`，路径相对脚本所在目录定位。保留一行 `mkdir(exist_ok=False)` 防止覆盖已有结果；再次保存新实验时修改顶部 `OUTPUT_DIR`。产物仍只有模型、测试预测 CSV 和 summary。Summary 仅保留主要模型参数、种子、样本数和验证/测试指标表。

本轮没有重复真实数据的全量训练。已有入口测试改为通过顶部配置传入临时数据，验证完整划分、实际 100 棵树、单次 train-only 拟合、保存后的评估顺序和模型重载预测。网格脚本仍可安全导入指标函数。

```text
.venv/bin/python -B -m pytest -q
24 passed in 0.85s

git diff --check
exit code 0
```
