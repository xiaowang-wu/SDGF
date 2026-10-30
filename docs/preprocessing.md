# 数据来源与处理流程

本项目使用 Panama、Australia 和 Dominican Republic 三组负荷数据，依次编号为 Load 1、Load 2、Load 3。实验输入均为作者在来源数据上处理得到的表格。对应代码为 [sdgf/preprocessing.py](../sdgf/preprocessing.py)，命令行入口为 `python -m sdgf prepare`。

## 来源链接

| 编号 | 数据集 | 来源 | 项目内链接文件 |
| --- | --- | --- | --- |
| Load 1 | Panama | [Kaggle: Electricity Load Forecasting](https://www.kaggle.com/datasets/saurabhshahane/electricity-load-forecasting) | [load_1_links.txt](../data/sources/load_1_links.txt) |
| Load 2 | Australia | [IEEE DataPort: Power Load Forecasting](https://ieee-dataport.org/documents/power-load-forecasting) | [load_2_links.txt](../data/sources/load_2_links.txt) |
| Load 3 | Dominican Republic | [Kaggle: Short-Term Forecasting Dataset SENI Dominican Rep](https://www.kaggle.com/datasets/rafaeluceta/short-term-forecasting-dataset-seni-dominican-rep) | [load_3_links.txt](../data/sources/load_3_links.txt) |

Load 2 的来源链接来自作者保存 Australia.csv 时的同目录来源记录。发布整理时该站点限制了自动读取，因此保留原始来源链接，不推测其 DOI 或下载权限。数据文件需要按各平台的要求获取，原始文件和处理后数据均不包含在源码压缩包中。

## Load 1：Panama

输入文件为 `continuous_dataset.csv`，时间列为 `datetime`，目标为 `nat_demand`。处理流程：

1. 解析时间戳并按时间升序排序，保留原始小时观测。
2. 保留三地气象变量及历史负荷；保留来源文件中的 `Holiday_ID`、`holiday`、`school`，不重新推测节假日或学校日历。
3. 从时间戳提取小时 `hour`、月份 `month` 和星期，生成小时、月份的正弦/余弦特征及周末标记。
4. 输出配置要求的列。已有预测列与本方法无关，不需要先训练 GRU。

周期编码采用：

```text
hour_sin  = sin(2*pi*hour/24)
hour_cos  = cos(2*pi*hour/24)
month_sin = sin(2*pi*month/12)
month_cos = cos(2*pi*month/12)
is_weekend = 1 if weekday >= 5 else 0
```

`month` 为 1–12，周一至周日的 `weekday` 为 0–6。按照原处理代码，Load 1 的小时、月份编码以 float32 计算。若使用保存在其他来源包中的 Panama 表格，处理代码也接受 `date -> datetime`、`pl -> nat_demand` 的列名映射；不同来源文件的小数精度可能不同，复现实验优先使用记录的原始版本。

```bash
python -m sdgf prepare --dataset load_1 --input data/raw/continuous_dataset.csv --output data/load_1.csv
```

## Load 2：Australia

输入为 `Australia.csv`，包含 `date, Dbt, Dpt, Wbt, Hum, pl`，其中 `pl` 为目标。来源文件为半小时序列，处理方式为：

1. 按年/月/日解析 `date` 并排序。
2. **仅保留分钟数为 0 的记录**。这是整点抽取，保留对应时刻的原始负荷与气象值，没有对两条半小时记录求平均。
3. 添加小时、月份周期编码和周末标记。编码公式与 Load 1 相同，采用 float64 计算。
4. 输出五个时序变量、五个已知协变量及时间戳。

作者保存的来源文件为 87,648 行，处理后为 43,824 行；范围为 2006-01-01 01:00 至 2011-01-01 00:00。保留自然时间对应的时间戳，不人为平移半小时。

```bash
python -m sdgf prepare --dataset load_2 --input data/raw/Australia.csv --output data/load_2.csv
```

## Load 3：Dominican Republic

输入为 `datos_modelo.csv`，时间列为 `timestamp`，目标为 `active_power_mw`。处理流程：

1. 按 `%d/%m/%y %H:%M` 解析原始日期，例如 `2/1/21 1:00` 表示 **2021 年 1 月 2 日 01:00**。
2. 按正确时间排序，并检查小时连续性。
3. 从修正后的时间戳重新提取小时和月份，重新计算周期编码，不直接沿用可能基于错误日期解析生成的编码。
4. 小时编码使用 `hour % 24`；月份编码使用 `month - 1`，使 1 月对应周期起点。
5. 保留来源文件已有的 `is_holiday`、`is_weekend`、`is_school_vacation`。保留负荷、体感温度和云量；不使用来源文件预先生成的负荷滞后列。

```text
hour_sin  = sin(2*pi*(hour % 24)/24)
hour_cos  = cos(2*pi*(hour % 24)/24)
month_sin = sin(2*pi*(month - 1)/12)
month_cos = cos(2*pi*(month - 1)/12)
```

```bash
python -m sdgf prepare --dataset load_3 --input data/raw/datos_modelo.csv --output data/load_3.csv
```

处理代码也支持已经正确解析时间的中间 XLSX 文件；从原始 CSV 开始处理时，需要保留上述日/月解析约定。

## 数据准备与模型训练的边界

数据准备阶段不做标准化、不填补缺失值、不剔除异常值、不生成其他模型的预测。程序对重复时间戳、时间间隙及选中数值列的缺失值/无穷值报错，避免悄悄改变时间步含义。

模型训练阶段再构造过去 24 个时间步的窗口，按时间顺序划分 70%/15%/15% 的训练/验证/测试样本，仅在训练样本上拟合 StandardScaler。目标保留原单位。数据准备不提前使用整个数据集计算归一化参数。

每次准备数据都会写出 `<输出文件>.preprocess.json`，记录来源链接、源文件名、输入输出 SHA-256、处理前后行数、时间范围、输出列和实际执行的操作。命令不会覆盖输入文件或已有输出。

对于不同的源日期格式，可以明确指定格式，例如：

```bash
python -m sdgf prepare --dataset load_3 --input data/raw/datos_modelo.csv --output data/load_3.csv --date-format "%d/%m/%y %H:%M"
```

## 已处理文件与训练配置

| 编号 | 目标 | 时序特征数 | 已知协变量数 | 小时观测行数 |
| --- | --- | --- | --- | --- |
| Load 1 | nat_demand | 13 | 8 | 48,048 |
| Load 2 | pl | 5 | 5 | 43,824 |
| Load 3 | active_power_mw | 3 | 7 | 35,039 |

完整特征顺序以 `configs/load_1.json`、`configs/load_2.json`、`configs/load_3.json` 为准。处理命令输出可直接用于相应训练命令，示例中的数据文件路径都相对于当前工作目录。

## 超参数

三组配置文件显式列出全部模型与训练超参数。参数含义、固定模型结构、学习率调度和修改方式见 [hyperparameters.md](hyperparameters.md)。各次训练实际采用的完整配置保存在输出目录的 `config.json` 中。
