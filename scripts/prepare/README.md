# 数据准备

`data/raw/` 保存原始下载、来源记录和处理临时文件。所有整理结果放在 `data/`：单文件数据集直接保存，多文件数据集使用各自目录。默认从原始缓存处理；旧版找不到时使用可用新版，记录实际来源和哈希，不自动使用旧结果快照。

## 运行

从 Git 获取 GEM 数据后，可以仅使用 Python 标准库解压，无需安装项目依赖：

```bash
python3 scripts/prepare/prepare_gem.py
```

默认读取 `data/gem.jsonl.gz`，展开为被 Git 忽略的 `data/gem.jsonl`，并记录准备清单。可通过 `--source-root` 指定包含压缩文件的数据目录，通过 `--output-root` 指定接收 `data/` 的项目根目录；成功校验 gzip 和 JSONL 后才替换输出文件，解压失败会保留已有文件。

在仓库根目录、安装 scientific extra 的环境中执行：

```bash
.venv/bin/python scripts/prepare/prepare_all.py
.venv/bin/python scripts/prepare/prepare_all.py --offline
.venv/bin/python scripts/prepare/prepare_all.py --offline --output-root outputs/prepare-check
.venv/bin/python scripts/prepare/prepare_ocean.py --offline
```

总入口顺序执行 `prepare_gem.py`、`prepare_deep_earth.py`、`prepare_geotoolkg.py`、`prepare_answer_formats.py`、`prepare_geoquestions.py`、`prepare_mineral.py`、`prepare_climate.py`、`prepare_energy.py`、`prepare_ocean.py`、`prepare_biodeeptime.py`、`prepare_ccs.py`、`prepare_remote_sensing.py`、`prepare_hazard.py`。

- `--output-root`：接收 `data/` 整理结果的项目根目录，默认本仓库。
- `--cache-root`：原始缓存，默认本仓库 `data/raw/`。
- `--source-root`：需要额外保留的整理数据来源，默认本仓库 `data/`；适用于原附件、手工资产、深地和 KG 输入。
- `--offline`：仅使用现有原始缓存。
- `--archive-threshold-mb`：多文件数据集超过该大小时额外保存 ZIP，默认 64 MiB。
- `--archive`：对应领域的独立 prepare 入口从指定 ZIP 校验并解压，产物仍在 `data/<数据集>/`。

ZIP 包放在 `data/<数据集>.zip`。对应 prepare 脚本负责展开到 `data/<数据集>/`，运行时始终读取整理后的数据目录。展开目录属于处理结果，不放进原始数据区；不存在 `raw/prepared` 工作模式。

```bash
.venv/bin/python scripts/prepare/prepare_remote_sensing.py --archive data/earthbench.zip
.venv/bin/python scripts/prepare/prepare_geotoolkg.py --archive data/geotoolkg.zip
```

对深地、KG、BioDeepTime、CCS，若源目录只有 ZIP，prepare 会自动校验并展开。其他数据默认从原始来源重新处理；可以显式使用 `--archive` 展开已准备的数据包。ZIP 使用稳定文件顺序和时间戳，内含逐文件 SHA-256 清单；解压校验路径及内容。多文件目录仍是可直接读取的本地处理结果，生成的数据包和展开目录均不纳入 Git；需要额外保留的输入包按 `.gitignore` 的精确例外维护。

## 整理结果

| 数据集 | 整理结果 | 默认准备方式 |
| --- | --- | --- |
| GeoToolEval | `data/geotooleval.jsonl` | 用户提供的原记录，保留 603 题、全部原字段 |
| GEM 原始观测 | `data/gem.jsonl.gz`，展开为 `data/gem.jsonl` | 完整保留原工作簿各工作表、行序、单元格值及公式缓存；Git 保存压缩 JSONL |
| 矿产 | `data/mineral.csv` | 从 MRDS 按原 8,000 个记录 ID 选择 |
| 灾害 | `data/hazard.jsonl` | 额外保留的原灾害附件完整记录，10,764 行、47 列 |
| 回答格式 | `data/answer_formats/` | 从 GeoToolEval 公开问题、格式和说明生成 |
| GeoQuestions | `data/geoquestions/` | 从上游原文件生成 40 条问题和查询结果 JSONL |
| 深地 | `data/deep_earth/` | 保留现有模型采样、USGS 目录及来源说明 |
| KG | `data/geotoolkg/` | 保留原索引 JSON 和导入说明 |
| 气候 | `data/climate/` | NOAA 观测、NASA 年表、IPCC SSP 表，4 张 CSV |
| 能源 | `data/energy/` | 从 GEM、MRDS、OPSD、OWID、World Bank 生成 11 张 CSV |
| 海洋 | `data/ocean/` | 从 SOCAT、GLODAP、SNAPO 原始数据生成 9 张 CSV |
| BioDeepTime | `data/biodeeptime/` | 原观测子集、系列汇总与手工区界 |
| CCS | `data/ccs/` | 保留空间资产，公开表从原始来源整理；许可与 schema 放在 metadata.json |
| 遥感 | `data/earthbench.zip`、展开目录 `data/earthbench/` | 固定 49 个目录的 1,415 个原文件及资产索引；影像无损保留 |

表格统一为 UTF-8 CSV，换行符为 LF；空间数据使用 GeoJSON，复杂记录使用 JSONL，索引和来源说明使用 JSON。重新序列化保留行、列、数值与区界，不强制旧附件的 BOM 或换行字节。

公共来源下载到 `data/raw/<来源>/`，模型缓存为 `data/raw/models/fastembed/`。能源准备直接读取 `data/gem.jsonl` 或 `data/gem.jsonl.gz`，不依赖 XLSX。首次导入新附件时运行 `prepare_gem.py --workbook data/raw/gem/文件名.xlsx`；平时 `prepare_gem.py` 校验并展开 Git 保留的压缩 JSONL。World Bank 旧缓存缺失时读取当前 API ZIP，按内容寻找指标 CSV。NASA 年表从当前原始月表的 J-D 列生成，仅保留完整年度值。

## Git 与校验

Git 放行 GeoToolEval、GEM 压缩 JSONL、原灾害记录、BioDeepTime 原子集/区界、CCS 原空间资产、深地输入、KG JSON，以及 `data/metadata/geotooleval.json` 的原附件来源说明。对应多文件输入的 ZIP 形式也允许保留。可生成的矿产、气候、能源、海洋、GeoQuestions、回答格式、遥感包与准备清单均被忽略；`data/raw/` 始终被忽略。

GeoToolEval 其他工作表内容与来源清单统一保存在 `data/metadata/geotooleval.json`，主数据集仍只有一个 JSONL。资源定位兼容记录为 `data/metadata/resources.json`，用于把原题资源路径映射到新布局。

各领域清单为 `data/metadata/prepare_<领域>.json`，总清单为 `prepare_all.json`。清单记录实际源哈希、处理代码/参数哈希和结果哈希。成功返回 0；缺输入返回 2，标记 partial；处理异常返回非零。总入口继续其他领域，保留各项失败状态。

固定参考资料已从 src 移到 data：`data/reference/energy.json` 保存首都坐标、区域、国家/能源类型/关键矿产分类和容量序列选择；`ocean.json` 保存海洋区域；`geography.json` 保存地球半径；`data/geoquestions/profile.json` 保存固定查询名单及来源指纹。这些都是 Git 必须保留的输入。prepare 会复制这些参考文件并记录哈希；计算公式、质控算法、协议常量和路径映射仍由代码维护。
