# 原始数据获取

所有脚本只获取原始文件，不解压、不筛选、不清洗、不生成问答。Python 仅使用标准库；统一入口显式使用项目 `.venv`。用 `uv sync` 准备环境。

Windows：

```powershell
pwsh -File scripts/download/download_all.ps1
pwsh -File scripts/download/download_all.ps1 --verify-only
.venv/Scripts/python.exe scripts/download/download_socat.py --help
```

Linux：

```bash
bash scripts/download/download_all.sh
bash scripts/download/download_all.sh --verify-only
.venv/bin/python scripts/download/download_socat.py --help
```

可以从任何工作目录调用脚本。脚本从自身位置向上查找 `pyproject.toml`，默认输出为 `data/<dataset>/`；所有相对参数路径均相对于仓库根目录。Windows Git Bash 也可运行 `.sh`，会使用 `.venv/Scripts/python.exe`。

每个 CLI 支持 `--output-dir`、`--source-dir`、`--verify-only`、`--list`、`--timeout` 和 `--retries`。统一入口会依次运行所有 `download_*.py`，继续报告后续数据集，任何数据集失败时整体返回非零；统一入口参数会原样传递给各脚本。`--output-dir` 和 `--source-dir` 建议只用于单个数据集，以免多个数据集混在同一目录。

```powershell
.venv/Scripts/python.exe scripts/download/download_gem.py --source-dir D:/originals/gem
```

本地导入目录内部使用 manifest 的相对文件名，例如 EarthSE 要有 `Earth-Iron/*.parquet`。已有文件逐个校验哈希与大小，通过则跳过；校验失败会报错并保留原文件，不会悄悄替换。缺失文件下载到临时文件，校验后才发布；失败删除该临时文件。无需关闭 TLS 证书验证，若本机证书链异常请修复 Python/系统证书配置。

## 数据集与输出

| CLI 后缀 / data 子目录 | 原始输入与范围 |
| --- | --- |
| litho1 | LITHO1.0 的 litho1pt0 1.5.0 wheel，保留包，不安装 |
| semucb | 官方 SEMUCB-WM1 固定 Git commit ZIP |
| usgs | 2010–2020、M≥7 的旧版冻结目录 CSV |
| climate | Bintanja 2005、Friedrich 2016、NASA GISTEMP 原始文件 |
| mrds | USGS MRDS 原始 CSV ZIP |
| opsd | 2020-08-25 可再生电站与容量 CSV |
| gem | Global Integrated Power March 2026 II 原始 XLSX |
| world_bank | EG.FEC.RNEW.ZS 原始下载 ZIP |
| population | OWID 人口及 UN 预测原始 CSV |
| glodap | GLODAP v2.2023 太平洋 CSV |
| socat | SOCAT v2026 北太平洋 TSV、月度网格 CSV |
| snapo | SNAPO 原始 113597.xlsx |
| ccs | Zenodo 17003094 的 CCS v1.0 数据、schema 与说明 |
| biodeeptime | Zenodo 8154672 的说明与完整 CSV ZIP |
| yago2geo | 原始 yago2_plus_geo.zip，约 4.2 GB |
| earthse | Earth-Iron / Earth-Silver，各四个固定 revision parquet |
| earthbench | 旧项目使用的 49 个问题目录、1415 个原始资产，以及原版 question.json；不是全量影像仓库 |
| geoquestions1089 | 完整原版问题 JSON 和答案 JSON |
| disastervqa | Zenodo 20306919 的官方 Dataset-v3 ZIP |
| rscc | 旧版 RSCC 用户原始包装 ZIP，需本地导入 |
| geotooleval | 项目自编工作簿、资源表与原始问题 JSON，需本地导入 |
| unidentified_sources | 115871.nc、D4.xlsx，来源身份待确认，仅本地保全 |

版本、URL、校验和维护在 `manifests/*.json`，数据文件本身被 Git 忽略。旧版 GEM、RSCC、自编资料及不明来源文件没有已确认的同版本公开直链，脚本要求本地导入。下载全量不代表这些本地项能在全新机器上自动恢复。

NASA、USGS、World Bank 和 OWID 的部分上游地址会变化；manifest 锁定旧项目原始字节，若服务器内容更新则明确失败。World Bank ZIP 甚至可能因打包时间变化而哈希不同。复现实验应导入旧文件；更新版本需另行审核 URL、校验和及后续 pipeline，不自动切换到最新版本。

## 本次迁移与手动验证

原始文件从旧项目和同级 `GeoToolAgent数据集` 直接复制，源文件保留。逐文件来源、目标、大小、SHA-256 与缺失项记录在被 Git 忽略的 `outputs/download_migration/migration.json`。

尚未找到完整 BioDeepTime CSV ZIP、原始 Earth-Agent question.json、GeoQuestions1089 两份完整 JSON、官方 DisasterVQA ZIP 和旧版 RSCC 包。脚本保留对应获取定义，离线校验应返回失败，不以旧项目生成的问答子集冒充原始数据。官方 DisasterVQA ZIP 与旧项目中文包装包并非相同文件。

不为这些脚本新增测试文件；手动验证日志记录在 `outputs/download_migration/`。已有原始数据的校验通过仅说明迁移和跳过逻辑通过；网络获取路径另用本机 HTTP 服务的小文件手动检查，不重新下载真实数据集。Linux 原生运行需要后续在 Linux 主机验证，当前 Bash 运行验证使用 Windows Git Bash。
