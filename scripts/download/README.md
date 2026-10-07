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
| biodeeptime | 原始说明与用户提供的 `6-7(1).zip`，其中有完整 D1/biodeeptime.csv 和同批 CCS 资料 |
| yago2geo | 原始 yago2_plus_geo.zip，约 4.2 GB |
| earthse | Earth-Iron / Earth-Silver，各四个固定 revision parquet |
| earthbench | 旧项目使用的 49 个问题目录、1415 个原始资产，以及原版 question.json；不是全量影像仓库 |
| geoquestions1089 | 完整原版问题 JSON 和答案 JSON |
| disastervqa | Zenodo 20306919 的官方 Dataset-v3 ZIP |
| rscc | RSCC 官方 xbd_subset.tgz 原始包，对应旧中文包装 ZIP 内的同名包 |
| geotooleval | 项目自编工作簿、资源表与原始问题 JSON，需本地导入 |
| unidentified_sources | 115871.nc、D4.xlsx，来源身份待确认，仅本地保全 |

版本、URL、校验和维护在 `manifests/*.json`，数据文件本身被 Git 忽略。旧版 GEM、自编资料及不明来源文件没有已确认的同版本公开直链，脚本要求本地导入。下载全量不代表这些本地项能在全新机器上自动恢复。

大于 8 MiB 且已知大小的文件使用最多四个并发 HTTP 分段请求，核对每段 Content-Range 与实际长度，再校验完整文件哈希。服务端不支持分段时回退到普通下载。分段失败会重试，不接受截断内容；下载步骤不会解压或重新包装数据。

NASA、USGS、World Bank 和 OWID 的部分上游地址会变化；manifest 锁定旧项目原始字节，若服务器内容更新则明确失败。World Bank ZIP 甚至可能因打包时间变化而哈希不同。复现实验应导入旧文件；更新版本需另行审核 URL、校验和及后续 pipeline，不自动切换到最新版本。

## 本次迁移与手动验证

原始文件从旧项目和同级 `GeoToolAgent数据集` 直接复制，源文件保留。逐文件来源、目标、大小、SHA-256 与缺失项记录在被 Git 忽略的 `outputs/download_migration/migration.json`。

首次迁移复制了 1,460 个原始文件，共 10,250,613,992 字节，覆盖 22 个分类，当时 18 个分类完整。BioDeepTime 完整 CSV 位于用户提供的原始 `6-7(1).zip` 中，保留原包，不在下载步骤解压。这个包装包没有公开同字节下载地址，需本地导入；官方 Zenodo 8154672 发布包有不同的包装方式。

首次迁移缺少原始 Earth-Agent question.json、GeoQuestions1089 两份完整 JSON、官方 DisasterVQA ZIP 和旧版 RSCC 包，共 5 项，涉及 4 个分类。随后按用户要求从官方来源重新获取，最新结果见 `outputs/download_migration/fresh_downloads.json` 和 `verification_summary.json`。官方 DisasterVQA ZIP 与旧项目中文包装包并非相同文件。

缺失内容现已全部补齐：重新获取 5 个文件，共 1,672,087,625 字节；当前 22 个分类的 1,465 个文件完整，共 11,922,701,617 字节。PowerShell 和 Windows Git Bash 的统一 `--verify-only` 均返回 0。原始迁移时的缺失清单保留在 migration.json 的 missing_at_initial_migration 字段，当前 missing 为空。DisasterVQA 另外完整检查了 ZIP 成员 CRC；RSCC 原始 tarball 可完整读取。

RSCC 直接保存官方团队提供的 `xbd_subset.tgz`，替代不可恢复的中文外层 ZIP；大小和 CRC32 与旧包装 ZIP 中的成员记录一致，下载后锁定 SHA-256。manifest 的 legacy_wrapper 字段保留外层旧包指纹以供追溯。获取说明见 [RSCC 官方数据卡](https://huggingface.co/datasets/richkoala/RSCC)。原始包内的影像使用仍遵循上游 xBD 许可。

不为这些脚本新增测试文件；手动验证日志记录在 `outputs/download_migration/`。首次验证使用本地复制的文件和本机 HTTP 服务的小文件；补齐缺失数据时实际运行下载 CLI 并校验官方文件。项目迁移到 Ubuntu-24.04 WSL 后，原生 Linux 的 Bash 统一校验也已通过：22 个分类、1,465 个文件，退出码 0。

已手动检查本地 HTTP 获取、错误内容/截断响应拒绝、重试、临时文件清理、已有文件保留、从其他工作目录调用 CLI、本地导入以及两个统一入口；分段获取也手动检查了并发组装、普通响应回退与错误 Content-Range 拒绝。HEAD 已确认 NASA GISTEMP 与 World Bank 当前响应大小不同于旧文件，这两项联网重建会被版本校验拒绝；本地旧版原始数据完整并通过校验。HEAD 可达性不等于完整远程下载及校验通过。
