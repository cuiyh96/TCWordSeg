# TCWordSeg

中文分词与文本去重工具包，服务于“大模型训练数据”的预处理阶段。

## 项目结构

```
TCWordSeg/
├── wordseg/                      # 分词库（Python 包）
│   ├── __init__.py               # 包公共 API 出口（segment / segment_with_pos 等）
│   ├── tokenizer.py              # 统一分词入口（仅按环境变量分发，隐藏引擎细节）
│   ├── tcwordseg/                # 引擎一：原生引擎（默认），引擎私有文件都在本子包内
│   │   ├── __init__.py           # 引擎实现：init / uninit / is_available / segment / segment_with_pos
│   │   ├── TCWordSeg.py          # SWIG 自动生成的 Python 绑定层
│   │   ├── TCWordSeg3.py         # Python3 面向对象封装
│   │   ├── _TCWordSeg.pyd        # C/C++ 分词引擎（Windows，锁定 CPython 3.8）
│   │   ├── _TCWordSeg.so         # C/C++ 分词引擎（Linux，锁定 CPython 3.8）
│   │   └── data/                 # 词典资源文件
│   │       ├── Core.Lex              # 核心词典
│   │       ├── Core.Prop             # 核心属性
│   │       ├── Ngram.Lex             # N-gram 语言模型
│   │       ├── Ngram_PN.Lex          # N-gram PN 模型
│   │       ├── POSTT.Lex             # 词性标注模型
│   │       ├── RoleTT.Lex            # 角色标注模型
│   │       ├── LNRoleTT.Lex          # LN 角色标注模型
│   │       ├── Sec.Lex / Sec.Prop    # 粗粒度分词词典
│   │       ├── UserAmbRule           # 用户消歧规则
│   │       ├── id_custname.dict      # 自定义人名词典
│   │       ├── cusdata.idx / .idxnull # 自定义词典索引
│   │       ├── ner_info              # 命名实体识别信息
│   │       └── t2s / t2s.word / t2s.disable  # 繁简转换
│   └── jieba/                    # 引擎二：jieba，与 tcwordseg 平级
│       ├── __init__.py           # 引擎实现（支持多张自定义分词表）
│       └── data/                 # 自定义词表目录，其下全部 *.txt 自动加载
│           ├── userdict.txt      # 样例词表：项目 / 大模型语料术语
│           ├── supplychain.txt   # 样例词表：供应链领域
│           └── hr.txt            # 样例词表：人力领域
├── dedup/                        # 去重包（MinHash + LSH，独立于分词，可单独复用）
│   └── __init__.py
├── pipelines/                    # 语料处理脚本（Step 1/2 为去重链路，Step 3 为独立分词工具）
│   ├── compute_minhash.py        # Step 1：MinHash 签名计算（内部调用分词，--save_id_map 可选落地 id 映射）
│   ├── dedup_by_lsh.py           # Step 2：LSH 近似去重
│   └── segment_corpus.py         # Step 3：多进程批量分词（独立使用，非 Step 1/2 的前置）
├── tests/                        # 测试
│   ├── test_dedup.py             # 去重核心单元测试（无需原生引擎）
│   ├── test_seg_snapshot.py      # 分词行为快照回归
│   └── notebooks/                # 探索用 notebook
├── output/                       # 运行产物输出（已 gitignore）
├── .gitignore
├── pytest.ini                    # 测试配置
└── README.md
```

## 处理流程

```
原始语料(data/raw)
├─ pipelines/segment_corpus.py ─→ 分词结果 {名}.jsonl（title / content / content_seg）
│                                 经 wordseg.tokenizer 调用，引擎由 TCWORDSEG_ENGINE 决定
└─ pipelines/compute_minhash.py ─→ hashvalues.npy
       内部：extract_sft_text（按 s1/s2 抽取文本） → tokenizer.segment 分词 → MinHash 签名
                          │
                          ▼
          pipelines/dedup_by_lsh.py ─→ 保留索引 keep_indexes / 重复映射 dup_map
                         （LSH 近似判重，Jaccard 相似度 ≥ threshold）
```

模块职责边界：

| 模块 | 职责 | 与其它模块的关系 |
| -- | -- | -- |
| `wordseg/` | 只负责分词，不感知去重 | 引擎由环境变量选择，对外只有 `tokenizer` 的 5 个接口 |
| `dedup/` | 只负责去重，不感知引擎 | 分词通过 `text_to_words` 参数注入，默认 `tokenizer.segment` |
| `pipelines/` | 只负责 IO 与并发调度，把前两者串成可执行步骤 | 脚本内把仓库根加入 `sys.path` 后 import 前两者 |

引擎切换点全项目只有环境变量 `TCWORDSEG_ENGINE` 一处；去重核心、脚本、测试对引擎零感知（详见文末「两种引擎的分词原理与差异」）。

## 核心功能

### 1. 中文分词引擎

分词底层有两种可切换的实现，默认是自研 C/C++ 引擎（通过 SWIG 生成 Python 绑定），设 `TCWORDSEG_ENGINE=jieba` 可切到 jieba。
原生引擎的主要特性：

- **词性标注** — 支持名词(n)、动词(v)、形容词(a)等 40+ 词性标签
- **自定义词类** — 支持用户自定义词典和分类识别
- **命名实体识别** — 人名、地名、机构名等
- **繁简转换** — 繁体中文到简体中文自动转换
- **数字/时间/邮箱/电话识别** — 特殊模式识别

#### 使用示例

```python
from wordseg import segment, segment_with_pos, current_engine

# 分词（引擎资源按需自动加载，每个进程仅需一次；多进程的每个子进程自行调用即可）
segment("用友网络科技股份有限公司")
# ['用友', '网络', '科技', '股份', '有限公司']

# 分词 + 词性标注
segment_with_pos("用友网络科技股份有限公司")
# ['用友/nr', '网络/n', '科技/n', '股份/n', '有限公司/n']

current_engine()   # 'tcwordseg'（默认）
```

> [tokenizer.py](wordseg/tokenizer.py) 是唯一入口，两种引擎实现（`wordseg.tcwordseg`、`wordseg.jieba`）
> 在包内平级、对外接口一致，`seghandle`、gb18030 编码等引擎细节都封装在内；
> 判断当前引擎能否加载用 `is_available()`，查看当前生效的引擎名用 `current_engine()`。
> 直接操作底层绑定（`TCWordSeg3`）仅在确有必要时使用。

#### 切换分词引擎（tcwordseg / jieba）

用环境变量 `TCWORDSEG_ENGINE` 在两种引擎间切换，默认 `tcwordseg`：

```bash
# Windows（PowerShell）
$env:TCWORDSEG_ENGINE = "jieba"
# Linux / macOS
export TCWORDSEG_ENGINE=jieba
```

| 引擎    | 取值              | 说明                               |
| ----- | --------------- | -------------------------------- |
| 原生引擎  | `tcwordseg`（默认） | 必须 Python 3.8（如虚拟环境 `minhash_py38`），词性标注更丰富 |
| jieba | `jieba`         | 任意已安装 jieba 的环境，纯 Python，支持自定义分词表 |

切换对 `tokenizer.segment` / `segment_with_pos` 以及去重管线（`dedup`、各 `pipelines/`）完全透明；
多进程（Windows spawn）时子进程会继承该环境变量，因此无需改代码即可整体切换。

jieba 引擎支持多张自定义分词表（jieba 用户词典），按以下优先级解析：

1. 显式传入的路径：`wordseg.jieba.init(userdict_path=...)`，也可传路径序列；
2. 环境变量 `TCWORDSEG_JIEBA_DICT`，多个路径用 `os.pathsep` 分隔（Windows 为 `;`，Linux/macOS 为 `:`）；
3. 默认目录 `wordseg/jieba/data` 下的全部 `*.txt`（仓库自带样例词表，零配置即生效）。

`data/` 下自带三张样例词表，可直接改用自己的词表，或新增 `*.txt` 自动生效（按文件名排序加载）：

| 词表 | 领域 | 词条数 |
| -- | -- | -- |
| `userdict.txt` | 项目 / 大模型语料术语 | 12 |
| `supplychain.txt` | 供应链 | 10 |
| `hr.txt` | 人力 | 15 |

路径可以是文件或目录——目录会展开为其下全部 `*.txt`（按文件名排序），便于按业务拆多张词表；
显式指定的路径不存在会抛 `FileNotFoundError`，默认目录不存在则视为没有自定义词表。
词典格式与 jieba 一致，每行「词语 [词频] [词性]」，词频、词性可省略，**不支持注释行**：

```
用友网络 100 nz
大模型训练 200 nz
```

```bash
# 多张词表：Windows（PowerShell）
$env:TCWORDSEG_JIEBA_DICT = "dicts/org.txt;dicts/biz.txt"
# Linux / macOS
export TCWORDSEG_JIEBA_DICT=dicts/org.txt:dicts/biz.txt
```

```python
from wordseg import jieba

jieba.init(userdict_path='my_userdict.txt')    # 单张词表（文件或目录）
jieba.init(userdict_path=['a.txt', 'b.txt'])   # 多张词表（多进程场景建议改用环境变量）
jieba.segment('用友网络')                       # ['用友网络']
jieba.resolve_dict_files()                     # 查看当前会加载哪些词表文件
```

### 2. 文本去重管线

基于 MinHash + LSH 的大规模文本去重，用于大模型训练数据清洗：

```
原始数据 → 分词 → MinHash 签名(128个哈希函数) → LSH 索引(Jaccard相似度≥0.95) → 去重结果
```

核心逻辑集中在 [dedup/](dedup/__init__.py)，作为独立包可单独调用（分词通过参数注入，默认用 `tokenizer.segment`）：

```python
import dedup

data = dedup.load_jsonl('data.jsonl')
texts = [dedup.extract_sft_text(item, 's2') for item in data]

hashvalues = dedup.calc_hashvalues(texts, num_perm=128, num_processes=32)
result = dedup.dedup_by_lsh(hashvalues, threshold=0.95)

result.keep_indexes   # 保留的数据索引（升序）
result.dup_map        # 重复索引 -> 最终保留索引
```

#### 去重流程

**Step 1：计算 MinHash 签名**

```bash
# 从文件读取数据，输出 hashvalues.npy
python pipelines/compute_minhash.py \
    --input_path data.jsonl \
    --num_perm 128 \
    --data_fmt s2 \
    --num_processes 10

# 额外落地 id 与哈希值的映射文件 hashvalues_info.jsonl
python pipelines/compute_minhash.py \
    --input_path data.jsonl \
    --save_id_map
```

**Step 2：LSH 去重**

```bash
python pipelines/dedup_by_lsh.py \
    --data_path data.jsonl \
    --threshold 0.95
```

**Step 3：多进程批量分词（独立工具，不参与上面 Step 1/2 的去重链路）**

```bash
# 指定清单：只处理清单内的文件，按清单标注的列号取标题与内容
python pipelines/segment_corpus.py \
    --filelist filelist1.txt \
    --dir_path data/raw \
    --output_path output/seg

# 不指定清单：目录下所有文件都按整行不分列处理
python pipelines/segment_corpus.py \
    --dir_path data/raw \
    --output_path output/seg
```

分词同样走统一入口，可先设 `TCWORDSEG_ENGINE=jieba` 再执行以改用 jieba。
输出为 `{文件名}.jsonl`，每行含 `title` / `content` / `content_seg`（分词结果，空格分隔）。

## 数据格式

去重管线支持两种 SFT 数据格式：

| 格式 | 字段                                  | 说明     |
| -- | ----------------------------------- | ------ |
| s1 | instruction, input, output, history | 单轮对话格式 |
| s2 | conversations                       | 多轮对话格式 |

## 环境

原生引擎（`TCWORDSEG_ENGINE=tcwordseg`，默认）依赖扩展 `_TCWordSeg`，该扩展由外部提供、
**无源码、锁定 CPython 3.8，不可重建**；jieba 引擎（`TCWORDSEG_ENGINE=jieba`）为纯 Python、无版本限制。
因此各部分的版本要求不同：

| 部分                                             | 版本要求                              |
| ---------------------------------------------- | --------------------------------- |
| 分词（`tcwordseg` 引擎）                             | **必须 Python 3.8**，本项目统一用虚拟环境 `minhash_py38` |
| 分词（`jieba` 引擎）                                 | **任意已安装 jieba 包的环境**（Python 3.7+）  |
| 去重核心（`dedup/`、`pipelines/dedup_by_lsh.py`） | 任意 Python 3.7+                    |

```bash
# tcwordseg 引擎（默认）：必须切到 Python 3.8 环境，本项目统一用 minhash_py38
conda activate minhash_py38
python -c "from wordseg import tokenizer; print(tokenizer.current_engine(), tokenizer.is_available())"

# jieba 引擎：任何已安装 jieba 的环境均可，不要求 3.8
$env:TCWORDSEG_ENGINE = "jieba"   # Windows（PowerShell）
export TCWORDSEG_ENGINE=jieba     # Linux / macOS
```

> 简言之：**跑 tcwordseg 就用 `minhash_py38`**（`_TCWordSeg` 扩展锁定 CPython 3.8）；
> **跑 jieba 则不限环境**，只要该环境里 `pip install jieba` 过即可。

其他 Python 版本下 `import wordseg.tokenizer` 不会报错（延迟导入），
但使用默认引擎调用 `segment()` 时会抛出带修复提示的 `RuntimeError`；
可用 `tokenizer.is_available()` 提前判断，或改用 jieba 引擎。

## 测试

```bash
# 去重核心单元测试（不依赖原生引擎，任意 Python 版本可跑）
python -m pytest tests/test_dedup.py

# 分词行为快照测试（需 Python 3.8 环境，如 minhash_py38；PowerShell 下用 $env:TCWORDSEG_SNAPSHOT_RECORD="1"）
TCWORDSEG_SNAPSHOT_RECORD=1 python -m pytest tests/test_seg_snapshot.py  # 首次生成基线
python -m pytest tests/test_seg_snapshot.py                             # 回归比对
```

`tests/test_seg_snapshot.py` 以「固定输入 → 期望切词结果」的快照作为回归基线，
更换 Python / 平台 / 引擎时用它验证"是否仍是同一个分词器"；基线文件为 `tests/seg_snapshot.jsonl`，
需在 Python 3.8 下首次生成。

## Python 依赖

| 包名         | 用途                                         |
| ---------- | ------------------------------------------ |
| numpy      | 哈希值数组计算与存储                                 |
| datasketch | MinHash 签名与 LSH 近似去重                       |
| jsonlines  | JSONL 格式读写                                 |
| jieba      | jieba 分词引擎（仅 `TCWORDSEG_ENGINE=jieba` 时需要） |

## 附：两种引擎的分词原理与差异

### 共同原理：词典 + 最大概率路径

两者属于同一算法族：先按词典列出句子里所有**可能成词的片段**，再用动态规划取整句**得分最高的一条切分路径**，得分以**词频**（该词在语料中的出现概率）为权重。所以「整体成词」能否赢过「拆开切」，取决于两边词频的相对大小，而不是词表里有没有这个词。差别在于：这份权重写在哪儿、用户能不能改。

### jieba（纯 Python，`wordseg/jieba`）

1. **前缀词典**：主词典维护 `FREQ[词] = 词频`（随 jieba 包分发，不在本仓库），自定义词表按行增量写入同一个词典；
2. **DAG**：对句子列出所有在词典中出现过的连续片段；
3. **动态规划取最优路径**：`route[i] = max(log(FREQ[词] / total) + route[j])`，即最大对数概率路径；
4. **未登录词**：DAG 中切不出词的连续单字片段交给 HMM（Viterbi，BEMS 标注）做新词发现；
5. **词性**：`jieba.posseg` 是「重新分词 + HMM 词性标注」的另一条路径，与 `jieba.cut` 不共享切分结果；
6. **词频可改**：自定义词表每行 `词语 [词频] [词性]`，词频是**可写的权重数字**，不必是真实统计值（省略时 jieba 用 `suggest_freq` 自动补一个"刚好压线"的值，临界点上不稳定）。实测（jieba 0.42.1，`total`≈6.0e7）：多数领域词给 `词频=1` 也能整体保留；只有片段自身高频的词（信息安全、员工关系、安全生产、质量管理等）需要词频才压得过，临界值在 3~162 之间。本项目样例词表统一给 `100`（`userdict.txt` 另有 `200`）；
7. 词典**只增不减**：运行期 `load_userdict` 无法撤销，同一进程内切换到另一批词表时旧词仍然生效。

### tcwordseg（原生 C++ 引擎，`wordseg/tcwordseg`）

1. **同一算法族，实现封在编译产物里**：`.so` 符号可见 `pFreq`、`MaxWeight`、`IdxMaxItemWeight`、`NER_WEIGHT_FACTOR`、`CSegment::_compute_ne_weight`、`NERevise::_get_ner_freq` 等，说明同样做加权最大路径打分；
2. **词典与权重一起固化在二进制资源**：`cusdata.idx`(84.4MB)、`Core.Prop`(23.5MB)、`Ngram.Lex`(8.8MB)、`Core.Lex`(7.0MB) 等 17 个文件合计约 131MB，全部不可读不可改；绑定层 18 个函数**没有一个带词频参数**；
3. **用户只能干预"怎么切"，不能干预"词典里有什么"**：
   - `data/UserAmbRule`（文本、可编辑、自带格式说明）：**强制切分规则**，格式 `串,字节长度1,字节长度2,...`，如 `开吊车证,2,4,2` → `开 | 吊车 | 证`，用于分词器切错时兜底，**不能加词频、不能加词性**；
   - `TCChangeUserDict()`：切换用户词典文件，对应 `cusdata.idx` 这类二进制索引，需要官方工具生成（本项目未使用）；
4. **额外语言能力**（jieba 没有）：繁简转换、半全角/大小写归一、中英混排、邮箱/电话/网址识别为专门 token 类型、自定义词类；
5. **执行逻辑**：gb18030 编解码进出（`errors='ignore'`，生僻字/emoji 会被静默丢弃）；**有状态句柄**——先 `TCSegment` 把结果写进句柄，再 `TCGetResultCnt` + `TCGetWordAt` 逐个取，因此**每线程一个句柄、进程内顺序执行**；资源加载（进程级 `initconf`）与句柄创建（线程级）分两层；`segment_with_pos` 复用同一次切分结果；
6. **运行环境**：锁定 CPython 3.8 + 平台二进制（Windows `.pyd` / Linux `.so`），无源码不可重建。

### 执行逻辑差异一览

统一入口 [tokenizer.py](wordseg/tokenizer.py) 抹平了这些差异，对外 5 个接口（`init` / `uninit` / `is_available` / `segment` / `segment_with_pos`）签名一致：

| 维度 | jieba | tcwordseg |
| -- | -- | -- |
| 词频/权重 | 文本词表里写，随时可改 | 固化在二进制词典内 |
| 加词方式 | 增删词条 + 词频 + 词性 | 只能写强制切分规则（无词频、无词性） |
| 生效时机 | 运行期增量加载（只增不减） | 启动期一次性加载，改规则需重新 `init` |
| 实例模型 | 全局单例、无状态、线程安全 | 有状态句柄、每线程一个、进程内顺序执行 |
| 生命周期 | `initialize()` 一次成型 | `initconf`(进程级) + 建句柄(线程级) 两层 |
| 文本编码 | 全程 Unicode | gb18030 编解码（`errors='ignore'` 丢字） |
| `segment_with_pos` | 重新切分一次 | 复用同一次切分 |
| `is_available()` | 需探测（未安装 jieba 时为假） | 需探测（延迟导入 + 3.8 ABI） |
| `uninit()` | 空操作（词典常驻） | 真卸载 `TCUnInitSeg` |

### 对使用的影响

- **下游代码零改动**：`dedup/`、`pipelines/`、`tests/` 只依赖 `tokenizer.segment`，切换到哪个引擎不影响调用代码；
- **结果不等价**：分词结果不同 → MinHash 用的词集合不同 → 去重结果可能变化（阈值 0.95 附近最敏感）。切换引擎请重跑并复核产物，不要当作等价替换；
- **选型**：需要低成本增删领域词 → jieba；需要繁简转换/中英混排等原生能力且能提供 Python 3.8 → tcwordseg（要保住某个词整体切出，只能在 `UserAmbRule` 里写切分规则）。

