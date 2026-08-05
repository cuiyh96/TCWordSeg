# TCWordSeg

中文分词与文本去重工具包，服务于“大模型训练数据”的预处理阶段。

## 项目结构

```
TCWordSeg/
├── tcwordseg/                    # 核心分词库（Python 包）
│   ├── __init__.py
│   ├── TCWordSeg.py              # SWIG 自动生成的 Python 绑定层
│   ├── TCWordSeg3.py             # Python3 面向对象封装
│   ├── _TCWordSeg.pyd            # C/C++ 分词引擎（Windows）
│   ├── _TCWordSeg.so             # C/C++ 分词引擎（Linux）
│   └── data/                     # 词典资源文件
│       ├── Core.Lex              # 核心词典
│       ├── Core.Prop             # 核心属性
│       ├── Ngram.Lex             # N-gram 语言模型
│       ├── Ngram_PN.Lex          # N-gram PN 模型
│       ├── POSTT.Lex             # 词性标注模型
│       ├── RoleTT.Lex            # 角色标注模型
│       ├── LNRoleTT.Lex          # LN 角色标注模型
│       ├── Sec.Lex / Sec.Prop    # 粗粒度分词词典
│       ├── UserAmbRule           # 用户消歧规则
│       ├── id_custname.dict      # 自定义人名词典
│       ├── cusdata.idx / .idxnull # 自定义词典索引
│       ├── ner_info              # 命名实体识别信息
│       └── t2s / t2s.word / t2s.disable  # 繁简转换
├── scripts/                      # 应用脚本
│   ├── WordSegmultiprocess.py    # 多进程分词工具
│   ├── calculate_hashvalues.py   # MinHash 签名计算（文件输入）
│   ├── calculate_hashvalues_db.py # MinHash 签名计算（MongoDB 输入）
│   └── dropduplicates_LSH.py    # LSH 近似去重
├── tests/                        # 测试
│   └── data_test/                # 测试 notebook
└── output/                       # 去重结果输出
```

## 核心功能

### 1. 中文分词引擎

底层为自研 C/C++ 分词引擎，通过 SWIG 生成 Python 绑定，主要特性：

- **粗/细粒度分词** — 双句柄模式，支持不同粒度的切分需求
- **词性标注** — 支持名词(n)、动词(v)、形容词(a)等 40+ 词性标签
- **自定义词类** — 支持用户自定义词典和分类识别
- **命名实体识别** — 人名、地名、机构名等
- **繁简转换** — 繁体中文到简体中文自动转换
- **数字/时间/邮箱/电话识别** — 特殊模式识别

#### 使用示例

```python
from tcwordseg.TCWordSeg3 import TCWordSeg3

# 初始化（每个进程加载一次）
TCWordSeg3.initconf("data")

# 创建分词实例
seg = TCWordSeg3()

# 分词输出
result = seg.TCSegOutput(seg.seghandle, "用友网络科技股份有限公司")
# 用友 网络 科技 股份 有限公司

# 分词 + 词性标注
result_pos = seg.TCSegOutputPos(seg.seghandle, "用友网络科技股份有限公司")
# 用友/nr 网络/n 科技/n 股份/n 有限公司/n

# 释放资源
del seg
TCWordSeg3.uninitconf()
```

### 2. 文本去重管线

基于 MinHash + LSH 的大规模文本去重，用于大模型训练数据清洗：

```
原始数据 → 分词 → MinHash 签名(128个哈希函数) → LSH 索引(Jaccard相似度≥0.95) → 去重结果
```

#### 去重流程

**Step 1：计算 MinHash 签名**

```bash
# 从文件读取数据
python scripts/calculate_hashvalues.py \
    --input_path data.jsonl \
    --num_perm 128 \
    --data_fmt s2 \
    --num_processes 10

# 从 MongoDB 读取数据
python scripts/calculate_hashvalues_db.py \
    --read_data_name std_sft_collect_baichuan_49997_cn_s2 \
    --num_perm 128 \
    --num_processes 32
```

**Step 2：LSH 去重**

```bash
python scripts/dropduplicates_LSH.py \
    --data_path data.jsonl \
    --threshold 0.95
```

**Step 3：多进程批量分词**

```bash
python scripts/WordSegmultiprocess.py
```

## 数据格式

去重管线支持两种 SFT 数据格式：

| 格式 | 字段                                  | 说明     |
| -- | ----------------------------------- | ------ |
| s1 | instruction, input, output, history | 单轮对话格式 |
| s2 | conversations                       | 多轮对话格式 |

## 环境

- Python 3.7+

## Python 依赖

| 包名         | 用途                                         |
| ---------- | ------------------------------------------ |
| numpy      | 哈希值数组计算与存储                                 |
| datasketch | MinHash 签名与 LSH 近似去重                       |
| jsonlines  | JSONL 格式读写                                 |
| pymongo    | MongoDB 数据读写（calculate\_hashvalues\_db.py） |
| tqdm       | 进度条显示                                      |

