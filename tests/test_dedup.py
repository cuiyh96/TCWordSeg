#coding=utf-8
"""去重核心（仓库根 dedup 包）单元测试。

这些用例不依赖原生分词引擎，用极简的按空格切分替代，可在任意 Python 版本下运行：
    python -m pytest tests/test_dedup.py
"""
import os
import sys

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.append(_project_root)

import dedup


def _space_split(text):
    """极简分词器：按空格切分（模块级函数，多进程下可被 pickle）。"""
    return text.split()


# ==================== 文本抽取 ====================

def test_extract_sft_text_s1():
    record = {
        'history': [['第一轮问题', '第一轮回答']],
        'instruction': '指令',
        'input': '',
        'output': '答案',
    }
    assert dedup.extract_sft_text(record, 's1') == '第一轮问题\n第一轮回答\n指令\n答案'


def test_extract_sft_text_s2():
    record = {'conversations': [{'from': 'human', 'value': '你好'},
                                {'from': 'assistant', 'value': '你好，有什么可以帮你'}]}
    assert dedup.extract_sft_text(record, 's2') == '你好\n你好，有什么可以帮你'


# ==================== 数据划分 ====================

def test_divide_chunks_is_balanced_and_complete():
    chunks = dedup._divide_chunks(10, 3)
    assert chunks == [(0, 4), (4, 7), (7, 10)]
    assert sum(end - start for start, end in chunks) == 10


# ==================== MinHash 签名 ====================

def test_calc_hashvalues_shape():
    hashvalues = dedup.calc_hashvalues(['a b c', 'd e f'], num_perm=32,
                                       text_to_words=_space_split)
    assert hashvalues.shape == (2, 32)
    assert hashvalues.dtype == np.uint64


def test_calc_hashvalues_same_text_same_signature():
    hashvalues = dedup.calc_hashvalues(['a b c', 'a b c'], num_perm=32,
                                       text_to_words=_space_split)
    assert np.array_equal(hashvalues[0], hashvalues[1])


def test_calc_hashvalues_multiprocess_matches_serial():
    texts = ['a b c', 'd e f', 'a b c', 'g h i', 'a b c', 'j k l']
    serial = dedup.calc_hashvalues(texts, num_perm=64, text_to_words=_space_split)
    parallel = dedup.calc_hashvalues(texts, num_perm=64, text_to_words=_space_split,
                                     num_processes=3)
    assert np.array_equal(serial, parallel)


# ==================== LSH 去重 ====================

def test_dedup_by_lsh_keeps_smallest_index():
    hashvalues = dedup.calc_hashvalues(['a b c', 'a b c', 'a b c'], num_perm=128,
                                       text_to_words=_space_split)
    result = dedup.dedup_by_lsh(hashvalues, threshold=0.95)
    assert result.keep_indexes == [0]
    assert result.dup_map == {1: 0, 2: 0}


def test_dedup_by_lsh_keeps_distinct_records():
    hashvalues = dedup.calc_hashvalues(['alpha beta gamma', 'delta epsilon zeta', 'eta theta iota'],
                                       num_perm=128, text_to_words=_space_split)
    result = dedup.dedup_by_lsh(hashvalues, threshold=0.95)
    assert result.keep_indexes == [0, 1, 2]
    assert result.dup_map == {}


def test_dedup_by_lsh_dup_map_points_to_kept_index():
    hashvalues = dedup.calc_hashvalues(['a b c', 'a b c', 'x y z', 'a b c'], num_perm=128,
                                       text_to_words=_space_split)
    result = dedup.dedup_by_lsh(hashvalues, threshold=0.95)
    for drop_index, keep_index in result.dup_map.items():
        assert drop_index not in result.keep_indexes
        assert keep_index in result.keep_indexes


def test_dedup_by_lsh_merges_two_clusters(monkeypatch):
    """新记录同时命中两个既有相似簇时，应整体归并到最小索引且不漏去重。

    构造 0~1、2~3 两个独立簇，再由记录 4 同时相似于 1 与 3 把两簇连起来，
    此时 0~4 属同一传递相似簇，只应保留索引 0。
    """

    class _FakeLSH:
        """按预设返回相似项，稳定复现跨簇合并场景。"""

        def __init__(self, threshold, num_perm):
            self.neighbors = {1: [0], 3: [2], 4: [1, 3]}

        def query(self, minhash):
            return list(self.neighbors.get(int(minhash.hashvalues[0]), []))

        def insert(self, index, minhash):
            pass

    monkeypatch.setattr(dedup, 'MinHashLSH', _FakeLSH)
    hashvalues = np.zeros((5, 4), dtype=np.uint64)
    for i in range(5):
        hashvalues[i, 0] = i

    result = dedup.dedup_by_lsh(hashvalues, threshold=0.95)
    assert result.keep_indexes == [0]
    assert result.dup_map == {1: 0, 2: 0, 3: 0, 4: 0}
