#coding=utf-8
"""基于 MinHash + LSH 的文本去重核心。

自包含实现，不依赖 CorpusForge 内部模块：数据划分、目录创建都在此完成。
分词通过 text_to_words 回调注入（默认 wordseg.tokenizer.segment），
因此也可配合其他分词器使用。

管线：原始数据 → 抽取文本 → 分词 → MinHash 签名 → LSH 近似去重
"""
import ast
import json
import multiprocessing
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from datasketch import MinHash, MinHashLSH

from wordseg.tokenizer import segment

DEFAULT_NUM_PERM = 128
DEFAULT_THRESHOLD = 0.95


@dataclass
class DedupResult:
    """去重结果。"""
    keep_indexes: List[int]      # 保留的数据索引（升序）
    dup_map: Dict[int, int]      # 重复索引 -> 最终保留索引
    lsh: MinHashLSH = None       # 拟合后的 LSH 索引，便于复查与落盘


# ==================== 文本抽取 ====================

def extract_sft_text(record, data_fmt='s1'):
    """从 s1/s2 格式的 SFT 记录中拼接问答文本。

    这里内置一份最小实现，使本仓库可脱离父项目独立运行；
    s1/s2 的拼接行为与 CorpusForge 的 SFTDataProcessor.get_sft_txt 保持一致，
    若父项目调整该逻辑，需同步此处。

    Args:
        record: 单条 SFT 数据
        data_fmt: 数据格式 s1（instruction/input/output/history）或 s2（conversations）
    """
    if data_fmt == 's1':
        parts = []
        history = record.get('history')
        if history is not None:
            if not isinstance(history, list):
                history = ast.literal_eval(history)
            for pair in history:
                parts.extend(str(item).strip() for item in pair)
        for key in ['instruction', 'input', 'output']:
            if record.get(key) is not None:
                parts.append(str(record[key]).strip())
        return '\n'.join(item for item in parts if item != '')

    # s2：多轮对话
    parts = [str(it.get('value', '')).strip() for it in record.get('conversations', [])]
    return '\n'.join(item for item in parts if item != '')


# ==================== MinHash 签名 ====================

def _hashvalues_serial(texts, num_perm, text_to_words):
    """单进程顺序计算 MinHash 签名。"""
    hashvalues = np.empty((len(texts), num_perm), dtype=np.uint64)
    for i, text in enumerate(texts):
        # 与引擎一致：按词计算，词需编码为 bytes
        words = [word.encode('utf-8') for word in text_to_words(text)]
        minhash = MinHash(num_perm=num_perm)
        minhash.update_batch(words)
        hashvalues[i, :] = minhash.hashvalues
    return hashvalues


def _hashvalues_worker(texts, num_perm, text_to_words, out_path):
    """子进程工作函数：计算一段文本的签名并落地临时文件。"""
    np.save(out_path, _hashvalues_serial(texts, num_perm, text_to_words))


def _divide_chunks(total_num, n_workers):
    """把 [0, total_num) 尽量均匀地划分给 n_workers 个进程。"""
    base_size, remainder = divmod(total_num, n_workers)
    chunks = []
    start = 0
    for i in range(n_workers):
        end = start + base_size + (1 if i < remainder else 0)
        chunks.append((start, end))
        start = end
    return chunks


def calc_hashvalues(texts, num_perm=DEFAULT_NUM_PERM, text_to_words=segment,
                    num_processes=1, tmp_dir=None):
    """对一批文本计算 MinHash 签名。

    Args:
        texts: 文本列表
        num_perm: 每个 MinHash 的哈希函数个数
        text_to_words: 分词回调 str -> list[str]，多进程时需可被 pickle（用模块级函数）
        num_processes: 进程数；实际进程数不会超过文本条数，为 1 时顺序计算
        tmp_dir: 多进程中间结果目录，默认使用系统临时目录并在结束后清理

    Returns:
        np.ndarray，形状 (len(texts), num_perm)
    """
    total_num = len(texts)
    if total_num == 0:
        return np.empty((0, num_perm), dtype=np.uint64)

    num_processes = max(1, min(int(num_processes), total_num))
    if num_processes == 1:
        return _hashvalues_serial(texts, num_perm, text_to_words)

    chunks = _divide_chunks(total_num, num_processes)
    print(f'分词去重：实际使用 {num_processes} 个进程，各进程数据索引范围依次为 {chunks}.')

    own_tmp = tmp_dir is None
    work_dir = tempfile.mkdtemp(prefix='tcwordseg_hashvalues_') if own_tmp else tmp_dir
    os.makedirs(work_dir, exist_ok=True)
    try:
        processes, out_paths = [], []
        for i, (start_line, end_line) in enumerate(chunks):
            out_path = os.path.join(work_dir, f'hashvalues_{i}.npy')
            out_paths.append(out_path)
            process = multiprocessing.Process(
                target=_hashvalues_worker,
                args=(texts[start_line:end_line], num_perm, text_to_words, out_path))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
        failed = [process.exitcode for process in processes if process.exitcode]
        if failed:
            raise RuntimeError('计算哈希值的子进程异常退出（exitcode=%s）' % failed)

        return np.concatenate([np.load(path) for path in out_paths], axis=0)
    finally:
        if own_tmp:
            shutil.rmtree(work_dir, ignore_errors=True)


# ==================== LSH 近似去重 ====================

def _resolve_keep(dup_map, index):
    """沿 dup_map 链回溯到最终保留的索引。"""
    while index in dup_map:
        index = dup_map[index]
    return index


def _rebuild_minhash(hashvalues):
    """由已算好的签名重建 MinHash（供 LSH 建索引/查询用）。

    签名数组本身不带方案信息，datasketch 2.0 起重建时必须显式声明 scheme；
    本模块的签名均由默认方案 affine32 生成，datasketch 1.x 无该参数。
    """
    try:
        return MinHash(hashvalues=hashvalues, scheme='affine32')
    except TypeError:   # datasketch < 2.0 的 MinHash 不接受 scheme 参数
        return MinHash(hashvalues=hashvalues)


def dedup_by_lsh(hashvalues, threshold=DEFAULT_THRESHOLD):
    """按 Jaccard 相似度阈值做 LSH 近似去重。

    同一相似簇内保留索引最小的数据，其余索引记入 dup_map 并指向该保留索引。

    Args:
        hashvalues: MinHash 签名数组，形状 (N, num_perm)
        threshold: 相似度阈值

    Returns:
        DedupResult
    """
    hashvalues = np.asarray(hashvalues)
    if hashvalues.size == 0:
        return DedupResult(keep_indexes=[], dup_map={}, lsh=None)

    total_num, num_perm = hashvalues.shape
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    dup_map = {}
    for cur_i in range(total_num):
        cur_minhash = _rebuild_minhash(hashvalues[cur_i, :])
        # 先查询后插入：命中的必然都是索引更小的已插入数据
        similar = sorted(lsh.query(cur_minhash))
        if similar:
            # 命中的相似数据可能来自多个既有簇：取各簇根索引的最小者为保留索引，
            # 并把其余簇的根一并并入，避免合并后仍留下本应被去重的索引
            roots = sorted({_resolve_keep(dup_map, index) for index in similar})
            keep_index = roots[0]
            for root in roots[1:]:
                if root > keep_index:
                    dup_map[root] = keep_index
            # cur_i 与已插入的相似数据同属一簇：保留索引最小者，其余全部归并过去
            for drop_i in similar + [cur_i]:
                if drop_i > keep_index:
                    dup_map[drop_i] = keep_index
        lsh.insert(cur_i, cur_minhash)

    keep_indexes = [i for i in range(total_num) if i not in dup_map]
    return DedupResult(keep_indexes=keep_indexes, dup_map=dup_map, lsh=lsh)


# ==================== 数据读取 ====================

def load_jsonl(file_path):
    """读取 jsonl 文件为 dict 列表。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]
