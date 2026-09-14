#coding=utf-8
"""Step 2：LSH 近似去重，输出保留索引与重复映射。

用法：
    python pipelines/dedup_by_lsh.py --data_path data.jsonl --threshold 0.95
    python pipelines/dedup_by_lsh.py --hashvalues_path data_result/hashvalues.npy
"""
import argparse
import os
import pickle
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.append(_project_root)

import jsonlines
import numpy as np

import dedup


def get_parser():
    parser = argparse.ArgumentParser(description="数据去重")
    parser.add_argument("--data_path", type=str, help="数据文件路径（用于推导 *_result/hashvalues.npy）",
                        default=None)
    parser.add_argument("--hashvalues_path", type=str, help="数据对应的哈希值文件路径",
                        default=None)
    parser.add_argument("--threshold", type=float, help="去重算法阈值", default=0.95)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    # 获取哈希值的存储文件路径：默认是数据文件同级 *_result 文件夹下的 hashvalues.npy
    if args.hashvalues_path is None:
        if args.data_path is None:
            raise SystemExit('请至少指定 --hashvalues_path 或 --data_path 之一。')
        hashvalues_path = os.path.join(os.path.splitext(args.data_path)[0] + '_result',
                                       'hashvalues.npy')
    else:
        hashvalues_path = args.hashvalues_path

    out_dir = os.path.dirname(os.path.abspath(hashvalues_path))
    hashvalues = np.load(hashvalues_path)
    result = dedup.dedup_by_lsh(hashvalues, threshold=args.threshold)

    # 保存 LSH 索引
    with open(os.path.join(out_dir, 'lsh.pkl'), 'wb') as f:
        pickle.dump(result.lsh, f)

    print('保留数据的index：', result.keep_indexes)
    with open(os.path.join(out_dir, 'dedupindex.txt'), 'w') as f:
        for index in result.keep_indexes:
            f.write(str(index) + '\n')

    print('重复数据的index与保留数据index的映射关系：', result.dup_map)
    with jsonlines.open(os.path.join(out_dir, 'dupindex_map.jsonl'), mode='w') as f:
        for drop_index, keep_index in sorted(result.dup_map.items()):
            f.write({drop_index: keep_index})
