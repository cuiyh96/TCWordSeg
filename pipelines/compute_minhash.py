#coding=utf-8
"""Step 1（文件输入）：计算每条数据的 MinHash 签名，落地 hashvalues.npy。

加 --save_id_map 时额外落地 id 与哈希值的映射文件 hashvalues_info.jsonl。

用法：
    python pipelines/compute_minhash.py --input_path data.jsonl \
        --num_perm 128 --data_fmt s2 --num_processes 5
    python pipelines/compute_minhash.py --input_path data.jsonl --save_id_map
"""
import argparse
import multiprocessing
import os
import sys
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.append(_project_root)

import jsonlines
import numpy as np

import dedup
from wordseg.tokenizer import segment


def get_parser():
    parser = argparse.ArgumentParser(description="计算每条数据的哈希值")
    parser.add_argument("--input_path", type=str, required=True, help="输入 jsonl 文件路径")
    parser.add_argument("--output_path", type=str, help="输出目录路径（默认与输入同级 *_result）",
                        default=None)
    parser.add_argument("--num_perm", type=int, help="指定Minhash生成num_perm个哈希函数",
                        default=128)
    parser.add_argument("--data_fmt", type=str, help="数据格式 s1/s2", default='s2')
    parser.add_argument("--num_processes", type=int, help="指定的进程数", default=5)
    parser.add_argument("--save_id_map", action='store_true',
                        help="额外落地 id 与哈希值的映射文件 hashvalues_info.jsonl")
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    output_path = args.output_path or (os.path.splitext(args.input_path)[0] + '_result')
    os.makedirs(output_path, exist_ok=True)

    data = dedup.load_jsonl(args.input_path)
    texts = [dedup.extract_sft_text(item, args.data_fmt) for item in data]

    start = time.time()
    hashvalues = dedup.calc_hashvalues(
        texts,
        num_perm=args.num_perm,
        text_to_words=segment,
        num_processes=min(args.num_processes, multiprocessing.cpu_count()),
    )
    end = time.time()
    print("Total time = %f minutes." % ((end - start) / 60))

    np.save(os.path.join(output_path, 'hashvalues.npy'), hashvalues)

    if args.save_id_map:
        with jsonlines.open(os.path.join(output_path, 'hashvalues_info.jsonl'), mode='w') as f:
            for i, item in enumerate(data):
                f.write({'id': item.get('id'), 'hashvalues': hashvalues[i, :].tolist()})
    print(f'哈希值已保存到: {output_path}')
