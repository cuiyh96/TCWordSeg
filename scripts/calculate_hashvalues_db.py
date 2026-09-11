import time
import json
import multiprocessing
import argparse
import os
import sys
import shutil
import numpy as np
import jsonlines
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.append(project_root)

from datasketch import MinHash
from tcwordseg.TCWordSeg3 import TCWordSeg3

# 自定义函数
from divide import divide_worker
from corpus.sft import SFTDataProcessor
from base.file_io import FileIO

# --------------------------------------------------
tmp_dirname = os.path.join(project_root, 'tmp')  # 本地临时文件夹，保存各进程中间结果


def load_data(file_path):
    """从本地 jsonl 文件读取数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def perprocess_calculate_hashvalue(data, data_fmt, start_line, end_line, process_i, num_perm):
    """
    每个进程的工作函数：读取数据、分词、生成MinHash值，并将结果保存到第 i 个临时文件
    """
    seginst = TCWordSeg3()  # 创建一个分词实例
    hHandle = seginst.seghandle  # 获取分词句柄
    hashvalues_array = np.array([])
    for dic in data[start_line:end_line]:
        line = SFTDataProcessor.get_sft_txt(dic, data_fmt)
        if line == '':
            print('SFT数据为空，请检查对应的数据格式“s1/s2”是否正确！')
            break
        seginst.TCSegment(seghandle=hHandle, line=line)
        rescount = seginst.TCGetResultCnt(hHandle)
        seg_text = [seginst.TCGetWord(hHandle, i).encode('utf-8') for i in range(rescount)]
        minhash = MinHash(num_perm=num_perm)  # 创建 MinHash 对象
        minhash.update_batch(seg_text)  # 更新 MinHash 对象
        hashvalues_array = np.concatenate((hashvalues_array, minhash.hashvalues))
    pathf = os.path.join(tmp_dirname, f"hashvalues_{process_i}")
    np.save(pathf, hashvalues_array)
    del seginst  # 销毁实例


def calculate_hashvalues(input_path, output_path, data_fmt, num_processes, num_perm):
    """
    从本地文件读取数据，计算每条数据的 MinHash 哈希值，并落地到本地文件
    """
    data = load_data(input_path)
    total_num = len(data)
    num_processes, chunks = divide_worker(total_num, num_processes)
    print(f'最终实际工作的进程数为：{num_processes}个.')
    print(f'每个进程需要处理的数据索引范围依次为：{chunks}.\n')

    processes = []  # 进程池
    with tqdm(total=len(chunks), desc="Processing chunks", ncols=100) as pbar:
        for i, (start_line, end_line) in enumerate(chunks):
            process = multiprocessing.Process(
                target=perprocess_calculate_hashvalue,
                args=(data, data_fmt, start_line, end_line, i, num_perm))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()

    # 汇总所有进程计算完的哈希值
    hashvalues_array = np.array([])
    for process_i in tqdm(range(num_processes)):
        pathf = os.path.join(tmp_dirname, f"hashvalues_{process_i}.npy")
        hashvalues_i = np.load(pathf)
        hashvalues_array = np.concatenate((hashvalues_array, hashvalues_i))
    hashvalues_array = hashvalues_array.reshape(-1, num_perm)

    # 落地到本地文件：汇总数组 + id 与哈希值映射
    np.save(os.path.join(output_path, 'hashvalues.npy'), hashvalues_array)
    hashvalues_info_path = os.path.join(output_path, 'hashvalues_info.jsonl')
    with jsonlines.open(hashvalues_info_path, mode='w') as f:
        for i in range(total_num):
            f.write({'id': data[i].get('id'), 'hashvalues': hashvalues_array[i, :].tolist()})
    print(f'哈希值已保存到: {output_path}')


def get_parser():
    parser = argparse.ArgumentParser(description="计算每条数据的哈希值（本地文件落地）")
    parser.add_argument("--input_path", type=str, help="输入 jsonl 文件路径",
                        default='data_test/test_data.jsonl')
    parser.add_argument("--output_path", type=str, help="输出目录路径（默认与输入同级 *_result）", default=None)
    parser.add_argument("--num_perm", type=int, help="指定Minhash生成num_perm个哈希函数", default=128)
    parser.add_argument("--data_fmt", type=str, help="数据格式 s1/s2", default='s2')
    parser.add_argument("--num_processes", type=int, help="指定的进程数", default=32)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    FileIO.ensure_dir(tmp_dirname, clear=True)  # 清空临时文件夹，用于保存中间计算结果

    TCWordSeg3.initconf("data")  # 词典资源仅需加载一次
    num_processes = min(args.num_processes, multiprocessing.cpu_count())

    if args.output_path is None:
        output_path = os.path.splitext(args.input_path)[0] + '_result'
    else:
        output_path = args.output_path
    FileIO.ensure_dir(output_path, clear=True)

    start = time.time()
    calculate_hashvalues(args.input_path, output_path, args.data_fmt, num_processes, args.num_perm)
    end = time.time()
    print("Total time = %f minutes." % ((end - start) / 60))

    TCWordSeg3.uninitconf()  # 仅需加卸载一次
    shutil.rmtree(tmp_dirname)  # 删除临时文件夹
