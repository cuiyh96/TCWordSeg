import time
import json
import multiprocessing
import argparse
import os
import sys
import numpy as np
import shutil
from tqdm import tqdm
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.append(project_root)

from datasketch import MinHash
from tcwordseg.TCWordSeg3 import TCWordSeg3

sys.path.append(r"/LLM_data/common_util/")
# 自定义函数
from public import setDir
from divide_data import divide_worker
from get_ids import get_sft_txt
from mongodb_connect_5 import write_data, read_data, createIndex

# --------------------------------------------------
ecd_set = 'utf-8'

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def perprocess_calculate_hashvalue(data, data_fmt, start_line, end_line, process_i):
    """
    定义每个进程的工作函数，用于读取文件、分词和生成MinHash值，并将结果保存到第i个文件中
    :param data: 全量数据
    :param start_line: 当前进程处理的数据，对应索引开始位置
    :param end_line:   当前进程处理的数据，对应索引结束位置
    :param process_i: 第i个进程
    :return:
    """
    seginst = TCWordSeg3() # 创建一个分词实例
    hHandle = seginst.seghandle  # 获取分词句柄
    hashvalues_array=np.array([])
    for dic in data[start_line:end_line]:
        line = get_sft_txt(dic, data_fmt)
        if line == '':
            print('SFT数据为空，请检查对应的数据格式“s1/s2”是否正确！')
            break
        # 执行分词：用指定的分词句柄（seghandle）对输入文本（line）进行分词。
        seginst.TCSegment(seghandle=hHandle, line=line)
        # 获取分词结果的数量，即分词后的单词或短语的数量
        rescount = seginst.TCGetResultCnt(hHandle)
        # 按指定索引位置，获取分词结果
        seg_text = [seginst.TCGetWord(hHandle, i).encode('utf-8') for i in range(rescount)]
        # 计算哈希值
        minhash = MinHash(num_perm = args.num_perm) # 创建 MinHash 对象
        minhash.update_batch(seg_text) # 更新 MinHash 对象
        hashvalues_array = np.concatenate((hashvalues_array, minhash.hashvalues))
    # 保存哈希值
    pathf = os.path.join(tmp_dirname, f"hashvalues_{process_i}")
    np.save(pathf, hashvalues_array)
    del seginst # 销毁实例

# 主函数，用于创建工作进程并汇总结果  
def calculate_hashvalues(read_data_name, num_processes):
    split_name = read_data_name.split('_')
    data_fmt = split_name[-1]
    if data_fmt=='s1':
        select_ = {'_id':0, 'id':1, 'instruction':1, 'input':1, 'output':1, 'history':1}
    elif data_fmt=='s2':
        select_ = {'_id': 0, 'id': 1, 'conversations': 1}
    else:
        raise f'输入的表名 {read_data_name} 不是以“s1/s2”结尾的，命名不规范，请确认！'
    data = read_data(split_name[0], read_data_name, select=select_)
    total_num = len(data)
    # 数据划分，划分后 num_processes=len(chunks)
    num_processes, chunks = divide_worker(total_num, num_processes)
    print(f'最终实际工作的进程数为：{num_processes}个.')
    print(f'每个进程需要处理的数据索引范围依次为：{chunks}.\n')

    processes = [] # 进程池
    # 使用 tqdm 包装 enumerate，以显示进度条
    with tqdm(total=len(chunks), desc="Processing chunks", ncols=100) as pbar:
        for i, (start_line, end_line) in enumerate(chunks):
            process = multiprocessing.Process(target=perprocess_calculate_hashvalue,
                                              args=(data, data_fmt, start_line, end_line, i))
            processes.append(process)
            process.start()

        # 等待所有进程完成
        for process in processes:
            process.join()

    # 汇总所有进程计算完的哈希值
    hashvalues_array = np.array([])
    for process_i in tqdm(range(num_processes)):
        # 读取每个进程计算完的哈希值
        pathf = os.path.join(tmp_dirname, f"hashvalues_{process_i}.npy")
        hashvalues_i = np.load(pathf)
        hashvalues_array = np.concatenate((hashvalues_array, hashvalues_i))
    # 汇总的hashvalues
    hashvalues_array = hashvalues_array.reshape(-1, args.num_perm)
    # 保存汇总的hashvalues
    hashvalues_info = []
    for i in range(total_num):
        tmp={}
        tmp['id'] = data[i]['id']
        tmp['hashvalues'] = hashvalues_array[i,:].tolist() # 将 NumPy 数组转换为列表
        hashvalues_info.append(tmp)

    write_data_name = 'hv_'+'_'.join(split_name[:-1]+str(args.num_perm))
    createIndex('hv', write_data_name)
    write_data('hv', write_data_name, hashvalues_info, rewrite=True)

def get_parser():
    parser = argparse.ArgumentParser(description="计算每条数据的哈希值")
    parser.add_argument("--read_data_name", type=str, help="读取数据的表名",
                        default='std_sft_collect_baichuan_49997_cn_s2')
    parser.add_argument("--num_processes", type=str, help="指定的进程数", default=32)
    parser.add_argument("--num_perm", type=float, help="指定Minhash生成num_perm个哈希函数", default=128)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    tmp_dirname = 'cuiayh/tmp' #临时文件夹
    setDir(tmp_dirname, type=1)  # 清空临时文件夹，用于保存中间计算结果

    # 准备工作
    TCWordSeg3.initconf("data")  #词典资源仅需加载一次
    num_processes = min(args.num_processes, multiprocessing.cpu_count())

    # 开始计算哈希值
    start = time.time()
    calculate_hashvalues(args.read_data_name, num_processes)
    end = time.time()
    print("Total time = %f minutes." % ((end - start) / 60))

    TCWordSeg3.uninitconf()  # 仅需加卸载一次
