from datasketch import MinHash, MinHashLSH
import argparse
import pickle
import os
import numpy as np
import jsonlines

def dedup_MinHashLSH(hashvalues_path):
    path_m = os.path.dirname(hashvalues_path)
    # 加载哈希值
    hashvalues_array = np.load(hashvalues_path)
    total_num, num_perm = hashvalues_array.shape
    lsh = MinHashLSH(threshold=args.threshold, num_perm=num_perm)

    # 添加签名、查询、去重
    result = []
    dedup_index_info = []
    dup_index_info = []
    for cur_i in range(total_num):
        # 直接根据计算的哈希值，生成MinHash签名
        cur_hashvalue_i = hashvalues_array[cur_i,:]
        cur_minhash = MinHash(hashvalues=cur_hashvalue_i)
        # 插入MinHash签名到LSH索引
        lsh.insert(cur_i, cur_minhash)
        # 查询重复（相似度阈值以上的）数据
        similar_txts = lsh.query(cur_minhash)
        sorted_similar_txts = sorted(similar_txts)
        keep_index = sorted_similar_txts[0]
        drop_indexs = sorted_similar_txts[1:]
        dedup_index_info.append(keep_index)
        if drop_indexs!=[]:
            for drop_item in drop_indexs:
                tmp = {drop_item: keep_index}
                dup_index_info.append(tmp)
    dedup_index_info = list(set(dedup_index_info))

    # 保存 LSH 索引到文件
    lsh_path =  os.path.join(path_m, 'lsh.pkl')
    with open(lsh_path, 'wb') as f:
        pickle.dump(lsh, f)

    print('保留数据的index：', dedup_index_info)
    dedup_path = os.path.join(path_m, 'dedupindex.txt')
    with open(dedup_path, 'w') as f:
        for item in dedup_index_info:
            f.write(str(item)+'\n')

    print('重复数据的index与保留数据index的映射关系：', dup_index_info)
    dup_path = os.path.join(path_m, 'dupindex_map.jsonl')
    with jsonlines.open(dup_path, mode='w') as f:
        for dict_js in dup_index_info:
            f.write(dict_js)

def get_parser():
    parser = argparse.ArgumentParser(description="数据去重")
    parser.add_argument("--data_path", type=str, help="数据文件路径",
                        default='/cuiyah/Projects/TCWordSeg/data_test/test_data.jsonl')
    parser.add_argument("--hashvalues_path", type=str, help="数据对应的哈希值文件路径",
                        default=None)
    parser.add_argument("--threshold", type=float, help="去重算法阈值", default=0.95)
    return parser



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    # 获取哈希值的存储文件路径：默认是数据文件同级***_result文件夹下的'hashvalues.npy'
    if args.hashvalues_path is None:
        path_m = os.path.splitext(args.data_path)[0] + '_result'
        hashvalues_path = os.path.join(path_m, 'hashvalues.npy')
    else:
        hashvalues_path = args.hashvalues_path
    # 去重
    dedup_MinHashLSH(hashvalues_path)