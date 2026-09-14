#coding=utf-8
"""Step 3：批量中文分词（多进程）。

分词走统一入口 wordseg.tokenizer，引擎由环境变量 TCWORDSEG_ENGINE 选择
（默认 tcwordseg，设为 jieba 即改用 jieba），无需改代码。

用法（路径按需替换为相对或绝对路径）：
    python pipelines/segment_corpus.py \
        --dir_path data/raw \
        --output_path output/seg

    python pipelines/segment_corpus.py \
        --filelist filelist1.txt \
        --dir_path data/raw \
        --output_path output/seg

filelist 每行：文件名\\t标题列号\\t内容列号（列号 -1 表示整行不分列）；
不指定 --filelist 时，目录下所有文件都按整行不分列处理。
输出 {输出名}.jsonl，每行含 title / content / content_seg（分词结果，空格分隔）。
"""
import argparse
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
from itertools import islice

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.append(_project_root)

from wordseg import tokenizer

ecd_set = 'gb18030'
NUM_PROCESSES = 10


def process_file_chunk(file_path, start_line, end_line, pid, tmp_dir, titleloc, contloc):
    """子进程工作函数：切分一段行，结果写入 tmp_dir/{pid}_tmpseg.txt。

    引擎资源要求进程内单独初始化，故每个子进程各自调用 tokenizer.init()。
    """
    tokenizer.init()
    out_path = os.path.join(tmp_dir, '%d_tmpseg.txt' % pid)
    with open(file_path, 'r', encoding=ecd_set, errors='ignore') as fpr, \
            open(out_path, 'w', encoding='utf-8') as fpw:
        for line in islice(fpr, start_line, end_line):
            line = line.strip()
            if not line:
                continue

            if titleloc == -1:
                title = content = line
            else:
                items = line.split('\t')
                if max(titleloc, contloc) >= len(items):
                    continue   # 列数不足，跳过该行
                title, content = items[titleloc], items[contloc]

            title = title.replace('\\n', '\n')
            content = content.replace('\\n', '\n')
            record = {
                'title': title,
                'content': content,
                'content_seg': ' '.join(tokenizer.segment(content)),
            }
            fpw.write(json.dumps(record, ensure_ascii=False) + '\n')
    tokenizer.uninit()


def multiSeg(file_path, output_file, titleloc, contloc):
    """把单个文件的指定列切成 NUM_PROCESSES 段并行分词，再按原顺序合并输出。"""
    num_lines = 0
    with open(file_path, 'r', encoding=ecd_set, errors='ignore') as fpr:
        for _ in fpr:
            num_lines += 1

    num_processes = max(1, min(NUM_PROCESSES, num_lines))
    lines_per_process = num_lines // num_processes
    print('%s: 共 %d 行，使用 %d 个进程' % (file_path, num_lines, num_processes))

    tmp_dir = tempfile.mkdtemp(prefix='tcwordseg_seg_')
    try:
        processes = []
        for i in range(num_processes):
            start_line = i * lines_per_process
            end_line = (i + 1) * lines_per_process if i != num_processes - 1 else num_lines
            process = multiprocessing.Process(
                target=process_file_chunk,
                args=(file_path, start_line, end_line, i, tmp_dir, titleloc, contloc))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        failed = [process.exitcode for process in processes if process.exitcode]
        if failed:
            raise RuntimeError('子进程异常退出（exitcode=%s），已中断输出' % failed)

        out_path = os.path.splitext(output_file)[0] + '.jsonl'
        total = 0
        with open(out_path, 'w', encoding='utf-8') as fpw:
            for i in range(num_processes):
                with open(os.path.join(tmp_dir, '%d_tmpseg.txt' % i),
                          'r', encoding='utf-8') as fpr:
                    for line in fpr:
                        fpw.write(line.strip() + '\n')
                        total += 1
        print('%s: 输出 %d 行' % (out_path, total))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


mapfile = {}


def iterate_through_directory(dir_path, output_path):
    """遍历输入（单个文件或目录），逐个切分到 output_path 下同名的 jsonl。

    指定了 filelist 时只处理清单内的文件；未指定时目录下所有文件都按整行不分列处理。
    """
    if os.path.isfile(dir_path):
        file_paths = [dir_path]
    else:
        file_paths = [os.path.join(dir_path, name) for name in sorted(os.listdir(dir_path))
                      if os.path.isfile(os.path.join(dir_path, name))]

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        loc = mapfile.get(filename)
        if loc is None:
            if mapfile:      # 有清单但该文件不在清单内：跳过
                continue
            loc = (-1, -1)   # 无清单：整行不分列
        multiSeg(file_path, os.path.join(output_path, filename), loc[0], loc[1])


def get_parser():
    parser = argparse.ArgumentParser(description="批量中文分词（多进程）")
    parser.add_argument("--filelist", type=str, help="文件清单路径（每行：文件名\\t标题列号\\t内容列号），"
                                                    "不指定则处理目录下所有文件",
                        default=None)
    parser.add_argument("--dir_path", type=str, help="输入目录（或文件）路径", required=True)
    parser.add_argument("--output_path", type=str, help="输出目录路径", required=True)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    if args.filelist:
        with open(args.filelist, "r", encoding=ecd_set) as filelist:
            for line in filelist:
                name = line.strip()
                if not name or name.startswith('#'):
                    continue
                namelist = name.split('\t')
                if len(namelist) < 3:
                    raise SystemExit('filelist 行格式错误（应为 文件名\\t标题列号\\t内容列号）：%s' % name)
                mapfile[namelist[0]] = (int(namelist[1]), int(namelist[2]))

    os.makedirs(args.output_path, exist_ok=True)

    begin = time.time()
    iterate_through_directory(args.dir_path, args.output_path)
    end = time.time()
    print("total time = %f m" % ((end - begin) / 60))
