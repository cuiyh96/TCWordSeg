#coding=utf-8

import os, time, json, re, multiprocessing, threading 
from tcwordseg.TCWordSeg3 import TCWordSeg3

ecd_set = 'gb18030'
#ecd_set = 'utf-8'

wordseginst = None

# 定义进程工作函数，用于读取文件、分词和生成MinHash值，并将结果保存到单独的文件中  
def process_file_chunk(file_path, start_line, end_line, pid, titleloc, contloc):  
    TCWordSeg3.initconf("data") #词典资源仅需加载一次
    seginst = TCWordSeg3() #可创建多个实例

    fpw = open(str(pid)+'tmpseg.txt', 'w', encoding='utf-8')
    with open(file_path, 'r', encoding=ecd_set, errors='ignore') as file:  
        current_line = 0  
        for line in file:  
            if current_line < start_line:  
                current_line += 1  
                continue  # 跳过前面的行  
            if current_line >= end_line:  
                break  # 到达结束行，退出循环  

            
            line = line.strip()
            if titleloc != -1:
                items = line.split('\t')
                if(len(items) < 2):
                    continue
                line = items[titleloc] + '\t' + items[contloc]

            #seginst.TCSegment(seginst.seghandle, line)
            #rescount = seginst.TCGetResultCnt(seginst.seghandle)  #获得粗粒度
            #segout = ' '.join((seginst.TCGetWord(seginst.seghandle, i) for i in range(rescount)))
            segout = seginst.TCSegOut(seginst.seghandle, line)
            linejson = json.dumps({"title": line.split('\t')[0].replace('\\n','\n'), "content": line.split('\t')[1].replace('\\n','\n')}, ensure_ascii=False)

            fpw.write(linejson +'\n')
            current_line += 1  
    fpw.close()
    del seginst
    TCWordSeg3.uninitconf() #仅需加卸载一次

# 主函数，用于创建工作线程并汇总结果  
def multiSeg(file_path, output_file, titleloc, contloc):  
    # 无法直接获取总行数，所以需要先进行一次完整的文件遍历来确定行数 
    num_lines = 0
    with open(file_path, 'r', encoding=ecd_set, errors='ignore') as file:  
        num_lines = sum(1 for _ in file)  # 获取文件总行数          
    num_processes= 10 #multiprocessing.cpu_count() #10 #
    lines_per_processe = num_lines // num_processes  # 计算每个线程处理的行数  
    processes  = []  
    #results =  [ [] for _ in range(num_processes) ]
    #manager = multiprocessing.Manager()
    #results =  [ manager.list() for _ in range(num_processes) ]

    print(num_lines, lines_per_processe,  "num_processes = %d" %num_processes)
    for i in range(num_processes):  
        start_line = i * lines_per_processe  
        end_line = (i + 1) * lines_per_processe if i != num_processes - 1 else num_lines  
        #print(start_line, end_line)
        process = multiprocessing.Process(target=process_file_chunk, args=(file_path, start_line, end_line, i, titleloc, contloc))  
        process.start()  
        processes.append(process) 
    for process in processes:  
        process.join()  # 等待进程完成        

    fpw1= open(output_file.rsplit('.', 1)[0] + '.jsonl', 'w', encoding='utf-8')
    id = 0
    for i in range(num_processes):  
        fpr = open(str(i)+'tmpseg.txt', 'r', encoding='utf-8')
        for line in fpr:
            fpw1.write(line.strip()+'\n')
            id += 1
        fpr.close()
        print(output_file, id)
    fpw1.close()

mapfile = {}                       
def iterate_through_directory(dir_path, output_path):
    if os.path.isfile(dir_path):
        file = os.path.basename(dir_path)
        titleloc = -1
        contloc = -1
        if file in mapfile:
            titleloc, contloc = mapfile[file]
            print(titleloc, contloc)
            multiSeg(dir_path, os.path.join(output_path, file),titleloc, contloc )
    else:
        #for root, dirs, files in os.walk(dir_path):
        files = [file for file in os.listdir(dir_path)]
        for file in files:
            titleloc = -1
            contloc = -1
            if file in mapfile:
                titleloc, contloc = mapfile[file]
                multiSeg(os.path.join(dir_path, file), os.path.join(output_path, file), titleloc, contloc)
    #print(num)
    
if __name__ == '__main__':
    filelist = open("filelist1.txt", "r", encoding=ecd_set )
    for filenames in filelist:
        namelist = filenames.strip().split('\t')
        if filenames[0] == '#':
            continue
        filename = namelist[0]
        titleloc = int(namelist[1])
        contloc = int(namelist[2])
        mapfile[filename] = (titleloc, contloc)
    filelist.close()
    dir_path = 'c:/data/data/' # 替换为你的目录路径d
    output_path = 'c:/data/temp/' # 替换为你的输出文件路径
    #TCWordSeg3.initconf("data") #词典资源仅需加载一次
    #wordseginst = TCWordSeg3() #可创建多个实例
    begin = time.time()
    iterate_through_directory(dir_path, output_path)
    end = time.time()
    print("total time = %f m" %((end-begin)/60))
    #del wordseginst    
    #TCWordSeg3.uninitconf() #仅需加卸载一次
    
 
