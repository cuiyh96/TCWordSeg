#!/usr/bin/env python
#coding=utf-8
#for python3 
import sys,os
from .TCWordSeg import *

ecd_set = 'gb18030'
#ecd_set = 'utf-8'

class TCWordSeg3(object):

    def __init__(self):
        SEG_MODE = TC_U2L|TC_POS|TC_S2D|TC_CONV|TC_T2S|TC_ENGU|TC_CN
        SEG_MODE_CLS = SEG_MODE | TC_USR | TC_CLS 

        self.seghandle = TCCreateSegHandle(SEG_MODE_CLS) #每个线程创建一个对象，进程内顺序执行
        self.fineseghandle = TCCreateSegHandle(SEG_MODE)
        self.posbuf = bytes(b'\0'*6)
         
    def __del__(self):
        TCCloseSegHandle(self.seghandle)  #释放实例资源
        TCCloseSegHandle(self.fineseghandle) 

    def TCGetWord(self, seghandle, idx):  #获得分词
        wordpos = TCGetAt(seghandle, idx)
        word = wordpos.word.decode('gb18030', errors='ignore')
        return word
        
    def TCGetWordPos(self, seghandle, idx):  #获得词性
        wordpos = TCGetAt(seghandle, idx)
        TCPosId2Str(wordpos.pos,  self.posbuf, 6) 
        posbuf = self.posbuf.decode('gb18030').strip('\0')  #输出也是GBK
        return posbuf

    def TCGetWordCls(self, seghandle, idx):  #获得自定义词类标识
        wordpos = TCGetAt(seghandle, idx)
        wordcls = ''
        if (wordpos.bcw and wordpos.cls != -1):
            wordcls = TCGetClsAt(seghandle, wordpos.cls, 1)
        if wordcls != '':
            wordcls = wordcls.decode('gb18030') 
        return wordcls

    def TCGetResultCnt(self, seghandle):   #获得粗切分分词数目
        rescount = TCGetResultCnt(seghandle)
        return rescount
        
    def TCSegment(self, seghandle, line):  #进行切分，一般场景使用
        TCSegment(seghandle, line.encode('gb18030', errors='ignore')) 

    def TCSegOutput(self, seghandle, line):  #进行切分，并输出按空格分隔的串
        TCSegment(seghandle, line.encode('gb18030', errors='ignore')) 
        rescount = TCGetResultCnt(seghandle)
        return ' '.join([self.TCGetWord(seghandle, i) for i in range(rescount)])

    def TCSegOutputPos(self, seghandle, line):  #进行切分，并输出按空格分隔的串以及词性
        TCSegment(seghandle, line.encode('gb18030', errors='ignore')) 
        rescount = TCGetResultCnt(seghandle)
        return ' '.join([self.TCGetWord(seghandle, i) + '/'+ self.TCGetWordPos(seghandle, i)  for i in range(rescount)])

    @staticmethod
    def initconf(dict):  #resouce only load once in one process
        origin_workdir = os.getcwd()
        cur_file_dir = os.path.dirname(__file__)
        #print(cur_file_dir)
        #print(origin_workdir)
        #os.chdir(cur_file_dir)

        nbSucc = TCInitSeg(dict.encode('gb18030'))    #每个进程加载一次，进程共享
        #os.chdir(origin_workdir)
        return nbSucc
        
    @staticmethod
    def uninitconf(): #resouce only unload once in one process
        TCUnInitSeg()

# python TCWordSeg3.py testutf.txt 1 
setnum = ['一', '二', '三', '四', '五', '六', '七', '八', '九']
months = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
if __name__ == '__main__':   # 
    TCWordSeg3.initconf("data") #词典资源仅需加载一次
    
    fp = open(sys.argv[1], 'r', encoding=ecd_set, errors="ignore")  #如果输入为utf的，encoding='utf-8'，如果是gbk，encoding='gb18030'
    outs = open(sys.argv[1] + 'out.txt', 'w', encoding='utf-8', errors="ignore" ) # sys.stdout #
    flag = sys.argv[2] # 1输出词性和词类比，0输出分词
    
    wordseginst = TCWordSeg3() #可创建多个实例
    for line in fp:     
        line = line.strip()
        line_seg = ''
        if flag == '0':
            line_seg = wordseginst.TCSegOutput(wordseginst.seghandle, line)
        else:
            line_seg = wordseginst.TCSegOutputPos(wordseginst.seghandle, line)
        '''
    
        wordseginst.TCSegment(wordseginst.seghandle, line) #进行大细粒度切分，为实体识别、分类、关键词使用
        
        rescount = wordseginst.TCGetResultCnt(wordseginst.seghandle)  #获得粗粒度
        line_seg = ''
        for i in range(rescount):
            # 只输出词
            # 输出词和词性
            word = wordseginst.TCGetWord(wordseginst.seghandle, i) 
            
            wordseginst.TCSegment(wordseginst.fineseghandle, word)
            finerescount = wordseginst.TCGetResultCnt(wordseginst.fineseghandle) 
            for isub in range(finerescount):
                fineword = wordseginst.TCGetWord(wordseginst.fineseghandle, isub)
                finepos = wordseginst.TCGetWordPos(wordseginst.fineseghandle, isub)
            
            posbuf = wordseginst.TCGetWordPos(wordseginst.seghandle, i)
            wordcls = wordseginst.TCGetWordCls(wordseginst.seghandle, i)
            line_seg += ' ' if i!=0 else ''
            if flag == '0':
                line_seg += '%s' %(word)
            else: 
                if posbuf == 'm' and (word[0].isdigit() and len(word)>=2 or len(word)>=2 and word[0] in setnum and  word[1] in setnum):
                    line_seg += '%s\t%s' %(line, '[NUM]')
                elif word not in months and posbuf == 't' and word[0].isdigit():
                        line_seg += '%s\t%s' %(line, '[TIME]')
                elif posbuf == 'email':
                        line_seg += '%s\t%s' %(line, '[EML]')
                elif posbuf == 'tele':
                        line_seg += '%s\t%s' %(line, '[TEL]') 
                elif posbuf == 'nx' and len(word) > 18:
                        line_seg += '%s\t%s' %(line, '[UNK]')
                break
        '''
        if line_seg:
            outs.write(line_seg + '\n' )
    del wordseginst    
    TCWordSeg3.uninitconf() #仅需加卸载一次
