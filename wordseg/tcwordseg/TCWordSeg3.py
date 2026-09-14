#!/usr/bin/env python
#coding=utf-8
"""原生引擎 _TCWordSeg 的 Python3 面向对象封装（SWIG 绑定见 TCWordSeg.py）。

由 wordseg.tcwordseg（引擎子包入口）统一调用：词典资源每进程加载一次（initconf），
分词实例每线程各持一个（引擎要求进程内顺序执行）。
"""
from .TCWordSeg import *


class TCWordSeg3(object):

    def __init__(self):
        SEG_MODE = TC_U2L|TC_POS|TC_S2D|TC_CONV|TC_T2S|TC_ENGU|TC_CN
        SEG_MODE_CLS = SEG_MODE | TC_USR | TC_CLS 

        self.seghandle = TCCreateSegHandle(SEG_MODE_CLS) #每个线程创建一个对象，进程内顺序执行
        self.posbuf = bytes(b'\0'*6)

    def __del__(self):
        TCCloseSegHandle(self.seghandle)  #释放实例资源

    def TCGetWord(self, seghandle, idx):  #获得分词
        wordpos = TCGetAt(seghandle, idx)
        word = wordpos.word.decode('gb18030', errors='ignore')
        return word
        
    def TCGetWordPos(self, seghandle, idx):  #获得词性
        wordpos = TCGetAt(seghandle, idx)
        TCPosId2Str(wordpos.pos,  self.posbuf, 6) 
        posbuf = self.posbuf.decode('gb18030').strip('\0')  #输出也是GBK
        return posbuf

    def TCGetResultCnt(self, seghandle):   #获得粗切分分词数目
        rescount = TCGetResultCnt(seghandle)
        return rescount
        
    def TCSegment(self, seghandle, line):  #进行切分，一般场景使用
        TCSegment(seghandle, line.encode('gb18030', errors='ignore')) 

    @staticmethod
    def initconf(dict_dir):  #resouce only load once in one process
        # 词典资源目录由调用方给出（绝对路径），不依赖当前工作目录
        return TCInitSeg(dict_dir.encode('gb18030'))    #每个进程加载一次，进程共享
        
    @staticmethod
    def uninitconf(): #resouce only unload once in one process
        TCUnInitSeg()
