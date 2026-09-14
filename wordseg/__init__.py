#coding=utf-8
"""TCWordSeg：中文分词与文本去重工具包。

- 分词：统一入口 segment / segment_with_pos，引擎由环境变量 TCWORDSEG_ENGINE 选择；
  两个引擎子包平级：wordseg.tcwordseg（原生引擎）、wordseg.jieba（jieba 引擎）；
- 去重：仓库根目录的独立包 dedup（MinHash + LSH），按需单独 import。
"""
from .tokenizer import (
    ENGINE_ENV,
    ENGINE_JIEBA,
    ENGINE_TCWORDSEG,
    current_engine,
    init,
    is_available,
    segment,
    segment_with_pos,
    uninit,
)

__all__ = [
    'ENGINE_ENV', 'ENGINE_TCWORDSEG', 'ENGINE_JIEBA',
    'current_engine', 'init', 'uninit', 'is_available',
    'segment', 'segment_with_pos',
]
