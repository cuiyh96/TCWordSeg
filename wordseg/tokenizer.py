#coding=utf-8
"""统一分词入口：按环境变量 TCWORDSEG_ENGINE 分发到具体引擎实现。

两种引擎在包内平级（wordseg.tcwordseg、wordseg.jieba），对外接口完全一致，
调用方只依赖本模块：

- ``tcwordseg``（默认）：原生引擎 _TCWordSeg，见 wordseg.tcwordseg；
- ``jieba``：纯 Python 分词器（依赖 jieba 包），支持自定义分词表，见 wordseg.jieba。

切换只通过环境变量完成，多进程下子进程会继承，无需改代码::

    set TCWORDSEG_ENGINE=jieba        # Windows
    export TCWORDSEG_ENGINE=jieba     # Linux/macOS

init() 幂等，多进程（含 Windows spawn）中每个子进程自行调用即可。
"""
import os

# 引擎选择：环境变量名与可选取值
ENGINE_ENV = 'TCWORDSEG_ENGINE'
ENGINE_TCWORDSEG = 'tcwordseg'
ENGINE_JIEBA = 'jieba'


def current_engine():
    """返回当前生效的引擎名（由环境变量 TCWORDSEG_ENGINE 决定，默认 tcwordseg）。"""
    name = os.environ.get(ENGINE_ENV, ENGINE_TCWORDSEG).strip().lower()
    if name not in (ENGINE_TCWORDSEG, ENGINE_JIEBA):
        raise ValueError('不支持的分词引擎 %r（环境变量 %s），可选：%s、%s'
                         % (name, ENGINE_ENV, ENGINE_TCWORDSEG, ENGINE_JIEBA))
    return name


def _native():
    """延迟导入原生引擎子包。"""
    from . import tcwordseg
    return tcwordseg


def _jieba():
    """延迟导入 jieba 引擎子包。"""
    from . import jieba
    return jieba


def _engine_module():
    """返回当前环境变量选定的引擎实现模块。"""
    if current_engine() == ENGINE_JIEBA:
        return _jieba()
    return _native()


def is_available():
    """当前引擎在当前解释器是否可用（不抛异常）。"""
    return _engine_module().is_available()


def init(dict_dir=None):
    """加载引擎资源，每个进程仅需一次（幂等，可重复调用）。

    Args:
        dict_dir: 原生引擎的词典资源目录，默认 wordseg/tcwordseg/data；
            jieba 引擎忽略该参数，其自定义分词表见 wordseg.jieba。
    """
    if current_engine() == ENGINE_JIEBA:
        _jieba().init()
        return
    _native().init(dict_dir)


def uninit():
    """卸载引擎资源，进程退出前调用一次（jieba 无卸载动作，为空操作）。"""
    _engine_module().uninit()


def segment(text):
    """分词。

    Args:
        text: 待切分文本

    Returns:
        词列表；输入为空时返回 []。
    """
    if not text:
        return []
    return _engine_module().segment(text)


def segment_with_pos(text):
    """分词 + 词性标注。

    Returns:
        '词/词性' 列表，如 ['用友/nr', '网络/n']；输入为空时返回 []。
    """
    if not text:
        return []
    return _engine_module().segment_with_pos(text)
