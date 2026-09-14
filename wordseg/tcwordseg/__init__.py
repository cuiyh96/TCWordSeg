#coding=utf-8
"""原生引擎 _TCWordSeg 的分词实现（wordseg 的 tcwordseg 引擎子包）。

与 wordseg.jieba 平级，二者对外接口一致（segment / segment_with_pos /
init / uninit / is_available），由 wordseg.tokenizer 统一入口按环境变量分发。

引擎私有文件都在本子包内，与公共模块隔离：

- TCWordSeg.py（SWIG 绑定）、TCWordSeg3.py（面向对象封装）；
- _TCWordSeg.pyd / .so（原生扩展）、data/（词典资源）；
- 原生扩展由外部提供、无源码，且锁定 CPython 3.8，因此采用**延迟导入**：
  其他 Python 版本下 import 本模块不会报错，只有真正调用分词时才会抛出带修复提示的
  RuntimeError；
- init() 幂等，多进程（含 Windows spawn）中每个子进程自行调用即可。
"""
import os
import threading

# 词典资源目录（wordseg/tcwordseg/data），固定用绝对路径，避免依赖调用方的工作目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

_lock = threading.Lock()
_local = threading.local()
_inited = False

_ENGINE_HINT = (
    "无法加载 TCWordSeg 原生引擎 _TCWordSeg：{err}\n"
    "该扩展由外部提供、无源码，且锁定 CPython 3.8（ABI 不匹配），无法重建。\n"
    "请改用 Python 3.8 运行，或将其放入 Python 3.8 子进程中调用；\n"
    "也可改用 jieba 引擎：设置环境变量 TCWORDSEG_ENGINE=jieba。"
)


def _engine():
    """延迟导入原生引擎封装类 TCWordSeg3。"""
    try:
        from .TCWordSeg3 import TCWordSeg3
    except ImportError as err:
        raise RuntimeError(_ENGINE_HINT.format(err=err)) from err
    return TCWordSeg3


def is_available():
    """当前解释器能否加载原生引擎（不抛异常）。"""
    try:
        _engine()
    except RuntimeError:
        return False
    return True


def init(dict_dir=None):
    """加载词典资源，每个进程仅需一次（幂等，可重复调用）。

    Args:
        dict_dir: 词典资源目录，默认 wordseg/tcwordseg/data
    """
    global _inited
    if _inited:
        return
    with _lock:
        if _inited:
            return
        _engine().initconf(dict_dir or DATA_DIR)
        _inited = True


def uninit():
    """卸载词典资源，进程退出前调用一次。"""
    global _inited
    with _lock:
        if not _inited:
            return
        _engine().uninitconf()
        _inited = False


def _instance():
    """取当前线程的分词实例（引擎要求：每线程一个实例，进程内顺序执行）。"""
    init()   # 幂等；uninit 之后再次分词时确保资源已重新加载
    inst = getattr(_local, 'inst', None)
    if inst is None:
        inst = _engine()()
        _local.inst = inst
    return inst


def segment(text):
    """分词。

    Args:
        text: 待切分文本

    Returns:
        词列表；输入为空时返回 []。
    """
    if not text:
        return []
    inst = _instance()
    inst.TCSegment(inst.seghandle, text)
    rescount = inst.TCGetResultCnt(inst.seghandle)
    return [inst.TCGetWord(inst.seghandle, i) for i in range(rescount)]


def segment_with_pos(text):
    """分词 + 词性标注。

    Returns:
        '词/词性' 列表，如 ['用友/nr', '网络/n']；输入为空时返回 []。
    """
    if not text:
        return []
    inst = _instance()
    inst.TCSegment(inst.seghandle, text)
    rescount = inst.TCGetResultCnt(inst.seghandle)
    return ['%s/%s' % (inst.TCGetWord(inst.seghandle, i),
                       inst.TCGetWordPos(inst.seghandle, i))
            for i in range(rescount)]
