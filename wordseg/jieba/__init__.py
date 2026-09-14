#coding=utf-8
"""基于 jieba 的分词实现，作为原生引擎 _TCWordSeg 的可选替代。

对外提供与 tokenizer 一致的 segment / segment_with_pos，由 tokenizer 统一分发。
选用本引擎：设置环境变量 TCWORDSEG_ENGINE=jieba。

自定义分词表（jieba 用户词典）可以有多张，来源按优先级：
1. init(userdict_path=...) 显式传入的路径（文件、目录，或路径序列）；
2. 环境变量 TCWORDSEG_JIEBA_DICT 指定的路径，多个用 os.pathsep 分隔
   （Windows 为 ';'，Linux/macOS 为 ':'）；
3. 默认目录 wordseg/jieba/data 下的全部 *.txt（仓库自带 userdict.txt 样例，零配置生效）。
目录会被展开为其下全部 *.txt（按文件名排序）；词典格式与 jieba 一致：
每行「词语 [词频] [词性]」，词频、词性可省略，**不支持注释行**。

多进程场景（Windows spawn）子进程会重新导入本模块，因此自定义词典路径
通过环境变量传递最稳妥（子进程会继承父进程的环境变量）。
"""
import os
import threading

# 自定义分词表路径的环境变量，与 tokenizer.ENGINE_ENV 一样便于多进程传递
JIEBA_DICT_ENV = 'TCWORDSEG_JIEBA_DICT'
# 默认词表目录，仓库自带的 userdict.txt 即放在这里
DICT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

_ENGINE_HINT = (
    "无法加载 jieba 分词器：{err}\n"
    "请先安装：pip install jieba"
)

_lock = threading.Lock()
_inited = False
_loaded_dicts = []


def _jieba():
    """延迟导入 jieba，未安装时抛出带修复提示的 RuntimeError。"""
    try:
        import jieba
    except ImportError as err:
        raise RuntimeError(_ENGINE_HINT.format(err=err)) from err
    return jieba


def is_available():
    """当前解释器能否加载 jieba（不抛异常）。"""
    try:
        _jieba()
    except RuntimeError:
        return False
    return True


def _dict_files(path):
    """把「文件或目录」展开成词表文件列表：目录取其下全部 *.txt（按文件名排序）。"""
    if os.path.isdir(path):
        return [os.path.join(path, name) for name in sorted(os.listdir(path))
                if name.lower().endswith('.txt')
                and os.path.isfile(os.path.join(path, name))]
    if os.path.isfile(path):
        return [path]
    raise FileNotFoundError('自定义分词表路径不存在：%s' % path)


def resolve_dict_files(userdict_path=None):
    """解析当前应加载的词表文件列表（去重且保持顺序）。

    优先级：显式传入的路径 > 环境变量 TCWORDSEG_JIEBA_DICT > 默认目录 DICT_DIR。
    路径可以是文件或目录；显式指定的路径不存在时抛 FileNotFoundError，
    默认目录不存在则视为没有自定义词表（不报错）。

    Args:
        userdict_path: 词表文件或目录，也可传路径序列；None 表示按环境变量、默认目录解析。

    Returns:
        词表文件路径列表，如 ['.../wordseg/jieba/data/userdict.txt']。
    """
    if userdict_path is None:
        raw = os.environ.get(JIEBA_DICT_ENV)
        if raw:
            paths = [path.strip() for path in raw.split(os.pathsep) if path.strip()]
        else:
            paths = [DICT_DIR] if os.path.isdir(DICT_DIR) else []
    elif isinstance(userdict_path, (str, os.PathLike)):
        paths = [userdict_path]
    else:
        paths = list(userdict_path)

    files = []
    for path in paths:
        files.extend(_dict_files(path))
    return list(dict.fromkeys(files))


def init(userdict_path=None):
    """初始化 jieba 并加载自定义分词表（幂等，可重复调用）。

    Args:
        userdict_path: 词表文件或目录，也可传路径序列；为 None 时按
            「环境变量 > 默认目录 wordseg/jieba/data」解析。
            同一批路径不会重复加载，切换路径时加载新词表（jieba 词典只增不减）。
    """
    global _inited, _loaded_dicts
    files = resolve_dict_files(userdict_path)
    with _lock:
        if _inited and files == _loaded_dicts:
            return
        jieba = _jieba()
        jieba.initialize()
        for path in files:
            jieba.load_userdict(path)
        _loaded_dicts = files
        _inited = True


def load_userdict(userdict_path):
    """显式加载自定义分词表（等价于 init(userdict_path)）。"""
    init(userdict_path)


def uninit():
    """jieba 的词典常驻进程内，无卸载动作；保持与原生引擎一致的接口。"""


def _ensure_inited():
    """确保已初始化：仅在未初始化时按默认来源加载词表。

    不用 ``init(userdict_path=None)`` 重新解析词表，避免在用户先显式
    ``init(userdict_path=...)`` 后调用分词时，又把默认/环境词表叠加进来；
    也免去每条文本一次文件系统解析的开销。
    """
    if not _inited:
        init()


def segment(text):
    """分词。

    Args:
        text: 待切分文本

    Returns:
        词列表；输入为空时返回 []。
    """
    if not text:
        return []
    _ensure_inited()
    return list(_jieba().cut(text))


def segment_with_pos(text):
    """分词 + 词性标注。

    Returns:
        '词/词性' 列表，如 ['用友/nr', '网络/n']；输入为空时返回 []。
    """
    if not text:
        return []
    _ensure_inited()
    import jieba.posseg as pseg
    return ['%s/%s' % (word, flag) for word, flag in pseg.cut(text)]
