#coding=utf-8
"""分词行为快照测试。

原生引擎 _TCWordSeg 由外部提供、无源码，且锁定 CPython 3.8，无法重建；
因此用一份固定输入 → 期望切词结果的快照作为回归基线：
将来更换 Python / 平台 / 引擎时，用它验证"是否仍是同一个分词器"。

用法（生成与比对均需在 Python 3.8 下执行）：
    首次生成基线：TCWORDSEG_SNAPSHOT_RECORD=1 python -m pytest tests/test_seg_snapshot.py
    回归比对：    python -m pytest tests/test_seg_snapshot.py
"""
import json
import os
import sys

import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.append(_project_root)

from wordseg import tokenizer

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seg_snapshot.jsonl')

# 基线记录的是原生引擎的行为，故固定用原生引擎，不受外部 TCWORDSEG_ENGINE 影响
os.environ[tokenizer.ENGINE_ENV] = tokenizer.ENGINE_TCWORDSEG

# 覆盖 README 宣称的能力：机构/人名识别、数字时间电话邮箱、繁简转换、中英混合、空串
SNAPSHOT_CASES = [
    '用友网络科技股份有限公司',
    '张三在2024年3月15日给李四打了13812345678',
    '请把资料发到 service@example.com',
    '繁體中文分詞測試與簡體轉換',
    'AI模型训练需要清洗语料，GPU显存占用为12345MB。',
    '《三体》作者刘慈欣，出版社是重庆出版社。',
    '价格是 3.14 元，涨幅 12.5%。',
    'TCWordSeg 支持 s1/s2 格式。',
    '',
]

_ENGINE_UNAVAILABLE = not tokenizer.is_available()
_SKIP_REASON = '原生引擎 _TCWordSeg 在当前解释器不可用（锁定 CPython 3.8）'


def test_import_is_lazy():
    """import wordseg.tokenizer 不应因缺少 3.8 原生扩展而失败。"""
    assert callable(tokenizer.segment)
    assert callable(tokenizer.segment_with_pos)


@pytest.mark.skipif(_ENGINE_UNAVAILABLE, reason=_SKIP_REASON)
def test_segment_snapshot():
    tokenizer.init()
    actual = [{'text': text, 'words': tokenizer.segment(text)} for text in SNAPSHOT_CASES]

    if os.environ.get('TCWORDSEG_SNAPSHOT_RECORD') == '1':
        with open(GOLDEN_PATH, 'w', encoding='utf-8') as f:
            for item in actual:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        pytest.skip('已重新生成快照基线：%s' % GOLDEN_PATH)

    assert os.path.exists(GOLDEN_PATH), (
        '缺少快照基线文件 %s。请在 Python 3.8 下执行：'
        'TCWORDSEG_SNAPSHOT_RECORD=1 python -m pytest tests/test_seg_snapshot.py' % GOLDEN_PATH)

    with open(GOLDEN_PATH, 'r', encoding='utf-8') as f:
        golden = [json.loads(line) for line in f if line.strip()]

    assert [item['words'] for item in actual] == [item['words'] for item in golden], \
        '分词结果与基线不一致，引擎行为可能已变化'
