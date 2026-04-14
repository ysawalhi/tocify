import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digest import keyword_prefilter


def _item(title="", summary=""):
    return {"id": f"{title}|{summary}", "title": title, "summary": summary}


def test_title_match_passes_filter():
    matching = [_item(title=f"Quantum paper {i}") for i in range(2)]
    non_matching = _item(title="Cooking recipes", summary="flour and sugar")
    items = matching + [non_matching]

    result = keyword_prefilter(items, ["quantum"], keep_top=2)

    for m in matching:
        assert m in result
    assert non_matching not in result


def test_summary_match_passes_filter():
    matching = [
        _item(title=f"Paper {i}", summary="a study on quantum computing")
        for i in range(2)
    ]
    non_matching = _item(title="Weather report", summary="sunshine all day")
    items = matching + [non_matching]

    result = keyword_prefilter(items, ["quantum"], keep_top=2)

    for m in matching:
        assert m in result
    assert non_matching not in result


def test_no_keyword_match_rejected():
    matching = [_item(title=f"Quantum study {i}") for i in range(2)]
    non_matching = _item(title="Baseball recap", summary="home runs and strikeouts")
    items = matching + [non_matching]

    result = keyword_prefilter(items, ["quantum"], keep_top=2)

    assert non_matching not in result
    assert len(result) == 2


def test_matching_is_case_insensitive():
    matching = [_item(title=f"QUANTUM PAPER {i}") for i in range(2)]
    non_matching = _item(title="Baseball recap")
    items = matching + [non_matching]

    result = keyword_prefilter(items, ["quantum"], keep_top=2)

    for m in matching:
        assert m in result
    assert non_matching not in result


def test_empty_input_returns_empty_list():
    assert keyword_prefilter([], ["quantum"], keep_top=10) == []
