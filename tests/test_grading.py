from eval.compare_chunks import graded


def test_unicode_spacing_and_hyphens_fold():
    assert graded("within 2 hours", [["2 hours"]])
    assert graded("a three‑week course in Denver", [["three weeks", "three-week"], ["denver"]])
    assert graded("kept at 4 °C or lower", [["4 degrees", "4 °c"]])


def test_whole_token_matching():
    assert not graded("revoked within 24 hours", [["4 hours"]])
    assert not graded("minimum 140 characters", [["14"]])
    assert not graded("version 3.85", [["85"]])
    assert graded("a 6% royalty", [["6%"]])


def test_all_groups_required():
    assert not graded("three weeks of training", [["three weeks"], ["denver"]])
