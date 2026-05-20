from rwrt.types import Candidate


def test_final_score_prefers_cross() -> None:
    c = Candidate(word="kitap", bi_score=0.9, cross_score=0.1)
    assert c.final_score == 0.1


def test_final_score_falls_back_to_bi() -> None:
    c = Candidate(word="kitap", bi_score=0.75)
    assert c.final_score == 0.75


def test_final_score_negative_inf_when_unscored() -> None:
    c = Candidate(word="kitap")
    assert c.final_score == float("-inf")
