from rwrt.types import Candidate, format_candidate_scores


def test_final_score_prefers_cross() -> None:
    c = Candidate(word="kitap", bi_score=0.9, feature_score=0.5, cross_score=0.1)
    assert c.final_score == 0.1


def test_final_score_prefers_feature_over_bi() -> None:
    c = Candidate(word="kitap", bi_score=0.9, feature_score=0.5)
    assert c.final_score == 0.5


def test_final_score_falls_back_to_bi() -> None:
    c = Candidate(word="kitap", bi_score=0.75)
    assert c.final_score == 0.75


def test_final_score_negative_inf_when_unscored() -> None:
    c = Candidate(word="kitap")
    assert c.final_score == float("-inf")


def test_format_candidate_scores_shows_weighted_feature() -> None:
    c = Candidate(word="kitap", bi_score=0.9, feature_score=0.55, frequency=100)
    text = format_candidate_scores(c)
    assert "wf=0.5500" in text
    assert "cross=—" in text
