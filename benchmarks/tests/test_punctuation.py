import pytest
from benchmarks.core.punctuation import corpus_punctuation_scores, punctuation_scores


class TestPunctuationScores:
    def test_perfect_match(self):
        s = punctuation_scores("Hello, world. How are you?", "Hello, world. How are you?")
        assert s["overall"]["f1"] == 1.0
        assert s["casing_accuracy"] == 1.0

    def test_missing_punctuation_is_recall_loss(self):
        s = punctuation_scores("Hello, world.", "Hello world")
        assert s["overall"]["recall"] == 0.0
        assert s["overall"]["counts"]["fn"] == 2

    def test_spurious_punctuation_is_precision_loss(self):
        s = punctuation_scores("Hello world", "Hello, world.")
        assert s["overall"]["precision"] == 0.0
        assert s["overall"]["counts"]["fp"] == 2

    def test_wrong_mark_counts_as_both(self):
        s = punctuation_scores("Really?", "Really.")
        assert s["per_mark"]["?"]["counts"]["fn"] == 1
        assert s["per_mark"]["."]["counts"]["fp"] == 1

    def test_casing_is_scored(self):
        s = punctuation_scores("John went to Paris", "john went to paris")
        assert s["casing_accuracy"] == pytest.approx(0.5)

    def test_transcription_errors_do_not_punish_punctuation(self):
        # "world"/"word" misrecognised: only agreed words contribute, and the drop in
        # alignment is reported rather than hidden.
        s = punctuation_scores("Hello, world. Goodbye.", "Hello, word. Goodbye.")
        assert s["alignment_rate"] < 1.0
        assert s["overall"]["f1"] == 1.0

    def test_quotes_and_brackets_do_not_hide_the_mark(self):
        s = punctuation_scores('He said "yes".', 'He said "yes".')
        assert s["overall"]["f1"] == 1.0

    def test_empty_inputs_are_safe(self):
        s = punctuation_scores("", "")
        assert s["overall"]["f1"] == 0.0 and s["aligned_words"] == 0

    def test_corpus_pools_counts(self):
        c = corpus_punctuation_scores(["Hi, there.", "Yes."], ["Hi there", "Yes."])
        assert c["overall"]["counts"]["tp"] == 1
        assert c["overall"]["counts"]["fn"] == 2

    def test_corpus_length_mismatch_is_an_error(self):
        with pytest.raises(ValueError):
            corpus_punctuation_scores(["a"], [])
