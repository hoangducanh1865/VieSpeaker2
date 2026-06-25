import os


def test_entry_script_paths_exist():
    from viespeaker import pipeline_api as A
    for p in (A.P1_SCRIPT, A.P2_SCRIPT, A.P3_SCRIPT, A.FUSION_SCRIPT, A.EVAL_SCRIPT):
        assert os.path.exists(p), p


def test_stem():
    from viespeaker import pipeline_api as A
    assert A._stem("/a/b/interview_noise.wav") == "interview_noise"
    assert A._stem("movie.mp4") == "movie"


def test_run_returncodes():
    from viespeaker import pipeline_api as A
    assert A.run(["true"]) is True
    assert A.run(["false"]) is False
