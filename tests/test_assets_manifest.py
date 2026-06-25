def test_manifest_structure():
    import viespeaker.assets_manifest as M
    import viespeaker.paths as P

    assert M.ALL == M.WEIGHTS + M.DATA
    assert len(M.ALL) > 0

    # severities are valid; at least the 5 core embedding/vbx weights exist
    assert {a.severity for a in M.ALL} <= {"core", "soft"}
    assert sum(1 for a in M.WEIGHTS if a.severity == "core") >= 4

    # every destination lives under the assets root
    for a in M.ALL:
        assert str(a.dest).startswith(str(P.ASSETS_ROOT)), a

    # no duplicate destinations
    dests = [str(a.dest) for a in M.ALL]
    assert len(dests) == len(set(dests))

    # `old` paths are repo-relative (used to build relocate cp commands)
    for a in M.ALL:
        assert not a.old.startswith("/"), a
        assert "/" in a.old
