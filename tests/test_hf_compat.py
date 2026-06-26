from viespeaker.hf_compat import pretrained_auth_kwargs, translate_legacy_hf_auth


def test_pretrained_auth_kwargs_supports_old_and_new_loaders():
    def old_loader(checkpoint, use_auth_token=None):
        pass

    def new_loader(checkpoint, token=None):
        pass

    assert pretrained_auth_kwargs(old_loader, "secret") == {"use_auth_token": "secret"}
    assert pretrained_auth_kwargs(new_loader, "secret") == {"token": "secret"}
    assert pretrained_auth_kwargs(old_loader, None) == {}


def test_translate_legacy_hf_auth():
    calls = []

    def download(repo_id, filename, *, token=None):
        calls.append((repo_id, filename, token))
        return "/tmp/model"

    wrapped = translate_legacy_hf_auth(download)

    assert wrapped("owner/model", "config.yaml", use_auth_token="secret") == "/tmp/model"
    assert calls == [("owner/model", "config.yaml", "secret")]
