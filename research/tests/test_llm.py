import pytest
from alphax_lab import llm


@pytest.fixture()
def env_files(tmp_path, monkeypatch):
    package, rdagent = tmp_path / "package.env", tmp_path / "rdagent.env"
    package.write_text('# comment\nALPHAX_LAB_MODEL="openai/package-model"\n')
    rdagent.write_text("CHAT_MODEL=rdagent-model\nOPENAI_API_BASE=https://rdagent/v1\nOPENAI_API_KEY='rd-key'\n")
    monkeypatch.setattr(llm, "PACKAGE_ENV_FILE", package)
    monkeypatch.setattr(llm, "RDAGENT_ENV_FILE", rdagent)
    for name in ("ALPHAX_LAB_MODEL", "ALPHAX_LAB_API_BASE", "ALPHAX_LAB_API_KEY", "CHAT_MODEL", "OPENAI_API_BASE", "OPENAI_API_KEY", "ALPHAX_LAB_ENV_FILE"):
        monkeypatch.delenv(name, raising=False)
    return package, rdagent


def test_package_env_wins_and_rdagent_fills_the_rest(env_files):
    client = llm.LLM()
    assert client.model == "package-model"  # the openai/ prefix is stripped
    assert client.base_url == "https://rdagent/v1" and client.api_key == "rd-key"


def test_environment_overrides_files(env_files, monkeypatch):
    monkeypatch.setenv("ALPHAX_LAB_MODEL", "env-model")
    assert llm.LLM().model == "env-model"


def test_explicit_env_file_replaces_the_defaults(env_files, monkeypatch, tmp_path):
    other = tmp_path / "other.env"
    other.write_text("ALPHAX_LAB_MODEL=other-model\nALPHAX_LAB_API_KEY=other-key\n")
    monkeypatch.setenv("ALPHAX_LAB_ENV_FILE", str(other))
    client = llm.LLM()
    assert (client.model, client.api_key, client.base_url) == ("other-model", "other-key", None)


def test_missing_model_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "PACKAGE_ENV_FILE", tmp_path / "none.env")
    monkeypatch.setattr(llm, "RDAGENT_ENV_FILE", tmp_path / "none2.env")
    for name in ("ALPHAX_LAB_MODEL", "CHAT_MODEL", "ALPHAX_LAB_ENV_FILE"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="ALPHAX_LAB_MODEL"):
        llm.LLM()


@pytest.mark.parametrize(
    "text",
    ['{"a": 1}', 'here you go:\n```json\n{"a": 1}\n```', 'prose {"a": 1} trailing words'],
)
def test_parse_json_survives_fences_and_prose(text):
    assert llm.parse_json(text) == {"a": 1}


def test_parse_json_rejects_non_json():
    with pytest.raises(ValueError, match="no JSON object"):
        llm.parse_json("sorry, I cannot help with that")
