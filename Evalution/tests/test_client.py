import os
import pytest

from Evalution import client


def test_client_uses_openrouter_defaults(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("MODEL", raising=False)

    client._client = None
    assert client.get_llm_api_key() == "test-key"
    
    # Should raise ValueError when no model is configured
    with pytest.raises(ValueError, match="OPENROUTER_MODEL"):
        client.get_llm_model()


def test_client_uses_env_model(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-3-haiku")
    
    client._client = None
    assert client.get_llm_model() == "anthropic/claude-3-haiku"
