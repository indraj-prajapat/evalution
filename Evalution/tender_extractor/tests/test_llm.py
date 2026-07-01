"""Tests for the LLM client and extractor modules."""

import json
import pytest

from Evalution.tender_extractor.llm_extractor import (
    _build_extraction_prompt,
    _compact_json,
    _collect_doc_text,
)


# ---------------------------------------------------------------------------
# Prompt building tests (no API key needed)
# ---------------------------------------------------------------------------

class TestPromptBuilding:
    """Test that extraction prompts are built correctly."""

    def _make_doc(self, name="Test Doc", category="Test", pages=None):
        from Evalution.tender_extractor.models import DetectedDocument, PageSummary
        if pages is None:
            pages = [PageSummary(page_number=1, text="Sample text with value Rs. 1,00,000")]
        return DetectedDocument(
            document_id="doc_1",
            document_name=name,
            document_category=category,
            pages=pages,
        )

    def test_build_prompt_basic(self):
        doc = self._make_doc()
        fields = [
            {"name": "test_field", "datatype": "string", "description": "A test"},
        ]
        from Evalution.tender_extractor.models import FieldSpec
        field_specs = [FieldSpec(**f) for f in fields]

        prompt = _build_extraction_prompt(doc, field_specs)

        assert "Test Doc" in prompt
        assert "test_field" in prompt
        assert "string" in prompt
        assert "Sample text" in prompt

    def test_build_prompt_with_examples(self):
        doc = self._make_doc()
        fields = [
            {
                "name": "amount",
                "datatype": "currency",
                "description": "The amount",
                "examples": ["Rs. 1,00,000", "Rs. 5,00,000"],
            },
        ]
        from Evalution.tender_extractor.models import FieldSpec
        field_specs = [FieldSpec(**f) for f in fields]

        prompt = _build_extraction_prompt(doc, field_specs)
        assert "amount" in prompt
        assert "currency" in prompt

    def test_collect_doc_text_respects_budget(self):
        """Document text should be truncated to max_chars."""
        doc = self._make_doc(
            pages=[
                __import__("tender_extractor.models", fromlist=["PageSummary"]).PageSummary(
                    page_number=1, text="A" * 5000
                )
            ]
        )
        text = _collect_doc_text(doc, max_chars=1000)
        assert len(text) <= 1100  # Some overhead for headers

    def test_compact_json(self):
        obj = {"name": "test", "value": [1, 2, 3]}
        result = _compact_json(obj)
        parsed = json.loads(result)
        assert parsed == obj
        # Should not have extra spaces
        assert " " not in result


# ---------------------------------------------------------------------------
# LLM client tests (require API key to run fully)
# ---------------------------------------------------------------------------

class TestLLMClient:
    """Test LLM client initialization and error handling."""

    def test_no_api_key_raises(self):
        """Without OPENAI_API_KEY, should raise ValueError."""
        import os
        # Ensure no key is set
        old_key = os.environ.pop("OPENAI_API_KEY", None)

        try:
            # Reset the singleton
            import Evalution.tender_extractor.llm_client as lc
            lc._client = None

            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                lc.get_client()
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key
            lc._client = None

    def test_env_file_parsing(self):
        """Test that .env file loading works."""
        import os
        import tempfile

        # Create a temp directory with a .env file
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = os.path.join(tmpdir, ".env")
            with open(env_path, "w") as f:
                f.write('OPENAI_API_KEY="sk-test-123"\n# comment\nANOTHER_KEY=value')

            # Reset singleton
            import Evalution.tender_extractor.llm_client as lc
            lc._client = None

            # Save and clear env vars
            old_key = os.environ.pop("OPENAI_API_KEY", None)
            old_cwd = os.getcwd()

            try:
                os.chdir(tmpdir)
                lc._load_env()

                assert os.environ.get("OPENAI_API_KEY") == "sk-test-123"
                assert os.environ.get("ANOTHER_KEY") == "value"
            finally:
                os.chdir(old_cwd)
                os.environ.pop("OPENAI_API_KEY", None)
                os.environ.pop("ANOTHER_KEY", None)
                lc._client = None
                if old_key:
                    os.environ["OPENAI_API_KEY"] = old_key