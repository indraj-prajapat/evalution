"""Tests for the extraction pipeline (scoring-based v4.0)."""

import json
from pathlib import Path

import pytest

from Evalution.tender_extractor.adapter import adapt_tender_plan, adapt_company_json
from Evalution.tender_extractor.config import ExtractionConfig
from Evalution.tender_extractor.models import (
    BidDocument,
    ExtractionResult,
    FieldSpec,
    RequirementDocument,
    TenderInfoPlan,
)


FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Model validation tests
# ---------------------------------------------------------------------------

class TestModels:
    """Test that models accept both 'fields' and 'required_fields' formats."""

    def test_tender_plan_with_fields_key(self):
        """User's format: 'fields' key at document level."""
        plan = TenderInfoPlan.model_validate({
            "criterion_id": "CRIT001",
            "criterion": "Test",
            "required_documents": [
                {
                    "document": "Work Order",
                    "mode": "EXPLICIT",
                    "category": "Experience",
                    "priority": 1,
                    "required": True,
                    "fields": [
                        {"name": "contract_value", "datatype": "currency",
                         "description": "Value", "required": True,
                         "examples": ["Rs. 1,00,000"]},
                    ],
                }
            ],
        })
        assert len(plan.required_documents) == 1
        assert plan.required_documents[0].document == "Work Order"

    def test_field_spec_defaults(self):
        fs = FieldSpec(name="test", datatype="string")
        assert fs.repeatable is False
        assert fs.required is True
        assert fs.description == ""
        assert fs.examples == []
        assert fs.group_by is None

    def test_extraction_result_to_json(self):
        result = ExtractionResult(documents=[])
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed == {"documents": []}

    def test_new_model_fields_have_defaults(self):
        """New v4.0 fields should have safe defaults."""
        from Evalution.tender_extractor.models import (
            MatchedDocument, ExtractedRecord, RequirementOutput,
            ScoredDocument, ExcludedDocument, EntityMention,
        )
        md = MatchedDocument(document_id="d1", document_name="Test")
        assert md.selection_score == 0.0
        assert md.rag_score == 0.0
        assert md.llm_score == 0.0

        er = ExtractedRecord(entity_id="e1")
        assert er.entity_mentions == []

        ro = RequirementOutput(
            requirement_document="Test", mode="EXPLICIT", entity_type="Other"
        )
        assert ro.excluded_documents == []

        sd = ScoredDocument(document_id="d1", document_name="Test")
        assert sd.rag_score == 0.0
        assert sd.llm_score == 0.0
        assert sd.combined_score == 0.0
        assert sd.excluded is False

        ed = ExcludedDocument(
            document_id="d1", document_name="Test",
            selection_score=0.0, exclusion_reason="test",
        )
        assert ed.rag_score == 0.0
        assert ed.llm_score == 0.0

        em = EntityMention(document_id="d1", document_name="Test")
        assert em.context_snippet == ""


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------

class TestAdapter:
    """Test tender plan and company JSON adaptation."""

    def test_adapt_tender_plan_fields_key(self):
        """Adapter handles 'fields' key from user's format."""
        plan = {
            "criterion_id": "CRIT001",
            "criterion": "Test criterion",
            "required_documents": [
                {
                    "document": "Work Order",
                    "mode": "EXPLICIT",
                    "category": "Experience Documents",
                    "priority": 1,
                    "required": True,
                    "fields": [
                        {"name": "contract_value", "datatype": "currency",
                         "description": "Contract amount"},
                    ],
                }
            ],
        }
        result = adapt_tender_plan(plan)
        assert result.criterion_id == "CRIT001"
        assert len(result.required_documents) == 1
        rd = result.required_documents[0]
        assert rd.document == "Work Order"
        assert rd.category == "Experience Documents"
        assert rd.entity_type == "Experience"  # auto-inferred

    def test_adapt_tender_plan_required_fields_key(self):
        """Adapter also handles old 'required_fields' key."""
        plan = {
            "required_documents": [
                {
                    "document": "GST Registration",
                    "mode": "EXPLICIT",
                    "required_fields": [
                        {"name": "gstin", "datatype": "string"},
                    ],
                }
            ],
        }
        result = adapt_tender_plan(plan)
        rd = result.required_documents[0]
        assert len(rd.required_fields) == 1
        assert rd.required_fields[0].name == "gstin"

    def test_adapt_company_json(self):
        """Adapter converts company JSON to BidDocument list."""
        with open(FIXTURES / "sample_bid.json") as f:
            company_data = json.load(f)

        result = adapt_company_json(company_data)
        assert len(result) >= 1
        assert isinstance(result[0], BidDocument)
        assert len(result[0].detected_documents) >= 1


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_config(self):
        cfg = ExtractionConfig()
        assert cfg.use_llm_extraction is True
        assert cfg.use_llm_doc_scoring is True
        assert cfg.llm_max_doc_chars == 6000
        assert cfg.llm_max_retries == 1
        # New v4.0 defaults
        assert cfg.rag_score_weight == 0.4
        assert cfg.llm_score_weight == 0.6
        assert cfg.doc_exclusion_threshold == 0.05
        assert cfg.regex_safety_net is True
        assert cfg.cross_document_mentions is True
        assert cfg.skip_company_match is True  # Now default: annotation only

    def test_regex_only_config(self):
        cfg = ExtractionConfig(
            use_llm_extraction=False,
            use_llm_doc_scoring=False,
        )
        assert cfg.use_llm_extraction is False
        assert cfg.use_llm_doc_scoring is False


# ---------------------------------------------------------------------------
# Integration test (regex mode — no API key needed)
# ---------------------------------------------------------------------------

class TestRegexExtraction:
    """Test the full pipeline in regex-only mode (no LLM/API key needed)."""

    def _load_fixtures(self):
        with open(FIXTURES / "sample_bid.json") as f:
            company_data = json.load(f)
        with open(FIXTURES / "sample_tender_plan.json") as f:
            tender_plan = json.load(f)
        return company_data, tender_plan

    def test_full_pipeline_regex_mode(self):
        """End-to-end test with regex extraction (no LLM)."""
        company_data, tender_plan = self._load_fixtures()

        from Evalution.tender_extractor.api import extract_from_bid

        result = extract_from_bid(
            company_json=company_data,
            tender_plan=tender_plan,
            use_llm=False,
            skip_company_match=True,
        )

        assert "criterion_id" in result
        assert "documents" in result
        assert len(result["documents"]) > 0

        # Check that at least some documents matched
        matched_any = False
        for doc_out in result["documents"]:
            for md in doc_out.get("matched_documents", []):
                if md.get("status") == "FOUND" and len(md.get("records", [])) > 0:
                    matched_any = True
                    # Verify records have proper structure
                    for rec in md["records"]:
                        assert "entity_id" in rec
                        assert "fields" in rec
                        for field in rec["fields"]:
                            assert "name" in field
                            assert "value" in field
                            assert "datatype" in field
                            assert "page" in field
                            assert "snippet" in field
                            assert "confidence" in field
                            assert 0.5 <= field["confidence"] <= 1.0
                    # v4.0: check new optional fields exist
                    assert "selection_score" in md
                    assert "rag_score" in md
                    assert "llm_score" in md
                    # v4.0: check entity_mentions exists on records
                    assert "entity_mentions" in rec

        assert matched_any, "Expected at least one document with extracted fields"

    def test_output_is_valid_json(self):
        """Verify output serialises to valid JSON."""
        company_data, tender_plan = self._load_fixtures()

        from Evalution.tender_extractor.api import extract_from_bid

        result = extract_from_bid(
            company_json=company_data,
            tender_plan=tender_plan,
            use_llm=False,
            skip_company_match=True,
        )

        json_str = json.dumps(result, ensure_ascii=False)
        assert len(json_str) > 100

    def test_excluded_documents_in_output(self):
        """v4.0: excluded_documents should appear in output."""
        company_data, tender_plan = self._load_fixtures()

        from Evalution.tender_extractor.api import extract_from_bid

        result = extract_from_bid(
            company_json=company_data,
            tender_plan=tender_plan,
            use_llm=False,
            skip_company_match=True,
        )

        # Check that excluded_documents field exists
        for doc_out in result["documents"]:
            assert "excluded_documents" in doc_out
            assert isinstance(doc_out["excluded_documents"], list)


# ---------------------------------------------------------------------------
# Document Scorer tests
# ---------------------------------------------------------------------------

class TestDocumentScorer:
    """Test the new scoring-based document selection."""

    def test_scorer_without_llm(self):
        """Scorer should work without LLM (gives default scores)."""
        from Evalution.tender_extractor.document_scorer import DocumentScorer
        from Evalution.tender_extractor.models import DetectedDocument, PageSummary, RequirementDocument

        docs = [
            DetectedDocument(
                document_id="d1",
                document_name="GST Registration Certificate",
                document_category="Tax",
                pages=[PageSummary(page_number=1, text="GST registration text")],
            ),
            DetectedDocument(
                document_id="d2",
                document_name="Random Doc",
                document_category="Other",
                pages=[PageSummary(page_number=1, text="Some random text")],
            ),
        ]

        req = RequirementDocument(document="GST Certificate")

        config = ExtractionConfig(use_llm_doc_scoring=False)
        scorer = DocumentScorer(config)
        scored = scorer.score_documents(req, docs)

        assert len(scored) == 2
        # Without LLM, all docs get default 0.3 score → none excluded
        assert all(not s.excluded for s in scored)

    def test_scorer_rag_scores(self):
        """RAG scores should influence combined scoring."""
        from Evalution.tender_extractor.document_scorer import DocumentScorer
        from Evalution.tender_extractor.models import DetectedDocument, PageSummary, RequirementDocument

        docs = [
            DetectedDocument(
                document_id="d1",
                document_name="Balance Sheet FY 2023-24",
                document_category="Financial",
                pages=[PageSummary(page_number=1, text="Balance sheet with turnover")],
            ),
            DetectedDocument(
                document_id="d2",
                document_name="Medical Certificate",
                pages=[PageSummary(page_number=1, text="Doctor fitness certificate")],
            ),
        ]

        req = RequirementDocument(document="Financial Documents")

        config = ExtractionConfig(
            use_llm_doc_scoring=False,
            rag_score_weight=1.0,
            llm_score_weight=0.0,
        )
        scorer = DocumentScorer(config)
        scored = scorer.score_documents(
            req, docs,
            rag_doc_scores={"d1": 0.8, "d2": 0.0},
        )

        # d1 should have high RAG score, d2 should have zero
        d1_score = next(s for s in scored if s.document_id == "d1")
        d2_score = next(s for s in scored if s.document_id == "d2")

        assert d1_score.rag_score == 0.8
        assert d2_score.rag_score == 0.0
        assert d1_score.combined_score > d2_score.combined_score

    def test_low_threshold_excludes_only_very_low(self):
        """With threshold 0.05, only truly zero-scored docs are excluded."""
        from Evalution.tender_extractor.document_scorer import DocumentScorer
        from Evalution.tender_extractor.models import DetectedDocument, PageSummary, RequirementDocument

        docs = [
            DetectedDocument(
                document_id="d1",
                document_name="Test Doc",
                pages=[PageSummary(page_number=1, text="test")],
            ),
        ]

        req = RequirementDocument(document="Something")

        config = ExtractionConfig(
            use_llm_doc_scoring=False,
            doc_exclusion_threshold=0.05,
        )
        scorer = DocumentScorer(config)
        scored = scorer.score_documents(req, docs)

        # Default LLM score is 0.3, RAG is 0.0 → combined = 0.6*0.3 = 0.18
        # 0.18 > 0.05, so NOT excluded
        assert len(scored) == 1
        assert not scored[0].excluded


# ---------------------------------------------------------------------------
# Cross-document mention tests
# ---------------------------------------------------------------------------

class TestCrossDocumentMentions:
    """Test cross-document entity mention scanning."""

    def test_mentions_found_across_docs(self):
        """Entity names found in one doc should be traced in other docs."""
        from Evalution.tender_extractor.evidence import EvidenceTracker
        from Evalution.tender_extractor.models import (
            DetectedDocument, PageSummary, ExtractedRecord,
            ExtractedField, MatchedDocument,
        )

        docs = [
            DetectedDocument(
                document_id="partner_list",
                document_name="Partner List",
                pages=[PageSummary(
                    page_number=1,
                    text="Partners: 1. Rahul Sharma 2. Priya Patel"
                )],
            ),
            DetectedDocument(
                document_id="cisa_cert",
                document_name="CISA Certificate",
                pages=[PageSummary(
                    page_number=1,
                    text="CISA Certificate holder: Rahul Sharma, valid till 2025"
                )],
            ),
        ]

        # Simulate extraction result: CISA cert extracted with holder name
        record = ExtractedRecord(
            entity_id="cisa_cert",
            fields=[
                ExtractedField(
                    name="certificate_holder",
                    value="Rahul Sharma",
                    datatype="string",
                    page=1,
                    snippet="CISA Certificate holder: Rahul Sharma",
                    confidence=0.95,
                    raw_value="Rahul Sharma",
                ),
            ],
        )

        matched = MatchedDocument(
            document_id="cisa_cert",
            document_name="CISA Certificate",
            status="FOUND",
            records=[record],
        )

        tracker = EvidenceTracker()
        tracker.scan_cross_document_mentions([matched], docs)

        # The partner_list doc should be mentioned
        assert len(record.entity_mentions) == 1
        assert record.entity_mentions[0]["document_id"] == "partner_list"
        assert record.entity_mentions[0]["document_name"] == "Partner List"


class TestRAGContentSearch:
    """Tests for the enhanced RAG pipeline with content-based matching."""

    def test_rule_search_content_keywords(self):
        """Rule search should match pages by content keywords, not just doc name."""
        from Evalution.tender_extractor.retriever import RetrievalIndex, PageHit
        from Evalution.tender_extractor.models import BidDocument, DetectedDocument, PageSummary, RequirementDocument

        doc = DetectedDocument(
            document_id="doc_1",
            document_name="Some Random File",
            document_category="Other",
            pages=[
                PageSummary(
                    page_number=1,
                    text="This is to certify that the branch office registered "
                         "at Maharashtra is valid. The registered branch address "
                         "is 123, MG Road, Pune, Maharashtra 411001.",
                )
            ],
        )
        bid_doc = BidDocument(pdf_id="test.pdf", detected_documents=[doc])
        index = RetrievalIndex([bid_doc])
        index.build()

        req = RequirementDocument(
            document="Branch/Office Registered Certificate",
            category="Registration Documents",
            expected_documents=["Branch/Office Registered Certificate"],
        )

        hits = index._rule_search(req)
        assert len(hits) > 0
        assert hits[0].source == "rule"

    def test_document_level_scoring(self):
        """RAG should provide document-level scores."""
        from Evalution.tender_extractor.retriever import RetrievalIndex
        from Evalution.tender_extractor.models import BidDocument, DetectedDocument, PageSummary, RequirementDocument

        docs = [
            DetectedDocument(
                document_id="d1",
                document_name="Balance Sheet",
                document_category="Financial",
                pages=[PageSummary(page_number=1, text="Balance sheet with turnover Rs. 1,00,000")],
            ),
            DetectedDocument(
                document_id="d2",
                document_name="Random Doc",
                pages=[PageSummary(page_number=1, text="Some completely unrelated text")],
            ),
        ]

        bid_doc = BidDocument(pdf_id="test.pdf", detected_documents=docs)
        index = RetrievalIndex([bid_doc])
        index.build()

        req = RequirementDocument(
            document="Balance Sheet",
            category="Financial Documents",
        )

        scores = index.score_documents(req)
        # d1 should be scored, d2 might or might not be
        assert "d1" in scores
        assert scores["d1"] > 0

    def test_rag_pipeline_with_mismatched_name(self):
        """Full RAG pipeline should find document even when name doesn't match."""
        from Evalution.tender_extractor.retriever import RetrievalIndex
        from Evalution.tender_extractor.models import BidDocument, DetectedDocument, PageSummary, RequirementDocument

        doc = DetectedDocument(
            document_id="doc_1",
            document_name="Annexure G",
            document_category="Attachment",
            pages=[
                PageSummary(
                    page_number=1,
                    text="CERTIFICATE OF REGISTRATION OF BRANCH OFFICE\n\n"
                         "This is to certify that M/s ABC Associates has a "
                         "registered branch office at Mumbai, Maharashtra.\n"
                         "Registration No: BR/MH/2024/5678\n"
                         "Date of Registration: 15th January 2024",
                )
            ],
        )
        bid_doc = BidDocument(pdf_id="test.pdf", detected_documents=[doc])
        index = RetrievalIndex([bid_doc])
        index.build()

        req = RequirementDocument(
            document="Branch/Office Registered Certificate",
            category="Registration Documents",
            expected_documents=["Branch/Office Registered Certificate"],
            required_fields=[],
        )

        result = index.retrieve(req, top_k=5)
        assert len(result.hits) > 0