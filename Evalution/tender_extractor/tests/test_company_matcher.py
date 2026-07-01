"""Unit tests for the company matcher module."""

import pytest

from Evalution.tender_extractor.company_matcher import CompanyMatcher


class TestCompanyMatcher:
    """Company ownership verification tests."""

    def setup_method(self):
        self.matcher = CompanyMatcher(
            bidder_name="M/s ABC Pvt. Ltd.",
            bidder_pan="ABCDE1234F",
            bidder_gstin="27ABCDE1234F1Z5",
        )

    def test_pan_match(self):
        text = "Some company document with PAN: ABCDE1234F and other details."
        is_owner, reason = self.matcher.is_owned_by_bidder(text)
        assert is_owner is True
        assert "PAN" in reason

    def test_gstin_match(self):
        text = "GST Registration GSTIN: 27ABCDE1234F1Z5 for the company."
        is_owner, reason = self.matcher.is_owned_by_bidder(text)
        assert is_owner is True
        assert "GSTIN" in reason

    def test_name_match(self):
        text = "Name of the Company: M/s ABC Pvt. Ltd.\nThis is the company registration."
        is_owner, reason = self.matcher.is_owned_by_bidder(text)
        assert is_owner is True
        assert "name" in reason.lower()

    def test_no_match(self):
        text = "XYZ Corporation is a different company entirely."
        is_owner, reason = self.matcher.is_owned_by_bidder(text)
        assert is_owner is False

    def test_metadata_pan(self):
        text = "Some text without PAN"
        metadata = {"pan": "ABCDE1234F"}
        is_owner, _ = self.matcher.is_owned_by_bidder(text, doc_metadata=metadata)
        assert is_owner is True

    def test_experience_document_skip(self):
        assert self.matcher.is_experience_document("Experience", "Work Order") is True
        assert self.matcher.is_experience_document("Experience", "Completion Certificate") is True
        assert self.matcher.is_experience_document("Financial", "Balance Sheet") is False

    def test_substring_match(self):
        text = "ABC Pvt. Ltd. has been operating since 2020."
        is_owner, _ = self.matcher.is_owned_by_bidder(text)
        assert is_owner is True