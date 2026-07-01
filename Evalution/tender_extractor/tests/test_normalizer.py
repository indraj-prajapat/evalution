"""Unit tests for the normalizer module."""

import pytest

from Evalution.tender_extractor.normalizer import (
    normalise_date,
    normalise_currency,
    normalise_boolean,
    normalise_value,
)


class TestNormaliseDate:
    """Date normalisation tests."""

    def test_dd_mm_yyyy_slash(self):
        assert normalise_date("15/03/2020") == "2020-03-15"

    def test_dd_mm_yyyy_dash(self):
        assert normalise_date("15-03-2020") == "2020-03-15"

    def test_dd_month_yyyy(self):
        assert normalise_date("15 March 2020") == "2020-03-15"

    def test_month_dd_yyyy(self):
        assert normalise_date("March 15, 2020") == "2020-03-15"

    def test_yyyy_mm_dd(self):
        assert normalise_date("2020-03-15") == "2020-03-15"

    def test_already_iso(self):
        assert normalise_date("2023-04-20") == "2023-04-20"

    def test_empty(self):
        assert normalise_date("") is None

    def test_invalid(self):
        assert normalise_date("not a date") is None

    def test_february_29(self):
        assert normalise_date("29/02/2020") == "2020-02-29"


class TestNormaliseCurrency:
    """Currency normalisation tests."""

    def test_indian_format(self):
        assert normalise_currency("₹4,56,78,900") == 45678900

    def test_rs_prefix(self):
        assert normalise_currency("Rs. 1,23,456.78") == 123456

    def test_crore(self):
        assert normalise_currency("2.5 Crore") == 25000000

    def test_lakh(self):
        assert normalise_currency("15 Lakh") == 1500000

    def test_lac(self):
        assert normalise_currency("15 Lac") == 1500000

    def test_plain_number(self):
        assert normalise_currency("1234567") == 1234567

    def test_with_inr(self):
        assert normalise_currency("INR 50,00,000") == 5000000

    def test_empty(self):
        assert normalise_currency("") is None

    def test_no_number(self):
        assert normalise_currency("no amount") is None

    def test_large_crore(self):
        assert normalise_currency("Rs. 2,45,67,89,000") == 2456789000


class TestNormaliseBoolean:
    """Boolean normalisation tests."""

    def test_yes(self):
        assert normalise_boolean("Yes") is True

    def test_true(self):
        assert normalise_boolean("True") is True

    def test_no(self):
        assert normalise_boolean("No") is False

    def test_false(self):
        assert normalise_boolean("False") is False

    def test_affirmative(self):
        assert normalise_boolean("Affirmative") is True

    def test_negative(self):
        assert normalise_boolean("Not Available") is False

    def test_empty(self):
        assert normalise_boolean("") is None

    def test_unknown(self):
        assert normalise_boolean("something") is None


class TestNormaliseValue:
    """Generic normalisation dispatcher tests."""

    def test_currency_dispatch(self):
        value, _ = normalise_value("₹4,56,78,900", "currency")
        assert value == 45678900

    def test_date_dispatch(self):
        value, _ = normalise_value("15/03/2020", "date")
        assert value == "2020-03-15"

    def test_boolean_dispatch(self):
        value, _ = normalise_value("Yes", "boolean")
        assert value is True

    def test_integer_dispatch(self):
        value, _ = normalise_value("Total: 42 items", "integer")
        assert value == 42

    def test_string_dispatch(self):
        value, _ = normalise_value("  Hello World  ", "string")
        assert value == "Hello World"

    def test_empty_not_found(self):
        value, _ = normalise_value("", "string")
        assert value == "NOT_FOUND"