"""
Test suite for VoteGuide AI
Tests: endpoints, data integrity, prompt structure, input validation
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

from main import app, COUNTRY_CONTEXTS, SYSTEM_PROMPT, __version__, GEMINI_MODEL, GROQ_MODEL

client = TestClient(app)


# ── Health & Homepage ──────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self):
        """Health endpoint must return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_ok_status(self):
        """Health response must have status: ok."""
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_returns_correct_service_name(self):
        """Service name must be VoteGuide AI."""
        data = client.get("/health").json()
        assert data["service"] == "VoteGuide AI"

    def test_returns_version(self):
        """Health response must include version."""
        data = client.get("/health").json()
        assert data["version"] == __version__

    def test_returns_primary_model(self):
        """Health response must declare primary model."""
        data = client.get("/health").json()
        assert data["primary_model"] == GEMINI_MODEL

    def test_returns_fallback_model(self):
        """Health response must declare fallback model."""
        data = client.get("/health").json()
        assert data["fallback_model"] == GROQ_MODEL

    def test_returns_supported_countries(self):
        """Health response must list supported countries."""
        data = client.get("/health").json()
        assert "India" in data["supported_countries"]
        assert "USA" in data["supported_countries"]
        assert "UK" in data["supported_countries"]


class TestHomepage:
    def test_returns_200(self):
        """Homepage must return HTTP 200."""
        assert client.get("/").status_code == 200

    def test_contains_brand_name(self):
        """Homepage must contain VoteGuide branding."""
        assert "VoteGuide" in client.get("/").text

    def test_contains_all_countries(self):
        """Homepage must list all three supported countries."""
        html = client.get("/").text
        for country in ["India", "USA", "UK"]:
            assert country in html

    def test_content_type_is_html(self):
        """Homepage must return HTML content type."""
        assert "text/html" in client.get("/").headers["content-type"]


# ── Data Integrity ─────────────────────────────────────────────────

class TestCountryContexts:
    REQUIRED_FIELDS = ["body", "voting_age", "register_url",
                       "id_required", "election_types", "key_dates_process"]

    def test_all_countries_present(self):
        """All three countries must be in COUNTRY_CONTEXTS."""
        for country in ["India", "USA", "UK"]:
            assert country in COUNTRY_CONTEXTS

    @pytest.mark.parametrize("country", ["India", "USA", "UK"])
    def test_country_has_all_fields(self, country):
        """Each country must have all required fields."""
        for field in self.REQUIRED_FIELDS:
            assert field in COUNTRY_CONTEXTS[country], f"{country} missing: {field}"

    def test_voting_age_is_18_for_all(self):
        """Voting age must be 18 for all countries."""
        for country, ctx in COUNTRY_CONTEXTS.items():
            assert ctx["voting_age"] == 18, f"{country} voting age incorrect"

    def test_india_register_url(self):
        """India must point to official ECI registration portal."""
        assert "eci.gov.in" in COUNTRY_CONTEXTS["India"]["register_url"]

    def test_usa_register_url(self):
        """USA must point to official vote.gov portal."""
        assert "vote.gov" in COUNTRY_CONTEXTS["USA"]["register_url"]

    def test_uk_register_url(self):
        """UK must point to official gov.uk portal."""
        assert "gov.uk" in COUNTRY_CONTEXTS["UK"]["register_url"]

    def test_register_urls_are_https(self):
        """All registration URLs must use HTTPS."""
        for country, ctx in COUNTRY_CONTEXTS.items():
            assert ctx["register_url"].startswith("https://"), \
                f"{country} URL not HTTPS"


# ── Prompt Quality ─────────────────────────────────────────────────

class TestSystemPrompt:
    def test_contains_all_10_rules(self):
        """System prompt must contain all 10 numbered rules."""
        for i in range(1, 11):
            assert f"{i}." in SYSTEM_PROMPT, f"Rule {i} missing"

    def test_formats_without_error_for_all_countries(self):
        """System prompt must format cleanly for all countries."""
        for country, ctx in COUNTRY_CONTEXTS.items():
            formatted = SYSTEM_PROMPT.format(country=country, **ctx)
            assert country in formatted
            assert ctx["body"] in formatted
            assert ctx["register_url"] in formatted

    def test_contains_few_shot_example(self):
        """System prompt must contain a worked example answer."""
        assert "PERFECT EXAMPLE ANSWER" in SYSTEM_PROMPT

    def test_contains_source_rule(self):
        """Prompt must instruct model to cite official sources."""
        assert "SOURCE" in SYSTEM_PROMPT or "According to" in SYSTEM_PROMPT

    def test_contains_neutrality_rule(self):
        """Prompt must enforce political neutrality."""
        assert "NEUTRAL" in SYSTEM_PROMPT


# ── Input Validation ───────────────────────────────────────────────

class TestInputValidation:
    def test_rejects_missing_country(self):
        """Must reject request with no country field."""
        response = client.post("/chat/stream", json={"message": "hello"})
        assert response.status_code == 422

    def test_rejects_empty_message(self):
        """Must reject empty message."""
        response = client.post("/chat/stream",
                               json={"country": "India", "message": ""})
        assert response.status_code == 422

    def test_rejects_message_over_1000_chars(self):
        """Must reject messages exceeding 1000 characters."""
        response = client.post("/chat/stream",
                               json={"country": "India", "message": "x" * 1001})
        assert response.status_code == 422

    def test_invalid_country_falls_back_to_india(self):
        """Unknown country must fall back to India silently."""
        with patch("main.genai.GenerativeModel") as mock_model:
            mock_chat = MagicMock()
            mock_response = MagicMock()
            mock_response.__iter__ = MagicMock(return_value=iter([]))
            mock_chat.send_message.return_value = mock_response
            mock_model.return_value.start_chat.return_value = mock_chat

            response = client.post("/chat/stream", json={
                "country": "Mars",
                "message": "How do I vote?",
                "history": []
            })
            assert response.status_code == 200

    def test_accepts_valid_request(self):
        """Must accept a well-formed request."""
        with patch("main.genai.GenerativeModel") as mock_model:
            mock_chat = MagicMock()
            mock_response = MagicMock()
            mock_response.__iter__ = MagicMock(return_value=iter([]))
            mock_chat.send_message.return_value = mock_response
            mock_model.return_value.start_chat.return_value = mock_chat

            response = client.post("/chat/stream", json={
                "country": "India",
                "message": "How do I register to vote?",
                "history": []
            })
            assert response.status_code == 200

    def test_accepts_all_valid_countries(self):
        """Must accept requests for all supported countries."""
        with patch("main.genai.GenerativeModel") as mock_model:
            mock_chat = MagicMock()
            mock_response = MagicMock()
            mock_response.__iter__ = MagicMock(return_value=iter([]))
            mock_chat.send_message.return_value = mock_response
            mock_model.return_value.start_chat.return_value = mock_chat

            for country in ["India", "USA", "UK"]:
                response = client.post("/chat/stream", json={
                    "country": country,
                    "message": "How does voting work?",
                    "history": []
                })
                assert response.status_code == 200, f"Failed for country: {country}"