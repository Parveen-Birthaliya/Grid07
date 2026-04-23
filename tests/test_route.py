import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.routing import VectorRouter
from src.models import RoutingResult, ConfidenceLevel
from src.config import SIMILARITY_THRESHOLD, FALLBACK_TO_BEST


def make_mock_router(scores: list[float], indices: list[int] | None = None):
    """
    Factory that builds a VectorRouter with mocked heavy deps.
    """
    if indices is None:
        indices = list(range(len(scores)))

    with patch("src.routing.SentenceTransformer") as MockST, \
         patch("src.routing.faiss") as MockFaiss:

        fake_embed = np.ones((3, 384), dtype="float32")
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embed
        MockST.return_value = mock_model

        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([scores], dtype="float32"),
            np.array([indices])
        )
        MockFaiss.IndexFlatIP.return_value = mock_index

        router = VectorRouter()
        router.index = mock_index
        return router


@pytest.fixture(scope="module")
def real_router():
    """One real VectorRouter shared across the whole module (slow fixture)."""
    return VectorRouter()


class TestGetConfidence:
    def test_confidence_levels(self, score, expected):
        assert VectorRouter._get_confidence(score) == expected

    def test_returns_confidence_level_instance(self):
        result = VectorRouter._get_confidence(0.9)
        assert isinstance(result, ConfidenceLevel)

    def test_high_threshold_exclusive(self):
        """Score of exactly 0.8 should NOT be HIGH (strict >)."""
        assert VectorRouter._get_confidence(0.80) != ConfidenceLevel.HIGH

    def test_medium_threshold_exclusive(self):
        """Score of exactly 0.65 should NOT be MEDIUM (strict >)."""
        assert VectorRouter._get_confidence(0.65) != ConfidenceLevel.MEDIUM



class TestVectorRouterInit:

    def test_has_three_personas(self):
        router = make_mock_router([0.9, 0.8, 0.7])
        assert len(router.persona_keys) == 3

    def test_persona_keys_correct(self):
        router = make_mock_router([0.9, 0.8, 0.7])
        assert set(router.persona_keys) == {"bot_A", "bot_B", "bot_C"}

    def test_persona_texts_length_matches_keys(self):
        router = make_mock_router([0.9, 0.8, 0.7])
        assert len(router.persona_texts) == len(router.persona_keys)

    def test_bot_personas_dict_has_all_bots(self):
        router = make_mock_router([0.9, 0.8, 0.7])
        assert "bot_A" in router.bot_personas
        assert "bot_B" in router.bot_personas
        assert "bot_C" in router.bot_personas

    def test_persona_descriptions_are_non_empty(self):
        router = make_mock_router([0.9, 0.8, 0.7])
        for desc in router.bot_personas.values():
            assert isinstance(desc, str) and len(desc) > 0

class TestRoutePostToBots:

    def test_returns_a_list(self):
        router = make_mock_router([0.9, 0.7, 0.5])
        result = router.route_post_to_bots("test post")
        assert isinstance(result, list)

    def test_each_item_is_routing_result(self):
        router = make_mock_router([0.9, 0.7, 0.5])
        for item in router.route_post_to_bots("test post"):
            assert isinstance(item, RoutingResult)

    def test_result_has_at_most_one_item(self):
        """Router should always collapse to a single best match."""
        router = make_mock_router([0.9, 0.85, 0.8])
        result = router.route_post_to_bots("test post", threshold=0.0)
        assert len(result) == 1

    # RoutingResult field validation 
    def test_bot_id_is_valid_persona(self):
        router = make_mock_router([0.9, 0.7, 0.5])
        result = router.route_post_to_bots("test post")
        assert result[0].bot_id in {"bot_A", "bot_B", "bot_C"}

    def test_similarity_score_in_range(self):
        router = make_mock_router([0.9, 0.7, 0.5])
        result = router.route_post_to_bots("test post")
        score = result[0].similarity_score
        assert 0.0 < score <= 1.0

    def test_confidence_is_confidence_level_enum(self):
        router = make_mock_router([0.9, 0.7, 0.5])
        result = router.route_post_to_bots("test post")
        assert isinstance(result[0].confidence, ConfidenceLevel)

    def test_persona_field_is_non_empty_string(self):
        router = make_mock_router([0.9, 0.7, 0.5])
        result = router.route_post_to_bots("test post")
        assert isinstance(result[0].persona, str) and len(result[0].persona) > 0

    # Best-pick logic
    def test_highest_scoring_bot_is_returned(self):
        """
        FAISS returns [bot_B=0.88, bot_A=0.76, bot_C=0.70].
        All above threshold=0.0, so the router must pick bot_B (index 1).
        """
        router = make_mock_router(scores=[0.88, 0.76, 0.70], indices=[1, 0, 2])
        result = router.route_post_to_bots("test post", threshold=0.0)
        assert result[0].bot_id == "bot_B"
        assert result[0].similarity_score == pytest.approx(0.88, abs=1e-4)

    # Threshold filtering
    def test_results_respect_custom_threshold(self):
        """Only scores >= custom threshold should produce results."""
        # All scores below threshold=0.95 → must fall back to best
        router = make_mock_router([0.9, 0.7, 0.5])
        result = router.route_post_to_bots("test post", threshold=0.95)
        # FALLBACK_TO_BEST=True in config, so we still get 1 result
        assert len(result) == 1

# Edge cases & boundary conditions

class TestEdgeCases:

    def test_empty_string_post_does_not_raise(self):
        """Empty string is a valid (if unusual) query."""
        router = make_mock_router([0.9, 0.7, 0.5])
        result = router.route_post_to_bots("")
        assert isinstance(result, list)

    def test_very_long_post_does_not_raise(self):
        long_post = "AI " * 1000
        router = make_mock_router([0.9, 0.7, 0.5])
        result = router.route_post_to_bots(long_post)
        assert isinstance(result, list)

    def test_fallback_when_all_scores_below_threshold(self):
        """
        When NO score meets the threshold but FALLBACK_TO_BEST=True,
        the best result should still be returned.
        """
        router = make_mock_router([0.3, 0.2, 0.1])
        # threshold higher than all scores
        result = router.route_post_to_bots("some post", threshold=0.9)
        if FALLBACK_TO_BEST:
            assert len(result) == 1
        else:
            assert len(result) == 0

    def test_no_fallback_returns_empty_when_disabled(self):
        """With FALLBACK_TO_BEST patched to False and no score above threshold → []."""
        router = make_mock_router([0.3, 0.2, 0.1])
        with patch("src.routing.FALLBACK_TO_BEST", False):
            result = router.route_post_to_bots("some post", threshold=0.9)
            assert result == []

    def test_only_one_result_even_when_many_above_threshold(self):
        """Even if all 3 bots score above threshold, only 1 (the best) is returned."""
        router = make_mock_router([0.95, 0.92, 0.88])
        result = router.route_post_to_bots("some post", threshold=0.0)
        assert len(result) == 1

    def test_similarity_score_gt_zero_constraint_from_pydantic(self):
        """RoutingResult.similarity_score must be > 0.0 (Pydantic Field constraint)."""
        with pytest.raises(Exception):
            # score=0.0 violates gt=0.0 in RoutingResult
            RoutingResult(
                bot_id="bot_A",
                similarity_score=0.0,
                confidence=ConfidenceLevel.LOW,
                persona="some persona"
            )


# Integration tests
@pytest.mark.slow
class TestIntegration:
    """
    These tests load the real sentence-transformer model.
    Run with:  pytest tests/complete_test_route.py -m slow -v
    """

    def test_tech_optimism_routes_to_bot_a(self, real_router):
        post = "Elon Musk, Bitcoin, and AI will solve all of humanity's problems by 2030!"
        result = real_router.route_post_to_bots(post)
        assert len(result) == 1
        assert result[0].bot_id == "bot_A"

    def test_finance_post_routes_to_bot_c(self, real_router):
        post = "Interest rates are rising, hedge your portfolio with options and REITs for better ROI."
        result = real_router.route_post_to_bots(post)
        assert len(result) == 1
        assert result[0].bot_id == "bot_C"

    def test_anti_capitalist_post_routes_to_bot_b(self, real_router):
        post = "Big Tech monopolies are destroying privacy and society. We must regulate AI now."
        result = real_router.route_post_to_bots(post)
        assert len(result) == 1
        assert result[0].bot_id == "bot_B"

    def test_integration_result_is_valid_routing_result(self, real_router):
        post = "Crypto will go to the moon with Musk!"
        result = real_router.route_post_to_bots(post)
        assert len(result) >= 1
        r = result[0]
        assert isinstance(r, RoutingResult)
        assert 0.0 < r.similarity_score <= 1.0
        assert isinstance(r.confidence, ConfidenceLevel)
        assert r.bot_id in {"bot_A", "bot_B", "bot_C"}
