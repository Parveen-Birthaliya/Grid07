
import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from .models import RoutingResult, ConfidenceLevel
from .config import SIMILARITY_THRESHOLD, FALLBACK_TO_BEST,EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class VectorRouter:
    """Routes posts to bot personas via semantic matching."""
    
    def __init__(self, embedding_model = EMBEDDING_MODEL):
        self.embedder = SentenceTransformer(embedding_model)


        self.bot_personas = {
            "bot_A": "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns.",

            "bot_B": "I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI, social media, and billionaires. I value privacy and nature.",

            "bot_C": "I strictly care about markets, interest rates, trading algorithms, and making money. I speak in finance jargon and view everything through the lens of ROI."
        }

        self.persona_texts = list(self.bot_personas.values())
        self.persona_keys = list(self.bot_personas.keys())

        self.persona_embedding = self.embedder.encode(self.persona_texts, normalize_embeddings=True).astype("float32")
        
        embedding_dim = self.persona_embedding.shape[1]

        self.index = faiss.IndexFlatIP(embedding_dim)

        self.index.add(self.persona_embedding)
        logger.info("VectorRouter initialized with %d personas", len(self.persona_keys))


    def _get_confidence(self, score: float) -> ConfidenceLevel:
        if score > 0.8:
            return ConfidenceLevel.HIGH
        elif score > 0.65:
            return ConfidenceLevel.MEDIUM
        else: 
            return ConfidenceLevel.LOW

           
    
    def route_post_to_bots(self, post: str, threshold = SIMILARITY_THRESHOLD) -> List[RoutingResult]:
        query_vec = self.embedder.encode(post, normalize_embeddings=True).astype("float32")

        scores, indices = self.index.search(np.array([query_vec]), k=len(self.persona_keys))

        results = []
    
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold:
                results.append((self.persona_keys[idx], float(score)))

        # fallback logic
        if not results and FALLBACK_TO_BEST:
            best_idx = indices[0][0]
            best_score = scores[0][0]
            results = [(self.persona_keys[best_idx], float(best_score))]

        # pick highest if multiple
        if len(results) > 1:
            results = [max(results, key=lambda x: x[1])]

        # return structured output
        results = [
            RoutingResult(
                bot_id=bot_id,
                similarity_score=score,
                confidence=self._get_confidence(score),
                persona=self.bot_personas[bot_id]
            )
            for bot_id, score in results
        ]

        logger.info("Routing result: %s", results)

        return results

        