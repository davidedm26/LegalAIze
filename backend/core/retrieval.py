from typing import List, Dict, Any, Optional
import re
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class RetrievalEngine:
    def __init__(self, doc_client: QdrantClient, embedding_model: Optional[SentenceTransformer] = None):
        self.doc_client = doc_client
        self.embedding_model = embedding_model

    def _remove_law_names(self, text: str) -> str:
        """
        Removes common law names and references to focus search on semantic content.
        Strips regulatory identifiers while preserving the substantive content.
        """
        law_patterns = [
            # EU AI Act variants
            r"\bEU\s*AI\s*ACT\b",
            r"\bAI\s*Act\b",
            r"\bArtificial\s*Intelligence\s*Act\b",
            r"\bAI\s*Regulation\b",
            r"\bRegulation\s*\(EU\)\s*2024/1689\b",
            
            # ISO standards
            r"\bISO\s*/?IEC\s*42001\b",
            r"\bISO\s*42001\b",
            r"\bISO\s*/?IEC\s*27001\b",
            r"\bISO\s*/?IEC\s*27701\b",
            r"\bISO\s*/?IEC\b",
            r"\bISO\b",
            r"\bIEC\b",
            r"\b42001\b",
            r"\b27001\b",
            
            # EU regulations and directives
            r"\bGDPR\b",
            r"\bGeneral\s*Data\s*Protection\s*Regulation\b",
            r"\bRegulation\s*\(EU\)\b",
            r"\bDirective\s*\(EU\)\b",
            r"\bRegolamento\s*\(EU\)\b",
            
            # Annexes and articles
            r"\bAnnex\s*[IVX]+\b",
            r"\bArticle\s*\d+\b",
            r"\bArt\.\s*\d+\b",
            
            # Generic legal terms
            r"\b[Rr]egulation\b",
            r"\b[Dd]irective\b",
            r"\b[Ss]tandard\b",
            r"\b[Ll]egge\b",
            r"\bCode\s*of\s*Ethics\b",
            r"\bCodice\s*Etico\b",
        ]
        for pat in law_patterns:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)
        return text.strip()

    def query_for_requirement(self, collection_name: str, req_chunks_embeddings: List[Dict[str, Any]], top_k: int = 4) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieves relevant document chunks for each regulatory reference in the requirement.
        Returns a dictionary mapping regulatory group keys to lists of document chunks.
        """
        from collections import defaultdict

        group_to_chunks = defaultdict(list)
        for req in req_chunks_embeddings:
            reference = req.get("reference")
            source = req.get("source")
            assigned = False
            if reference and source:
                if source == "EU_AI_ACT":
                    group_to_chunks[f"AI_ACT::{reference}"].append(req)
                    assigned = True
                elif source == "ISO_42001":
                    group_to_chunks[f"ISO_42001::{reference}"].append(req)
                    assigned = True
            if not assigned:
                group_to_chunks['NO_GROUP'].append(req)

        results_by_group = {}

        for group, chunks in group_to_chunks.items():
            group_results = []
            for req in chunks:
                # 1. Clean query text
                reference_clean = self._remove_law_names(req.get('reference', ''))
                content_clean = self._remove_law_names(req.get('content', ''))

                # If both are empty, fallback to original reference
                query_text = reference_clean if reference_clean else req.get('reference', '')
                if content_clean:
                    query_text += " " + content_clean

                # 2. Re-embed query text
                if self.embedding_model:
                    req_emb = self.embedding_model.encode([query_text])[0]
                else:
                    # Fallback: use original embedding from regulatory chunk
                    req_emb = req['embedding']

                # 3. Query Qdrant
                hits = self.doc_client.query_points(
                    collection_name=collection_name,
                    query=req_emb,
                    limit=top_k
                ).points

                for hit in hits:
                    group_results.append({
                        'score': hit.score,
                        'content': hit.payload.get('content', ''),
                        'chunk_id': hit.payload.get('chunk_id', None)
                    })

            # 4. Deduplicate results for this group (keep highest score per chunk_id)
            seen = {}
            for r in group_results:
                cid = r['chunk_id']
                if cid not in seen or r['score'] > seen[cid]['score']:
                    seen[cid] = r

            # 5. Take top_k unique chunks for this group
            top_chunks = list(seen.values())
            top_chunks.sort(key=lambda x: x['score'], reverse=True)
            results_by_group[group] = top_chunks[:top_k]

        return results_by_group
