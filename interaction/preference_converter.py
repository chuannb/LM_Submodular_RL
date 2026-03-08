"""
Preference Converter

Chuyển đổi interaction log (click / no-click / add-to-cart / next-page)
thành pairwise preference data cho:
  1. DPO / ORPO training  →  {"prompt", "chosen", "rejected"}
  2. Reranker fine-tuning →  {"query", "positive", "negative"}

Thứ tự ưu tiên signal (cao hơn = preferred):
  purchase    > add_to_cart > click > next_page > no_click

Preference pair generation strategy:
  - Với mỗi impression, lấy tất cả items đã tương tác
  - Pair (high_signal_item, low_signal_item) = (chosen, rejected)
  - Nếu không có explicit positive: items ở vị trí cao hơn mà user
    click (position-based preference)
  - next_page signal: tất cả items trên trang đó → implicit negatives
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from interaction.logger import EVENT_WEIGHT, EventType, InteractionLogger


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RankerPreferencePair:
    """Training sample for reranker fine-tuning."""
    query: str
    positive_id: str       # item_id preferred
    positive_text: str
    negative_id: str       # item_id rejected
    negative_text: str
    weight: float = 1.0    # sample weight (e.g. purchase > click)


@dataclass
class DPOSample:
    """Training sample for DPO / ORPO."""
    prompt: str            # system + query context
    chosen: str            # text of preferred item (title + description)
    rejected: str          # text of rejected item
    chosen_id: str
    rejected_id: str


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------

class PreferenceConverter:
    """
    Convert raw interaction logs into preference pairs.

    Parameters
    ----------
    logger       : InteractionLogger instance to query the DB
    item_texts   : {item_id: text} mapping for building training text
    item_titles  : {item_id: title} mapping
    min_weight_gap : minimum difference in event weights to form a pair
    """

    def __init__(
        self,
        logger: InteractionLogger,
        item_texts: Dict[str, str],
        item_titles: Dict[str, str],
        min_weight_gap: float = 0.2,
    ):
        self.logger = logger
        self.item_texts = item_texts
        self.item_titles = item_titles
        self.min_weight_gap = min_weight_gap

    # ------------------------------------------------------------------
    def _get_item_signal(self, impression_id: str) -> Dict[str, float]:
        """
        Compute max event weight per item for a given impression.
        Returns {item_id: max_weight}.
        """
        interactions = self.logger.get_interactions(impression_id)
        signal: Dict[str, float] = {}
        for inter in interactions:
            evt = EventType(inter["event_type"])
            w = EVENT_WEIGHT.get(evt, 0.0)
            item_id = inter["item_id"]
            signal[item_id] = max(signal.get(item_id, 0.0), w)
        return signal

    # ------------------------------------------------------------------
    def extract_ranker_pairs(
        self,
        max_pairs_per_impression: int = 5,
    ) -> List[RankerPreferencePair]:
        """
        Extract (positive, negative) pairs for reranker training.

        Strategy:
          - For each impression, rank items by their event weight.
          - Pair high-weight items against low-weight items.
          - Gap must be >= min_weight_gap to filter noisy pairs.
        """
        pairs: List[RankerPreferencePair] = []

        impressions = self.logger.get_all_impressions()
        for imp in impressions:
            query = imp["query"]
            signal = self._get_item_signal(imp["impression_id"])

            # Items with known signal, sorted by weight descending
            ranked = sorted(signal.items(), key=lambda x: x[1], reverse=True)

            # Items without signal (shown but not interacted) are implicit negatives
            for shown_id in imp["shown_items"]:
                if shown_id not in signal:
                    signal[shown_id] = 0.0

            ranked = sorted(signal.items(), key=lambda x: x[1], reverse=True)

            added = 0
            for i, (pos_id, pos_w) in enumerate(ranked):
                for neg_id, neg_w in ranked[i + 1:]:
                    if pos_w - neg_w < self.min_weight_gap:
                        continue
                    pos_text = self.item_texts.get(pos_id, "")
                    neg_text = self.item_texts.get(neg_id, "")
                    if not pos_text or not neg_text:
                        continue
                    pairs.append(RankerPreferencePair(
                        query=query,
                        positive_id=pos_id,
                        positive_text=pos_text,
                        negative_id=neg_id,
                        negative_text=neg_text,
                        weight=pos_w - neg_w,
                    ))
                    added += 1
                    if added >= max_pairs_per_impression:
                        break
                if added >= max_pairs_per_impression:
                    break

        return pairs

    # ------------------------------------------------------------------
    def extract_dpo_samples(
        self,
        system_prompt: str = "You are a product search assistant. Given the query, rank the most relevant product first.",
        max_pairs_per_impression: int = 3,
    ) -> List[DPOSample]:
        """
        Extract DPO / ORPO training samples.

        Format:
          prompt   = system + "Query: <query>"
          chosen   = "Title: <title>\n<description>"  (higher signal)
          rejected = "Title: <title>\n<description>"  (lower signal)
        """
        samples: List[DPOSample] = []

        impressions = self.logger.get_all_impressions()
        for imp in impressions:
            query = imp["query"]
            signal = self._get_item_signal(imp["impression_id"])

            for shown_id in imp["shown_items"]:
                if shown_id not in signal:
                    signal[shown_id] = 0.0

            ranked = sorted(signal.items(), key=lambda x: x[1], reverse=True)

            added = 0
            for i, (pos_id, pos_w) in enumerate(ranked):
                for neg_id, neg_w in ranked[i + 1:]:
                    if pos_w - neg_w < self.min_weight_gap:
                        continue
                    pos_title = self.item_titles.get(pos_id, "")
                    neg_title = self.item_titles.get(neg_id, "")
                    pos_text = self.item_texts.get(pos_id, "")
                    neg_text = self.item_texts.get(neg_id, "")
                    if not pos_text or not neg_text:
                        continue

                    prompt = (
                        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                        f"<|im_start|>user\nQuery: {query}<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                    )
                    chosen = f"Title: {pos_title}\n{pos_text}"
                    rejected = f"Title: {neg_title}\n{neg_text}"

                    samples.append(DPOSample(
                        prompt=prompt,
                        chosen=chosen,
                        rejected=rejected,
                        chosen_id=pos_id,
                        rejected_id=neg_id,
                    ))
                    added += 1
                    if added >= max_pairs_per_impression:
                        break
                if added >= max_pairs_per_impression:
                    break

        return samples

    # ------------------------------------------------------------------
    def to_jsonl(
        self,
        samples: List,
        path: str,
    ) -> None:
        """Dump DPOSample or RankerPreferencePair list to JSONL."""
        import dataclasses
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(dataclasses.asdict(s), ensure_ascii=False) + "\n")
        print(f"Saved {len(samples)} samples -> {path}")

    # ------------------------------------------------------------------
    @staticmethod
    def load_jsonl(path: str) -> List[dict]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
