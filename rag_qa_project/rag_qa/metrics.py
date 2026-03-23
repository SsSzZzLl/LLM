from __future__ import annotations

import re
import string
from typing import Optional


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\[[\d, ]+\]", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def token_f1(pred: str, gold: str) -> float:
    p = set(normalize_answer(pred).split())
    g = set(normalize_answer(gold).split())
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    inter = p & g
    if not inter:
        return 0.0
    precision = len(inter) / len(p)
    recall = len(inter) / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def contains_answer(pred: str, gold: str) -> bool:
    """Loose check: all gold tokens appear in pred (good for short spans)."""
    g = normalize_answer(gold).split()
    if not g:
        return True
    pn = normalize_answer(pred)
    return all(t in pn for t in g)
