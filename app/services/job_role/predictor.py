from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable
from app.services.job_role.model_loader import (
    _load_encoder,
    _load_skill_map,
    _load_model,
    _load_tfidf,
)
import numpy as np


@dataclass(frozen=True)
class JobRolePrediction:
    label: str
    confidence: float | None


@dataclass(frozen=True)
class JobRoleRanking:
    predictions: list[JobRolePrediction]

    @property
    def best(self) -> JobRolePrediction:
        return self.predictions[0]

    def top(self, n: int = 3) -> list[JobRolePrediction]:
        return self.predictions[:n]


def _parse_skills(skillset: Iterable[str]) -> str:
    token = []
    exist_skill = set()

    for skill in skillset:
        if not skill:
            continue
        parsed_skill = skill.lower().strip().replace(" ", "_").replace("/", "_")
        if skill and parsed_skill not in exist_skill:
            exist_skill.add(parsed_skill)
            token.append(parsed_skill)
    return " ".join(token)


def _get_scores(skillset: Iterable[str]) -> SkillItem:
    model = _load_model()
    encoder = _load_encoder()
    tfidf = _load_tfidf()

    parsed_skills = _parse_skills(skillset=skillset)
    input_tfidf = tfidf.transform([parsed_skills]).toarray().astype(np.float32)

    pred = model.predict(input_tfidf, verbose=0)[0]

    scores_list = [round(float(v), 4) for v in pred]
    labels = encoder.classes_.tolist()

    paired = sorted(zip(labels, scores_list), key=lambda it: it[1], reverse=True)
    
    labels_sorted, scores_sorted = zip(*paired)

    return list(labels_sorted), list(scores_sorted)


def predict_job_role(skillset: Iterable[str]) -> JobRolePrediction:
    labels, scores = _get_scores(skillset)
    return JobRolePrediction(label=labels[0], confidence=scores[0])


def rank_job_roles(skillset: Iterable[str]) -> JobRoleRanking:
    labels, scores = _get_scores(skillset)
    return JobRoleRanking(
        predictions=[
            JobRolePrediction(label=label, confidence=score)
            for label, score in zip(labels, scores)
        ]
    )


def predict_job_field(skillset: Iterable[str]) -> JobRolePrediction:
    return predict_job_role(skillset)


def get_skill_gap(role: str, user_skills: Iterable[str]) -> list[tuple[str, float]]:
    skill_map = _load_skill_map()
    role_skills = skill_map.get(role, {})
    if not role_skills:
        return []
    user_set = {s.strip().lower() for s in user_skills if s}
    max_count = max(role_skills.values(), default=1)
    gaps = [
        (skill, round(count / max_count, 4))
        for skill, count in role_skills.items()
        if skill not in user_set
    ]
    return sorted(gaps, key=lambda x: x[1], reverse=True)


def get_user_skill_scores(
    role: str, user_skills: Iterable[str]
) -> list[tuple[str, float]]:
    skill_map = _load_skill_map()
    role_skills = skill_map.get(role, {})
    if not role_skills:
        return []
    max_count = max(role_skills.values(), default=1)
    user_set = {s.strip().lower() for s in user_skills if s}
    matched = [
        (skill, round(count / max_count, 4))
        for skill, count in role_skills.items()
        if skill in user_set
    ]
    return sorted(matched, key=lambda x: x[1], reverse=True)
