"""
Domain Entity: Report
Representa um relatório de contagem de pessoas seguindo DDD
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict
import uuid
from .person import Person, Gender, AgeGroup


@dataclass
class CountSummary:
    """Value Object para resumo de contagens"""

    total_persons: int
    men: int
    women: int
    children: int
    unknown_gender: int
    unknown_age: int

    def __post_init__(self):
        # Validação de negócio: total deve ser consistente
        calculated_total = self.men + self.women + self.unknown_gender
        if calculated_total != self.total_persons:
            raise ValueError("Inconsistência na contagem total de pessoas")


@dataclass
class Report:
    """
    Aggregate Root para relatórios de detecção
    Responsável por consolidar e validar as informações de contagem
    """

    id: str
    persons_detected: List[Person]
    generated_at: datetime
    session_duration_seconds: float

    def __init__(
        self, persons_detected: List[Person], session_duration_seconds: float = 0.0
    ):
        self.id = str(uuid.uuid4())
        self.persons_detected = persons_detected.copy()  # Imutabilidade
        self.generated_at = datetime.now()
        self.session_duration_seconds = session_duration_seconds

    @property
    def count_summary(self) -> CountSummary:
        """Gera resumo de contagens aplicando regras de negócio"""
        men_count = len(
            [
                p
                for p in self.persons_detected
                if p.gender == Gender.MALE and p.age_group == AgeGroup.ADULT
            ]
        )

        women_count = len(
            [
                p
                for p in self.persons_detected
                if p.gender == Gender.FEMALE and p.age_group == AgeGroup.ADULT
            ]
        )

        children_count = len(
            [p for p in self.persons_detected if p.age_group == AgeGroup.CHILD]
        )

        unknown_gender_count = len(
            [
                p
                for p in self.persons_detected
                if p.gender == Gender.UNKNOWN and p.age_group == AgeGroup.ADULT
            ]
        )

        unknown_age_count = len(
            [p for p in self.persons_detected if p.age_group == AgeGroup.UNKNOWN]
        )

        total = len(self.persons_detected)

        return CountSummary(
            total_persons=total,
            men=men_count,
            women=women_count,
            children=children_count,
            unknown_gender=unknown_gender_count,
            unknown_age=unknown_age_count,
        )

    @property
    def detection_rate_per_second(self) -> float:
        """Calcula taxa de detecção por segundo"""
        if self.session_duration_seconds <= 0:
            return 0.0
        return len(self.persons_detected) / self.session_duration_seconds

    def get_confidence_statistics(self) -> Dict[str, float]:
        """Retorna estatísticas de confiança das detecções"""
        if not self.persons_detected:
            return {"min": 0.0, "max": 0.0, "avg": 0.0}

        confidences = [p.bounding_box.confidence for p in self.persons_detected]
        return {
            "min": min(confidences),
            "max": max(confidences),
            "avg": sum(confidences) / len(confidences),
        }

    def to_dict(self) -> Dict:
        """Serialização para API REST"""
        summary = self.count_summary
        confidence_stats = self.get_confidence_statistics()

        return {
            "id": self.id,
            "generated_at": self.generated_at.isoformat(),
            "session_duration_seconds": self.session_duration_seconds,
            "detection_rate_per_second": self.detection_rate_per_second,
            "summary": {
                "total_persons": summary.total_persons,
                "men": summary.men,
                "women": summary.women,
                "children": summary.children,
                "unknown_gender": summary.unknown_gender,
                "unknown_age": summary.unknown_age,
            },
            "confidence_statistics": confidence_stats,
            "persons_detected": [
                {
                    "id": person.id,
                    "gender": person.gender.value,
                    "age_group": person.age_group.value,
                    "confidence": person.bounding_box.confidence,
                    "detected_at": person.detection_timestamp.isoformat(),
                }
                for person in self.persons_detected
            ],
        }
