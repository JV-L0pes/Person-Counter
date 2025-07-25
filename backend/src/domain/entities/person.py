"""
Domain Entity: Person
Representa uma pessoa detectada na imagem seguindo os princípios do DDD
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple
from datetime import datetime
import uuid


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class AgeGroup(Enum):
    CHILD = "child"  # 0-17 anos
    ADULT = "adult"  # 18+ anos
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BoundingBox:
    """Value Object para representar as coordenadas da detecção"""

    x: int
    y: int
    width: int
    height: int
    confidence: float

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence deve estar entre 0.0 e 1.0")

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class Person:
    """
    Entidade principal do domínio representando uma pessoa detectada
    Segue os princípios do DDD com identidade única e comportamentos específicos
    """

    id: str
    bounding_box: BoundingBox
    gender: Gender
    age_group: AgeGroup
    detection_timestamp: datetime

    def __init__(
        self,
        bounding_box: BoundingBox,
        gender: Gender = Gender.UNKNOWN,
        age_group: AgeGroup = AgeGroup.UNKNOWN,
    ):
        self.id = str(uuid.uuid4())
        self.bounding_box = bounding_box
        self.gender = gender
        self.age_group = age_group
        self.detection_timestamp = datetime.now()

    @property
    def is_child(self) -> bool:
        """Verifica se a pessoa é classificada como criança"""
        return self.age_group == AgeGroup.CHILD

    @property
    def is_adult(self) -> bool:
        """Verifica se a pessoa é classificada como adulto"""
        return self.age_group == AgeGroup.ADULT

    def update_classification(self, gender: Gender, age_group: AgeGroup) -> None:
        """Atualiza a classificação da pessoa mantendo a imutabilidade dos dados essenciais"""
        self.gender = gender
        self.age_group = age_group

    def __eq__(self, other) -> bool:
        if not isinstance(other, Person):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
