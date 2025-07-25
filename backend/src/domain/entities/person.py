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
    CHILD = "child"      # 0-17 anos
    ADULT = "adult"      # 18+ anos
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