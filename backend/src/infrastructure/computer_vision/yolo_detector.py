"""
Infrastructure Layer: YOLO Detector
Implementação concreta da detecção de pessoas usando YOLOv8
"""

import logging
from typing import List
import numpy as np
from ultralytics import YOLO

from ...domain.entities.person import Person, BoundingBox, Gender, AgeGroup


class YOLOPersonDetector:
    """
    Detector de pessoas usando YOLOv8
    Implementa o padrão Strategy para diferentes algoritmos de detecção
    """

    def __init__(
        self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5
    ):
        """
        Inicializa o detector YOLO

        Args:
            model_path: Caminho para o modelo YOLO (padrão: yolov8n.pt)
            confidence_threshold: Limiar mínimo de confiança para detecções
        """
        self._validate_confidence_threshold(confidence_threshold)

        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)

        try:
            self.model = YOLO(model_path)
            self.logger.info("Modelo YOLO carregado: %s", model_path)
        except Exception as e:
            self.logger.error("Erro ao carregar modelo YOLO: %s", e)
            raise RuntimeError(f"Falha ao inicializar detector YOLO: {e}") from e

    def _validate_confidence_threshold(self, threshold: float) -> None:
        """Valida o limiar de confiança seguindo fail-fast principle"""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold deve estar entre 0.0 e 1.0")

    def detect_persons(self, frame: np.ndarray) -> List[Person]:
        """
        Detecta pessoas em um frame de vídeo

        Args:
            frame: Frame de vídeo como array numpy

        Returns:
            Lista de pessoas detectadas

        Raises:
            ValueError: Se o frame for inválido
            RuntimeError: Se ocorrer erro na detecção
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame inválido fornecido para detecção")

        try:
            # YOLO inference - classe 0 é 'person' no COCO dataset
            results = self.model(frame, verbose=False)
            persons = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Filtra apenas detecções de pessoas (classe 0)
                        if (
                            int(box.cls[0]) == 0
                            and float(box.conf[0]) >= self.confidence_threshold
                        ):
                            person = self._create_person_from_detection(box)
                            persons.append(person)

            self.logger.debug("Detectadas %d pessoas no frame", len(persons))
            return persons

        except Exception as e:
            self.logger.error("Erro durante detecção YOLO: %s", e)
            raise RuntimeError(f"Falha na detecção de pessoas: {e}") from e

    def _create_person_from_detection(self, box) -> Person:
        """
        Factory method para criar Person a partir de detecção YOLO
        Aplica o padrão Factory para encapsular a criação de objetos
        """
        # Converte coordenadas YOLO para formato padrão
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])

        bounding_box = BoundingBox(
            x=x1, y=y1, width=x2 - x1, height=y2 - y1, confidence=confidence
        )

        # Pessoa criada sem classificação de gênero/idade (será feita posteriormente)
        return Person(
            bounding_box=bounding_box, gender=Gender.UNKNOWN, age_group=AgeGroup.UNKNOWN
        )

    def update_confidence_threshold(self, new_threshold: float) -> None:
        """Atualiza o limiar de confiança com validação"""
        self._validate_confidence_threshold(new_threshold)
        self.confidence_threshold = new_threshold
        self.logger.info("Confidence threshold atualizado para: %s", new_threshold)

    def get_model_info(self) -> dict:
        """Retorna informações sobre o modelo carregado"""
        return {
            "model_type": "YOLOv8",
            "confidence_threshold": self.confidence_threshold,
            "classes_detected": ["person"],
            "input_size": "640x640",  # Padrão YOLOv8
        }
