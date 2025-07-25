# pylint: disable=no-member
"""
Application Layer: Detect Persons Use Case
Orquestra a detecção e classificação de pessoas seguindo Clean Architecture
"""

import logging
from typing import List, Optional
from datetime import datetime
import cv2
import numpy as np

from ...domain.entities.person import Person
from ...domain.repositories.detection_repository import DetectionRepository


class DetectPersonsUseCase:
    """
    Caso de uso principal para detecção e classificação de pessoas
    Implementa o padrão Use Case da Clean Architecture
    """

    def __init__(
        self,
        person_detector,
        age_gender_classifier,
        detection_repository: DetectionRepository,
    ):
        """
        Injeta dependências seguindo Dependency Inversion Principle (SOLID)

        Args:
            person_detector: Detector de pessoas (YOLOv8)
            age_gender_classifier: Classificador de idade/gênero
            detection_repository: Repositório para persistir detecções
        """
        self.person_detector = person_detector
        self.age_gender_classifier = age_gender_classifier
        self.detection_repository = detection_repository
        self.logger = logging.getLogger(__name__)

        # Estado da sessão para controle
        self.session_start_time: Optional[datetime] = None
        self.total_frames_processed = 0

    def start_detection_session(self) -> None:
        """Inicia uma nova sessão de detecção"""
        self.session_start_time = datetime.now()
        self.total_frames_processed = 0
        self.detection_repository.clear_session()
        self.logger.info("Nova sessão de detecção iniciada")

    def process_frame(self, frame: np.ndarray) -> List[Person]:
        """
        Processa um frame individual detectando e classificando pessoas

        Args:
            frame: Frame de vídeo como array numpy

        Returns:
            Lista de pessoas detectadas e classificadas neste frame

        Raises:
            ValueError: Se frame for inválido
            RuntimeError: Se ocorrer erro no processamento
        """
        self._validate_frame(frame)

        if self.session_start_time is None:
            self.start_detection_session()

        try:
            # Passo 1: Detectar pessoas no frame usando YOLO
            detected_persons = self.person_detector.detect_persons(frame)

            # Passo 2: Classificar idade e gênero de cada pessoa detectada
            classified_persons = []
            for person in detected_persons:
                classified_person = self.age_gender_classifier.classify_person(
                    frame, person
                )
                classified_persons.append(classified_person)

            # Passo 3: Persistir detecções no repositório
            for person in classified_persons:
                self.detection_repository.save_detection(person)

            self.total_frames_processed += 1

            self.logger.debug(
                "Frame processado: %d pessoas detectadas", len(classified_persons)
            )

            return classified_persons

        except Exception as e:
            self.logger.error("Erro ao processar frame: %s", e)
            raise RuntimeError(f"Falha no processamento do frame: {e}") from e

    def process_video_stream(self, video_source: int = 0) -> None:
        """
        Processa stream de vídeo em tempo real (webcam)

        Args:
            video_source: Índice da câmera (0 para webcam padrão)
        """
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera: {video_source}")

        self.logger.info("Iniciando captura de vídeo da câmera: %s", video_source)
        self.start_detection_session()

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    self.logger.warning("Falha ao capturar frame da câmera")
                    break

                # Processa frame atual
                persons = self.process_frame(frame)

                # Desenha detecções no frame para visualização
                annotated_frame = self._draw_detections(frame, persons)

                # Exibe frame com detecções
                cv2.imshow("Person Detection", annotated_frame)

                # Sai com 'q'
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            self.logger.info("Detecção interrompida pelo usuário")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Stream de vídeo finalizado")

    def get_current_statistics(self) -> dict:
        """Retorna estatísticas da sessão atual"""
        if self.session_start_time is None:
            return {"error": "Nenhuma sessão ativa"}

        session_duration = (datetime.now() - self.session_start_time).total_seconds()
        all_detections = self.detection_repository.get_all_detections()

        return {
            "session_duration_seconds": session_duration,
            "frames_processed": self.total_frames_processed,
            "total_persons_detected": len(all_detections),
            "frames_per_second": (
                self.total_frames_processed / session_duration
                if session_duration > 0
                else 0.0
            ),
            "detection_rate": (
                len(all_detections) / session_duration if session_duration > 0 else 0.0
            ),
        }

    def _validate_frame(self, frame: np.ndarray) -> None:
        """Valida se o frame é válido para processamento"""
        if frame is None:
            raise ValueError("Frame não pode ser None")

        if frame.size == 0:
            raise ValueError("Frame não pode estar vazio")

        if len(frame.shape) != 3:
            raise ValueError("Frame deve ter 3 dimensões (altura, largura, canais)")

    def _draw_detections(self, frame: np.ndarray, persons: List[Person]) -> np.ndarray:
        """
        Desenha bounding boxes e classificações no frame
        Método auxiliar para visualização das detecções
        """
        annotated_frame = frame.copy()

        for person in persons:
            bbox = person.bounding_box

            # Define cor baseada na classificação
            color = self._get_color_for_classification(person)

            # Desenha bounding box
            cv2.rectangle(
                annotated_frame,
                (bbox.x, bbox.y),
                (bbox.x + bbox.width, bbox.y + bbox.height),
                color,
                2,
            )

            # Prepara texto da classificação
            classification_text = self._format_classification_text(person)

            # Desenha fundo para o texto
            text_size = cv2.getTextSize(
                classification_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )[0]
            cv2.rectangle(
                annotated_frame,
                (bbox.x, bbox.y - text_size[1] - 10),
                (bbox.x + text_size[0], bbox.y),
                color,
                -1,
            )

            # Desenha texto da classificação
            cv2.putText(
                annotated_frame,
                classification_text,
                (bbox.x, bbox.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Adiciona contador no canto superior esquerdo
        total_persons = len(persons)
        counter_text = f"Pessoas detectadas: {total_persons}"
        cv2.putText(
            annotated_frame,
            counter_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        return annotated_frame

    def _get_color_for_classification(self, person: Person) -> tuple:
        """Retorna cor baseada na classificação da pessoa"""
        if person.is_child:
            return (0, 255, 255)  # Amarelo para crianças
        elif person.gender.value == "male":
            return (255, 0, 0)  # Azul para homens
        elif person.gender.value == "female":
            return (255, 0, 255)  # Magenta para mulheres
        else:
            return (128, 128, 128)  # Cinza para não classificados

    def _format_classification_text(self, person: Person) -> str:
        """Formata texto da classificação para exibição"""
        gender_text = person.gender.value.capitalize()
        age_text = person.age_group.value.capitalize()
        confidence = person.bounding_box.confidence

        return f"{gender_text} {age_text} ({confidence:.2f})"
