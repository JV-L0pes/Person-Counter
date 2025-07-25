"""
Infrastructure Layer: Age and Gender Classifier
Classificador de idade e gênero usando modelo pré-treinado
"""

import logging
import random
import re
from typing import Optional
import cv2
import numpy as np

from ...domain.entities.person import Person, Gender, AgeGroup


class AgeGenderClassifier:
    """
    Classificador de idade e gênero para pessoas detectadas
    Implementa o padrão Strategy para diferentes algoritmos de classificação
    """

    # Constantes para classificação de idade (podem ser configuráveis via DI)
    CHILD_AGE_THRESHOLD = 17

    def __init__(
        self,
        age_model_path: str = "models/age_net.caffemodel",
        age_proto_path: str = "models/age_deploy.prototxt",
        gender_model_path: str = "models/gender_net.caffemodel",
        gender_proto_path: str = "models/gender_deploy.prototxt",
    ):
        """
        Inicializa o classificador com modelos pré-treinados

        Args:
            age_model_path: Caminho para modelo de idade
            age_proto_path: Caminho para arquivo prototxt de idade
            gender_model_path: Caminho para modelo de gênero
            gender_proto_path: Caminho para arquivo prototxt de gênero
        """
        self.logger = logging.getLogger(__name__)

        # Listas de classes - ordem importa!
        self.age_list = [
            "(0-2)",
            "(4-6)",
            "(8-12)",
            "(15-20)",
            "(25-32)",
            "(38-43)",
            "(48-53)",
            "(60-100)",
        ]
        self.gender_list = ["Male", "Female"]

        try:
            # Carrega modelos usando OpenCV DNN
            self.age_net = cv2.dnn.readNet(age_model_path, age_proto_path)
            self.gender_net = cv2.dnn.readNet(gender_model_path, gender_proto_path)

            self.logger.info("Modelos de classificação carregados com sucesso")

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.warning("Erro ao carregar modelos de classificação: %s", e)
            self.logger.info("Usando classificação mock para desenvolvimento")
            self.age_net = None
            self.gender_net = None

    def classify_person(self, frame: np.ndarray, person: Person) -> Person:
        """
        Classifica idade e gênero de uma pessoa detectada

        Args:
            frame: Frame original da imagem
            person: Pessoa detectada para classificar

        Returns:
            Pessoa com classificação atualizada
        """
        try:
            # Extrai região da pessoa usando bounding box
            face_region = self._extract_person_region(frame, person)

            if face_region is None:
                return person

            # Classifica gênero e idade
            gender = self._classify_gender(face_region)
            age_group = self._classify_age(face_region)

            # Atualiza classificação da pessoa (método do domínio)
            person.update_classification(gender, age_group)

            return person

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("Erro na classificação da pessoa %s: %s", person.id, e)
            return person

    def _extract_person_region(
        self, frame: np.ndarray, person: Person
    ) -> Optional[np.ndarray]:
        """Extrai a região da pessoa do frame usando bounding box"""
        bbox = person.bounding_box

        # Validação de limites
        height, width = frame.shape[:2]
        x1 = max(0, bbox.x)
        y1 = max(0, bbox.y)
        x2 = min(width, bbox.x + bbox.width)
        y2 = min(height, bbox.y + bbox.height)

        if x2 <= x1 or y2 <= y1:
            return None

        # Extrai região e redimensiona para entrada do modelo
        person_region = frame[y1:y2, x1:x2]

        if person_region.size == 0:
            return None

        # Redimensiona para entrada padrão dos modelos (224x224)
        # pylint: disable=no-member
        person_region = cv2.resize(person_region, (224, 224))

        return person_region

    def _classify_gender(self, face_region: np.ndarray) -> Gender:
        """Classifica gênero usando modelo DNN"""
        if self.gender_net is None:
            return self._mock_gender_classification()

        try:
            # Prepara entrada para o modelo
            blob = cv2.dnn.blobFromImage(
                face_region,
                1.0,
                (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
            )

            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()

            # Obtém predição com maior confiança
            gender_idx = np.argmax(gender_preds[0])

            return (
                Gender.MALE if self.gender_list[gender_idx] == "Male" else Gender.FEMALE
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("Erro na classificação de gênero: %s", e)
            return Gender.UNKNOWN

    def _classify_age(self, face_region: np.ndarray) -> AgeGroup:
        """Classifica faixa etária usando modelo DNN"""
        if self.age_net is None:
            return self._mock_age_classification()

        try:
            # Prepara entrada para o modelo
            blob = cv2.dnn.blobFromImage(
                face_region,
                1.0,
                (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
            )

            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()

            # Obtém predição com maior confiança
            age_idx = np.argmax(age_preds[0])
            predicted_age_range = self.age_list[age_idx]

            # Converte faixa etária para AgeGroup baseado na regra de negócio
            return self._age_range_to_group(predicted_age_range)

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("Erro na classificação de idade: %s", e)
            return AgeGroup.UNKNOWN

    def _age_range_to_group(self, age_range: str) -> AgeGroup:
        """
        Converte faixa etária do modelo para AgeGroup do domínio
        Aplica regras de negócio para classificação
        """
        match = re.match(r"\((\d+)-(\d+)\)", age_range)
        if match:
            max_age = int(match.group(2))
            if max_age <= self.CHILD_AGE_THRESHOLD:
                return AgeGroup.CHILD
            else:
                return AgeGroup.ADULT
        # fallback
        return AgeGroup.ADULT

    def _mock_gender_classification(self) -> Gender:
        """Mock para desenvolvimento quando modelo não está disponível"""
        return random.choice([Gender.MALE, Gender.FEMALE])

    def _mock_age_classification(self) -> AgeGroup:
        """Mock para desenvolvimento quando modelo não está disponível"""
        # 20% chance de ser criança, 80% adulto (distribuição realística)
        return AgeGroup.CHILD if random.random() < 0.2 else AgeGroup.ADULT

    def is_models_loaded(self) -> bool:
        """Verifica se os modelos foram carregados corretamente"""
        return self.age_net is not None and self.gender_net is not None
