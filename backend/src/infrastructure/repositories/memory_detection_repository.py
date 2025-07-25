"""
Infrastructure Repository: Memory Detection Repository
Implementação em memória do repositório de detecções seguindo DDD
"""

from typing import List, Optional, Dict
from datetime import datetime
import threading
import logging
import sys

from ...domain.entities.person import Person
from ...domain.repositories.detection_repository import DetectionRepository


class MemoryDetectionRepository(DetectionRepository):
    """
    Implementação em memória do repositório de detecções
    Implementa Repository Pattern para abstração de persistência
    Thread-safe para uso em aplicações web
    """

    def __init__(self):
        """Inicializa repositório em memória com controle de concorrência"""
        self._detections: Dict[str, Person] = {}
        self._session_detections: List[Person] = []
        self._lock = threading.RLock()  # ReentrantLock para thread safety
        self.logger = logging.getLogger(__name__)

        self.logger.info("Repositório em memória inicializado")

    def save_detection(self, person: Person) -> None:
        """
        Salva uma detecção no repositório

        Args:
            person: Pessoa detectada para salvar

        Raises:
            ValueError: Se pessoa for None ou inválida
        """
        if person is None:
            raise ValueError("Person não pode ser None")

        with self._lock:
            try:
                # Armazena no dicionário principal (histórico completo)
                self._detections[person.id] = person

                # Adiciona à sessão atual
                self._session_detections.append(person)

                self.logger.debug("Detecção salva: %s", person.id)

            except Exception as e:
                self.logger.debug("Erro ao salvar detecção: %s", e)
                raise RuntimeError(f"Falha ao salvar detecção: {e}") from e

    def get_detection_by_id(self, person_id: str) -> Optional[Person]:
        """
        Recupera detecção por ID

        Args:
            person_id: ID único da pessoa

        Returns:
            Pessoa encontrada ou None se não existir
        """
        if not person_id:
            return None

        with self._lock:
            return self._detections.get(person_id)

    def get_all_detections(self) -> List[Person]:
        """
        Retorna todas as detecções da sessão atual

        Returns:
            Lista com cópia das detecções (imutabilidade)
        """
        with self._lock:
            # Retorna cópia para manter imutabilidade
            return self._session_detections.copy()

    def get_detections_by_timerange(
        self, start_time: datetime, end_time: datetime
    ) -> List[Person]:
        """
        Recupera detecções em intervalo de tempo específico

        Args:
            start_time: Data/hora de início
            end_time: Data/hora de fim

        Returns:
            Lista de detecções no intervalo

        Raises:
            ValueError: Se intervalo for inválido
        """
        self._validate_timerange(start_time, end_time)

        with self._lock:
            filtered_detections = [
                person
                for person in self._session_detections
                if start_time <= person.detection_timestamp <= end_time
            ]

            return filtered_detections

    def get_detections_by_confidence(self, min_confidence: float) -> List[Person]:
        """
        Recupera detecções com confiança mínima

        Args:
            min_confidence: Confiança mínima (0.0-1.0)

        Returns:
            Lista de detecções filtradas

        Raises:
            ValueError: Se confiança for inválida
        """
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("Confiança mínima deve estar entre 0.0 e 1.0")

        with self._lock:
            filtered_detections = [
                person
                for person in self._session_detections
                if person.bounding_box.confidence >= min_confidence
            ]

            return filtered_detections

    def count_detections(self) -> int:
        """
        Conta total de detecções na sessão atual

        Returns:
            Número de detecções
        """
        with self._lock:
            return len(self._session_detections)

    def clear_session(self) -> None:
        """
        Limpa detecções da sessão atual
        Mantém histórico completo no dicionário principal
        """
        with self._lock:
            self._session_detections.clear()
            self.logger.info("Sessão de detecções limpa")

    def clear_all(self) -> None:
        """
        Limpa todas as detecções (sessão atual e histórico)
        Usar com cuidado - remove todos os dados
        """
        with self._lock:
            self._detections.clear()
            self._session_detections.clear()
            self.logger.warning("Todas as detecções foram removidas")

    def get_statistics(self) -> Dict[str, int]:
        """
        Retorna estatísticas das detecções da sessão atual

        Returns:
            Dicionário com estatísticas organizadas
        """
        with self._lock:
            detections = self._session_detections

            # Contadores por categoria
            stats = {
                "total_detections": len(detections),
                "men": 0,
                "women": 0,
                "children": 0,
                "unknown_gender": 0,
                "unknown_age": 0,
            }

            # Conta cada categoria aplicando regras de negócio
            for person in detections:
                if person.is_child:
                    stats["children"] += 1
                elif person.gender.value == "male":
                    stats["men"] += 1
                elif person.gender.value == "female":
                    stats["women"] += 1
                else:
                    stats["unknown_gender"] += 1

                if person.age_group.value == "unknown":
                    stats["unknown_age"] += 1

            return stats

    def get_detection_timeline(self) -> List[Dict]:
        """
        Retorna timeline das detecções para análise temporal

        Returns:
            Lista ordenada por timestamp com dados resumidos
        """
        with self._lock:
            timeline = []

            for person in sorted(
                self._session_detections, key=lambda p: p.detection_timestamp
            ):
                timeline.append(
                    {
                        "timestamp": person.detection_timestamp.isoformat(),
                        "person_id": person.id,
                        "gender": person.gender.value,
                        "age_group": person.age_group.value,
                        "confidence": person.bounding_box.confidence,
                    }
                )

            return timeline

    def remove_detection(self, person_id: str) -> bool:
        """
        Remove detecção específica por ID

        Args:
            person_id: ID da pessoa a ser removida

        Returns:
            True se removida com sucesso, False se não encontrada
        """
        if not person_id:
            return False

        with self._lock:
            # Remove do histórico completo
            removed_from_history = self._detections.pop(person_id, None) is not None

            # Remove da sessão atual
            original_length = len(self._session_detections)
            self._session_detections = [
                p for p in self._session_detections if p.id != person_id
            ]
            removed_from_session = len(self._session_detections) < original_length

            success = removed_from_history or removed_from_session

            if success:
                self.logger.debug("Detecção removida: %s", person_id)

            return success

    def update_detection(self, person: Person) -> bool:
        """
        Atualiza detecção existente

        Args:
            person: Pessoa com dados atualizados

        Returns:
            True se atualizada, False se não encontrada
        """
        if person is None or not person.id:
            return False

        with self._lock:
            # Atualiza no histórico
            if person.id in self._detections:
                self._detections[person.id] = person

                # Atualiza na sessão atual
                for i, session_person in enumerate(self._session_detections):
                    if session_person.id == person.id:
                        self._session_detections[i] = person
                        break

                self.logger.debug("Detecção atualizada: %s", person.id)
                return True

            return False

    def _validate_timerange(self, start_time: datetime, end_time: datetime) -> None:
        """Valida intervalo de tempo"""
        if start_time is None or end_time is None:
            raise ValueError("Start_time e end_time não podem ser None")

        if start_time >= end_time:
            raise ValueError("Start_time deve ser anterior a end_time")

    def get_repository_info(self) -> Dict[str, any]:
        """
        Retorna informações sobre o repositório

        Returns:
            Dicionário com metadados do repositório
        """
        with self._lock:
            return {
                "type": "MemoryDetectionRepository",
                "total_detections_in_history": len(self._detections),
                "current_session_detections": len(self._session_detections),
                "thread_safe": True,
                "persistent": False,
                "memory_usage_estimate": self._estimate_memory_usage(),
            }

    def _estimate_memory_usage(self) -> str:
        """Estima uso de memória do repositório"""
        total_size = (
            sys.getsizeof(self._detections)
            + sys.getsizeof(self._session_detections)
            + sum(sys.getsizeof(person) for person in self._detections.values())
        )

        # Converte para formato legível
        if total_size < 1024:
            return f"{total_size} bytes"
        elif total_size < 1024 * 1024:
            return f"{total_size / 1024:.2f} KB"
        else:
            return f"{total_size / (1024 * 1024):.2f} MB"
