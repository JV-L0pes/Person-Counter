"""
Domain Repository Interface: Detection Repository
Interface do repositório seguindo Repository Pattern e DDD
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from datetime import datetime

from ..entities.person import Person

# Import necessário para evitar import circular
from ...infrastructure.repositories.memory_detection_repository import (
    MemoryDetectionRepository,
)


# pylint: disable=unnecessary-ellipsis,too-few-public-methods,fixme
class DetectionRepository(ABC):
    """
    Interface abstrata para repositório de detecções
    Implementa Repository Pattern para inversão de dependência (DIP - SOLID)
    Define contrato para persistência de detecções independente da implementação
    """

    @abstractmethod
    def save_detection(self, person: Person) -> None:
        """
        Salva uma detecção no repositório

        Args:
            person: Pessoa detectada para persistir

        Raises:
            ValueError: Se dados da pessoa forem inválidos
            RuntimeError: Se falhar ao salvar
        """
        ...

    @abstractmethod
    def get_detection_by_id(self, person_id: str) -> Optional[Person]:
        """
        Recupera detecção específica por ID único

        Args:
            person_id: Identificador único da pessoa

        Returns:
            Pessoa encontrada ou None se não existir
        """
        ...

    @abstractmethod
    def get_all_detections(self) -> List[Person]:
        """
        Recupera todas as detecções da sessão atual

        Returns:
            Lista de todas as pessoas detectadas na sessão
        """
        ...

    @abstractmethod
    def get_detections_by_timerange(
        self, start_time: datetime, end_time: datetime
    ) -> List[Person]:
        """
        Recupera detecções em intervalo temporal específico

        Args:
            start_time: Data/hora de início do filtro
            end_time: Data/hora de fim do filtro

        Returns:
            Lista de detecções no intervalo especificado

        Raises:
            ValueError: Se intervalo temporal for inválido
        """
        ...

    @abstractmethod
    def get_detections_by_confidence(self, min_confidence: float) -> List[Person]:
        """
        Recupera detecções com confiança mínima especificada

        Args:
            min_confidence: Threshold mínimo de confiança (0.0-1.0)

        Returns:
            Lista de detecções filtradas por confiança

        Raises:
            ValueError: Se confiança estiver fora do range válido
        """
        ...

    @abstractmethod
    def count_detections(self) -> int:
        """
        Conta o número total de detecções na sessão atual

        Returns:
            Número inteiro de detecções
        """
        ...

    @abstractmethod
    def clear_session(self) -> None:
        """
        Limpa todas as detecções da sessão atual
        Mantém histórico se implementação suportar
        """
        ...

    @abstractmethod
    def clear_all(self) -> None:
        """
        Remove todas as detecções (sessão atual e histórico)
        Operação destrutiva - usar com cuidado
        """
        ...

    @abstractmethod
    def get_statistics(self) -> Dict[str, int]:
        """
        Calcula e retorna estatísticas das detecções

        Returns:
            Dicionário com contadores por categoria:
            - total_detections: Total de pessoas detectadas
            - men: Número de homens adultos
            - women: Número de mulheres adultas
            - children: Número de crianças
            - unknown_gender: Pessoas com gênero não identificado
            - unknown_age: Pessoas com idade não identificada
        """
        ...

    @abstractmethod
    def remove_detection(self, person_id: str) -> bool:
        """
        Remove detecção específica por ID

        Args:
            person_id: ID da pessoa a ser removida

        Returns:
            True se removida com sucesso, False se não encontrada
        """
        ...

    @abstractmethod
    def update_detection(self, person: Person) -> bool:
        """
        Atualiza dados de uma detecção existente

        Args:
            person: Pessoa com dados atualizados

        Returns:
            True se atualizada com sucesso, False se não encontrada
        """
        ...


class DetectionRepositoryFactory(ABC):
    """
    Factory abstrata para criação de repositórios
    Implementa Abstract Factory Pattern para flexibilidade de implementações
    """

    @abstractmethod
    def create_detection_repository(self) -> DetectionRepository:
        """
        Cria instância do repositório de detecções

        Returns:
            Implementação concreta do repositório
        """


class MemoryRepositoryFactory(DetectionRepositoryFactory):
    """
    Factory concreta para repositório em memória
    Implementação para desenvolvimento e testes
    """

    def create_detection_repository(self) -> DetectionRepository:
        """Cria repositório em memória"""
        return MemoryDetectionRepository()


class DatabaseRepositoryFactory(DetectionRepositoryFactory):
    """
    Factory concreta para repositório em banco de dados
    Implementação para produção (a ser implementada)
    """

    def __init__(self, connection_string: str):
        super().__init__()
        # Inicializa factory com string de conexão
        self.connection_string = connection_string

    def create_detection_repository(self) -> DetectionRepository:
        # TODO: Implementar repositório com banco de dados
        # return DatabaseDetectionRepository(self.connection_string)
        raise NotImplementedError(
            "Repositório com banco de dados ainda não implementado"
        )


# Função utilitária para obter factory baseada em configuração
def get_repository_factory(
    repository_type: str = "memory", **kwargs
) -> DetectionRepositoryFactory:
    """
    Factory method para obter factory de repositório

    Args:
        repository_type: Tipo do repositório ("memory" ou "database")
        **kwargs: Argumentos específicos do repositório

    Returns:
        Factory apropriada para o tipo solicitado

    Raises:
        ValueError: Se tipo não for suportado
    """
    if repository_type.lower() == "memory":
        return MemoryRepositoryFactory()

    if repository_type.lower() == "database":
        connection_string = kwargs.get("connection_string")
        if not connection_string:
            raise ValueError(
                "Connection string é obrigatória para repositório de banco"
            )
        return DatabaseRepositoryFactory(connection_string)

    raise ValueError(f"Tipo de repositório não suportado: {repository_type}")
