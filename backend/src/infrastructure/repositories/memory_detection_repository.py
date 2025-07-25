from ...domain.repositories.detection_repository import DetectionRepository

class MemoryDetectionRepository(DetectionRepository):
    """Implementação em memória do repositório de detecções para desenvolvimento e testes"""
    def save_detection(self, person):
        raise NotImplementedError()
    def get_detection_by_id(self, person_id):
        raise NotImplementedError()
    def get_all_detections(self):
        raise NotImplementedError()
    def get_detections_by_timerange(self, start_time, end_time):
        raise NotImplementedError()
    def get_detections_by_confidence(self, min_confidence):
        raise NotImplementedError()
    def count_detections(self):
        raise NotImplementedError()
    def clear_session(self):
        raise NotImplementedError()
    def clear_all(self):
        raise NotImplementedError()
    def get_statistics(self):
        raise NotImplementedError()
    def remove_detection(self, person_id):
        raise NotImplementedError()
    def update_detection(self, person):
        raise NotImplementedError()
