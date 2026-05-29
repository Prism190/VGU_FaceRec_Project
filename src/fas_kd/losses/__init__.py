from .composite import DistillationObjective
from .kd import cosine_kd_loss, rkd_angle_loss, rkd_distance_loss

__all__ = [
    "DistillationObjective",
    "cosine_kd_loss",
    "rkd_distance_loss",
    "rkd_angle_loss",
]
