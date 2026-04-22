from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SimplePose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class ObjectRecordModel:
    object_id: int
    pose: SimplePose = field(default_factory=SimplePose)
    texture_class: Optional[int] = None
    classified: bool = False
    delivered: bool = False
    pixel_xy: Tuple[int, int] = (0, 0)

    def set_pose(self, pose: SimplePose) -> None:
        self.pose = pose

    def set_texture_class(self, cls_id: int) -> None:
        self.texture_class = cls_id
        self.classified = True

    def mark_delivered(self) -> None:
        self.delivered = True


@dataclass
class CategoryMemory:
    objects: Dict[int, ObjectRecordModel] = field(default_factory=dict)
    delivered_categories: set = field(default_factory=set)

    def update_object(self, object_id: int, pose: SimplePose) -> None:
        record = self.objects.setdefault(object_id, ObjectRecordModel(object_id=object_id))
        record.pose = pose

    def set_object_category(self, object_id: int, cls_id: int) -> None:
        record = self.objects.setdefault(object_id, ObjectRecordModel(object_id=object_id))
        record.set_texture_class(cls_id)

    def get_objects_in_category(self, cls_id: int) -> List[int]:
        return [oid for oid, rec in self.objects.items() if rec.texture_class == cls_id and not rec.delivered]

    def mark_category_delivered(self, cls_id: int) -> None:
        self.delivered_categories.add(cls_id)
        for rec in self.objects.values():
            if rec.texture_class == cls_id:
                rec.mark_delivered()

    def get_next_undelivered_category(self) -> Optional[int]:
        categories = sorted({rec.texture_class for rec in self.objects.values() if rec.texture_class is not None})
        for cls_id in categories:
            if cls_id not in self.delivered_categories and self.get_objects_in_category(cls_id):
                return cls_id
        return None


class SceneRegistry:
    def __init__(self) -> None:
        self._objects: Dict[int, ObjectRecordModel] = {}

    def initialize_stub_scene(self, count: int = 3) -> List[int]:
        self._objects = {}
        for idx in range(count):
            self._objects[idx + 1] = ObjectRecordModel(object_id=idx + 1, pixel_xy=(100 + idx * 40, 100))
        return list(self._objects.keys())

    def get_objects(self) -> Dict[int, ObjectRecordModel]:
        return self._objects

    def get_pixel_location(self, object_id: int) -> Tuple[int, int]:
        return self._objects[object_id].pixel_xy


class ObjectPoseEstimator:
    def estimate_pose_for_object(self, object_id: int, registry: SceneRegistry) -> SimplePose:
        pixel_x, pixel_y = registry.get_pixel_location(object_id)
        return SimplePose(x=float(pixel_x) / 1000.0, y=float(pixel_y) / 1000.0, z=0.0)
