import unittest
from types import SimpleNamespace

from simkit.dataset import _preserve_renderer_visibility
from simkit.labels import to_coco


class VisibilityTests(unittest.TestCase):
    def test_integer_id_maps_to_named_actual_measurement(self):
        world = SimpleNamespace(objs=[dict(id=3, name="aircraft")])
        boxes = {3: dict(pixels=0, visible=True)}
        measured = dict(aircraft=dict(pixels=216, visible=True, visibility_source="rendered_instance_pass"))
        out = _preserve_renderer_visibility(world, boxes, measured)
        self.assertEqual(out[3]["pixels"], 216)
        self.assertEqual(out[3]["visibility_source"], "rendered_instance_pass")

    def test_empty_actual_mask_not_a_detector_positive(self):
        world = SimpleNamespace(objs=[dict(id=0, name="aircraft", cls_id=1)])
        boxes = {0: dict(x0=1, y0=1, x1=3, y1=3, pixels=0, visible=True)}
        out = _preserve_renderer_visibility(world, boxes, dict(aircraft=dict(pixels=0, visible=False)))
        self.assertFalse(out[0]["visible"])
        _, annotations = to_coco(out, world, 0, "0.png", 8, 8)
        self.assertEqual(annotations, [])


if __name__ == "__main__":
    unittest.main()
