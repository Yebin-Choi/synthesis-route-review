"""Nesting depth is bounded here, not left to whatever the serialiser allows.

Why this exists. The route and evidence validators used to reject
pathological input by trying `json.dumps` and catching what it raised. That
worked until it did not: measured on Python 3.14, a 2,000-deep list
serialises without error, while `copy.deepcopy` of the same value still
raises RecursionError. So a value could pass validation and then blow the
stack later, in code with no handling for it.

Two tests in this suite were catching that and were being read as a Python
3.14 quirk. They were not: a validator whose answer depends on the
interpreter's stack is a validator that changes its mind between versions.

Real data is shallow. A route report nests 9 deep and a generated route
payload 18, so the bound has room to spare and only malformed input meets it.
"""

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import engine
import workspace


def nested(depth):
    value = []
    for _ in range(depth):
        value = [value]
    return value


class DepthCheckTests(unittest.TestCase):
    def test_ordinary_data_is_nowhere_near_the_bound(self):
        report = engine.analyze_routes(engine.load_example())
        self.assertTrue(engine.json_depth_within(report))
        # Measured, so a future change that deepens the report is noticed
        # here rather than at the bound.
        self.assertLess(self.depth(report), engine.MAX_JSON_DEPTH // 2)

    @staticmethod
    def depth(value):
        best, stack = 0, [(value, 1)]
        while stack:
            item, level = stack.pop()
            best = max(best, level)
            if isinstance(item, dict):
                stack.extend((child, level + 1) for child in item.values())
            elif isinstance(item, (list, tuple)):
                stack.extend((child, level + 1) for child in item)
        return best

    def test_the_bound_is_where_it_says_it_is(self):
        self.assertTrue(engine.json_depth_within(nested(engine.MAX_JSON_DEPTH - 1)))
        self.assertFalse(engine.json_depth_within(nested(engine.MAX_JSON_DEPTH + 1)))

    def test_the_check_does_not_use_the_stack_it_protects(self):
        """A recursive check would raise instead of answering."""
        self.assertFalse(engine.json_depth_within(nested(100_000)))

    def test_a_cycle_is_refused_rather_than_looping_forever(self):
        cyclic = {}
        cyclic["self"] = cyclic
        self.assertFalse(engine.json_depth_within(cyclic))


class ValidatorTests(unittest.TestCase):
    """The gap this closes: serialisable here, not copyable there."""

    def setUp(self):
        self.deep = nested(2000)

    def test_dumps_and_deepcopy_disagree_on_this_interpreter(self):
        """Documents the gap the bound exists to close.

        The bound stays correct whatever this interpreter does; if the two
        ever agree again, the reason written above needs rewriting rather
        than the code.
        """
        def survives(action):
            try:
                action(self.deep)
            except RecursionError:
                return False
            return True

        serialises = survives(json.dumps)
        copies = survives(copy.deepcopy)
        if serialises == copies:
            # Either both survive 2,000 levels or neither does. Both answers
            # are the interpreter's, not the bound's, and the bound holds
            # under all three: the depth check above is what guards the app.
            self.skipTest("이 인터프리터에서는 둘이 같게 동작합니다 "
                          f"(직렬화 {serialises}, 복사 {copies})")
        self.assertNotEqual(serialises, copies,
                            "직렬화와 복사가 같은 값을 다르게 받아들여야 합니다")

    def test_the_report_hash_refuses_depth_instead_of_crashing(self):
        with self.assertRaises(ValueError):
            workspace.report_hash({"evidence": self.deep})

    def test_route_metadata_refuses_depth_before_assessing_anything(self):
        payload = copy.deepcopy(engine.load_example())
        payload["routes"][0]["steps"][0]["conditions"]["description"] = {
            "nested": [self.deep]}
        with self.assertRaisesRegex(ValueError, "conditions.*JSON"):
            engine.analyze_routes(payload)


if __name__ == "__main__":
    unittest.main()
