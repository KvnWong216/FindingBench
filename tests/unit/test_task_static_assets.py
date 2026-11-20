"""SceneSlots, PlacementPriors, eligibility."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rummagebench.authoring.tasks.eligibility import (
    ObjectEligibility, build_eligibility, lookalike_keys)
from rummagebench.authoring.tasks.priors import (
    PlacementPriors, apply_manual, apply_review, mine_priors, parse_init_facts)
from rummagebench.authoring.tasks.slots import (
    SceneSlots, classify_link, room_type_of)

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets" / "tasks"
GENERATED = REPO / "build" / "tasks"
needs_generated = pytest.mark.skipif(
    not (GENERATED / "priors" / "eligibility_v1.yaml").exists()
    or not (GENERATED / "scenes").exists(),
    reason="generated task assets not built (scripts/tasks/build_*.py, mine_priors.py)")
RULES = yaml.safe_load((ASSETS / "slot_rules_v1.yaml").read_text())


class FakeTaxonomy:
    CATS = {"cabinet.n.01": ["bottom_cabinet", "top_cabinet"],
            "chopping_board.n.01": ["chopping_board", "cutting_board"],
            "countertop.n.01": ["countertop"],
            "bowl.n.01": ["bowl"], "rag.n.01": ["rag"],
            "chair.n.01": ["straight_chair"], "piano.n.01": ["piano"],
            "tablefork.n.01": ["tablefork"], "tablespoon.n.02": ["tablespoon"]}
    ABIL = {"bowl.n.01": {"rigidBody", "nonSubstance"},
            "rag.n.01": {"nonSubstance", "cloth"},
            "chair.n.01": {"rigidBody", "nonSubstance", "sceneObject"},
            "piano.n.01": {"rigidBody", "nonSubstance", "sceneObject"},
            "tablefork.n.01": {"rigidBody", "nonSubstance"},
            "tablespoon.n.02": {"rigidBody", "nonSubstance"}}
    PARENTS = {"tablefork.n.01": ["cutlery.n.02"], "tablespoon.n.02": ["cutlery.n.02"],
               "bowl.n.01": ["vessel.n.03"]}
    SUBTREE = {"cutlery.n.02": ["tablefork", "tablespoon", "table_knife"],
               "vessel.n.03": [f"c{i}" for i in range(100)]}

    def get_categories(self, s):
        return self.CATS.get(s, [])

    def get_synset_from_category(self, c):
        for s, cats in self.CATS.items():
            if c in cats:
                return s
        return None

    def get_abilities(self, s):
        return {a: {} for a in self.ABIL.get(s, set())}

    def get_parents(self, s):
        return self.PARENTS.get(s, [])

    def get_subtree_categories(self, s):
        return self.SUBTREE.get(s, [])


BDDL = """(define (problem x-0)
 (:objects bowl.n.01_1 - bowl.n.01 cabinet.n.01_1 - cabinet.n.01)
 (:init (inside bowl.n.01_1 cabinet.n.01_1) (ontop bowl.n.01_2 countertop.n.01_1)
        (ontop chopping_board.n.01_1 countertop.n.01_1)
        (ontop rag.n.01_1 floor.n.01_1)
        (inroom cabinet.n.01_1 kitchen) (inroom countertop.n.01_1 kitchen)
        (inroom floor.n.01_1 kitchen))
 (:goal (and (inside ?bowl.n.01_1 ?countertop.n.01_1))))"""


def test_room_type_and_link_classification():
    assert room_type_of("kitchen_0") == "kitchen"
    assert room_type_of("private_office_12") == "private_office"
    lc = RULES["link_classification"]
    assert classify_link([0.03, 0.31, 0.51], lc) == "door"
    assert classify_link([0.38, 0.37, 0.13], lc) == "drawer"
    assert classify_link([0.56, 0.31, 0.79], lc) == "ambiguous"


def test_bddl_init_only_and_floor_skipped():
    facts = parse_init_facts(BDDL)
    assert ("inside", "bowl.n.01_1", "cabinet.n.01_1") in facts
    assert all("?" not in a for _, a, _ in facts)  # goal facts excluded
    pri = mine_priors([("x", BDDL)], FakeTaxonomy(), RULES, {"type": "test"})
    keys = {e.key for e in pri.entries}
    # drawer + cabinet_interior share ONE class: the inside fact counts once
    assert keys == {("bowl.n.01", "cabinet_storage", "kitchen"),
                    ("bowl.n.01", "countertop", "kitchen"),
                    ("chopping_board.n.01", "countertop", "kitchen")}
    # aliases share ONE entry: the fact is counted once, not once per category
    board = pri.get("cutting_board", "countertop", "kitchen")
    assert board is pri.get("chopping_board", "countertop", "kitchen")
    assert board.evidence_count == 1
    assert board.object_categories == ["chopping_board", "cutting_board"]
    st = pri.get("bowl", "drawer", "kitchen")
    assert st is pri.get("bowl", "cabinet_interior", "kitchen")
    assert st.evidence_count == 1 and st.covers_slot_types == ["cabinet_interior", "drawer"]
    assert all(not e.approved for e in pri.entries)  # mining never approves


def test_review_is_explicit_and_evidence_backed():
    mined = mine_priors([("x", BDDL)], FakeTaxonomy(), RULES, {})
    review = {"reviewer": "t", "status": "s",
              "approve": [{"object_synset": "bowl.n.01", "slot_class": "cabinet_storage",
                           "room_type": "kitchen", "weight": 0.5},
                          {"object_synset": "chopping_board.n.01",
                           "slot_class": "countertop", "room_type": "kitchen",
                           "exclude_categories": ["cutting_board"]}]}
    rev = apply_review(mined, review)
    assert rev.is_approved("bowl", "drawer", "kitchen")
    assert rev.is_approved("bowl", "cabinet_interior", "kitchen")
    assert not rev.is_approved("bowl", "countertop", "kitchen")
    e = rev.get("bowl", "drawer", "kitchen")
    # four quantities kept apart
    assert (e.evidence_count, e.approved, e.compatibility, e.generation_weight) == (
        1, True, "low", 0.5)
    # reviewer narrows the class without touching evidence
    assert rev.is_approved("chopping_board", "countertop", "kitchen")
    assert not rev.is_approved("cutting_board", "countertop", "kitchen")
    review["approve"][0]["exclude_slot_types"] = ["drawer"]
    rev = apply_review(mined, review)
    assert not rev.is_approved("bowl", "drawer", "kitchen")
    assert rev.is_approved("bowl", "cabinet_interior", "kitchen")
    assert rev.get("bowl", "drawer", "kitchen").evidence_count == 1
    review["approve"][0]["exclude_slot_types"] = ["countertop"]
    with pytest.raises(ValueError):
        apply_review(mined, review)  # excluding a type the class does not cover
    bad = {"approve": [{"object_synset": "piano.n.01", "slot_class": "cabinet_storage",
                        "room_type": "kitchen"}]}
    with pytest.raises(ValueError):
        apply_review(mined, bad)


def test_eligibility_is_a_positive_whitelist(tmp_path):
    objects = tmp_path / "objects"
    for cat, model, bbox in [("bowl", "m1", [0.2, 0.2, 0.1]),
                             ("bowl", "huge", [0.9, 0.9, 0.4]),
                             ("tablefork", "f1", [0.2, 0.03, 0.02]),
                             ("straight_chair", "c1", [0.5, 0.5, 0.9]),
                             ("rag", "r1", [0.2, 0.2, 0.02])]:
        d = objects / cat / model / "misc"
        d.mkdir(parents=True)
        (d / "metadata.json").write_text(f'{{"bbox_size": {bbox}}}')
    inv = {"bowl": ["m1", "huge"], "tablefork": ["f1"], "straight_chair": ["c1"],
           "rag": ["r1"]}
    rules = yaml.safe_load((ASSETS / "priors" / "eligibility_rules_v1.yaml").read_text())
    el = build_eligibility(["bowl", "tablefork", "straight_chair", "rag", "piano"],
                           FakeTaxonomy(), inv, objects, {}, rules, {})
    assert set(el.categories) == {"bowl", "tablefork"}
    assert [m.model_id for m in el.categories["bowl"].models] == ["m1"]
    assert "sceneObject" in el.rejected["straight_chair"]
    assert "rigidBody" in el.rejected["rag"]
    assert el.rejected["piano"] == "not in Level-1 usable catalog"
    again = ObjectEligibility.from_dict(el.to_dict())
    assert again.to_dict() == el.to_dict()


def test_lookalike_keys():
    tax = FakeTaxonomy()
    fork = lookalike_keys("tablefork", "tablefork.n.01", tax, 12)
    spoon = lookalike_keys("tablespoon", "tablespoon.n.02", tax, 12)
    bowl = lookalike_keys("bowl", "bowl.n.01", tax, 12)
    assert "parent:cutlery.n.02" in fork and "parent:cutlery.n.02" in spoon
    assert not any(k.startswith("parent:") for k in bowl)  # 100-wide group ignored
    assert "head:bowl" in lookalike_keys("mixing_bowl", "bowl.n.03", tax, 12)
    assert "form:jar" in lookalike_keys("basil_jar", "basil__jar.n.01", tax, 12)


# ---- generated assets --------------------------------------------------

FURNITURE = ("chair", "piano", "washer", "bookcase", "mirror", "sofa", "table",
             "cabinet", "desk", "fridge", "bed", "shelf", "countertop")


@needs_generated
def test_generated_whitelist_has_no_furniture():
    el = ObjectEligibility.load(GENERATED / "priors" / "eligibility_v1.yaml")
    assert len(el.categories) >= 20
    for name, c in el.categories.items():
        assert not any(name == f or name.endswith("_" + f) for f in FURNITURE), name
        for m in c.models:
            assert max(m.bbox) <= 0.35 + 1e-9, (name, m.model_id)


@needs_generated
def test_generated_priors_have_no_duplicated_evidence():
    pri = PlacementPriors.load(GENERATED / "priors" / "placement_v1.yaml")
    assert pri.slot_classes == {"drawer": "cabinet_storage",
                                "cabinet_interior": "cabinet_storage"}
    keys = [e.key for e in pri.entries]
    assert len(keys) == len(set(keys))
    assert not any(e.slot_class in ("drawer", "cabinet_interior") for e in pri.entries)
    # each asset category belongs to exactly one synset entry family
    board = pri.get("cutting_board", "countertop", "kitchen")
    assert board is pri.get("chopping_board", "countertop", "kitchen")
    assert board.object_categories == ["chopping_board", "cutting_board"]


@needs_generated
def test_generated_priors_only_approved_generate():
    pri = PlacementPriors.load(GENERATED / "priors" / "placement_v1.yaml")
    el = ObjectEligibility.load(GENERATED / "priors" / "eligibility_v1.yaml")
    assert pri.approved_entries()
    assert set(el.categories) <= pri.approved_categories()
    for e in pri.entries:
        if e.approved:
            assert e.generation_weight > 0
            assert e.review and e.review["reviewer"]
            # BDDL approvals are evidence-backed; manual ones say so
            assert e.evidence_count >= 1 or e.review.get("source") == "manual"


@needs_generated
def test_generated_scene_slots_load_and_are_robot_free():
    path = GENERATED / "scenes" / "Beechwood_0_int.yaml"
    ss = SceneSlots.load(path)
    assert ss.scene == "Beechwood_0_int"
    ids = ss.by_id()
    assert len(ids) == len(ss.slots)  # unique slot ids
    q = ids["Beechwood_0_int/bottom_cabinet_no_top_qohxjq_0/link_2"]
    assert (q.slot_type, q.relation, q.requires_open, q.room) == (
        "drawer", "inside", True, "kitchen_0")
    text = path.read_text()
    for robot_word in ("anchor", "r1pro", "urdf", "reach"):
        assert robot_word not in text  # no embodiment data in SceneSlots


def test_rect_distance():
    from rummagebench.authoring.tasks.regions import rect_distance
    import math
    a = (0.0, 0.0, 1.0, 1.0, 0.0)
    assert rect_distance(a, (2.0, 0.0, 1.0, 1.0, 0.0)) == pytest.approx(1.0)
    assert rect_distance(a, (0.5, 0.5, 1.0, 1.0, 0.0)) == 0.0  # overlap
    # rotated 45 deg: corner points toward a at distance 2 - sqrt(0.5) - 0.5
    d = rect_distance(a, (2.0, 0.0, 1.0, 1.0, math.pi / 4))
    assert d == pytest.approx(2.0 - math.sqrt(0.5) - 0.5, abs=1e-6)


def test_build_regions_boundary_and_uniqueness():
    from rummagebench.authoring.tasks.slots import (
        Furniture, Landmark, build_regions)
    rules = RULES["regions"]

    def furn(e, x):
        return Furniture(e, "bottom_cabinet", "m", "kitchen_0", "kitchen",
                         (x, 0.0, 0.4), 0.0, True, (0.6, 0.6, 0.8))

    def slot_of(e):
        class S:  # only parent_entity is read
            parent_entity = e
        return S()
    lm = Landmark("sink_0", "furniture_sink", "sink", "kitchen_0", (0.0, 0.0, 0.5), 0.0,
                  (0.6, 0.6, 1.0))
    # cab_a touches the sink; cab_b 0.9 m away -> clear band respected
    regs = build_regions("S", [lm], [furn("cab_a", 0.6), furn("cab_b", 2.1)],
                         [slot_of("cab_a"), slot_of("cab_b")], rules)
    assert regs[0].furniture == ("cab_a",) and regs[0].usable
    # cab_b 0.4 m away: outside touch_m but inside the clear band -> ambiguous
    regs = build_regions("S", [lm], [furn("cab_a", 0.6), furn("cab_b", 1.0)],
                         [slot_of("cab_a"), slot_of("cab_b")], rules)
    assert not regs[0].clear and not regs[0].usable
    # two sinks in a room -> name not unique
    lm2 = Landmark("sink_1", "furniture_sink", "sink", "kitchen_0", (5.0, 0.0, 0.5), 0.0,
                   (0.6, 0.6, 1.0))
    regs = build_regions("S", [lm, lm2], [furn("cab_a", 0.6)], [slot_of("cab_a")], rules)
    assert not any(r.unique_name for r in regs)


@needs_generated
def test_generated_scene_has_usable_kitchen_regions():
    ss = SceneSlots.load(GENERATED / "scenes" / "Beechwood_0_int.yaml")
    usable = {r.name for r in ss.regions if r.room == "kitchen_0" and r.usable}
    assert {"stove", "fridge"} <= usable
    sink = [r for r in ss.regions if r.room == "kitchen_0" and r.name == "sink"]
    assert sink and not sink[0].usable  # continuous counter run: ambiguous


def test_measured_drawer_geometry_replaces_metadata_aabb():
    """Simulator-measured drawer floor / footprint / overhead
    clearance replace the link-frame metadata approximation; other slots and
    failed measurements keep their geometry."""
    import math

    from rummagebench.authoring.tasks.slots import (
        MEASURED_SOURCE, SceneSlots, Slot, SlotGeometry, apply_measured_geometry)

    def slot(name, slot_type, link):
        g = SlotGeometry((0.45, 0.42, 0.143), 0.027, (0.45, 0.42), 0.3655)
        return Slot(f"S/{name}/{link or 'x'}", "S", "kitchen_0", "kitchen", name, "c", "m",
                    "inside", slot_type, link, True, True, g, (1.3, 6.7, 0.44), math.pi / 2)
    a, b, c = (slot("cab", "drawer", "link_2"), slot("cab", "drawer", "link_3"),
               slot("cab", "cabinet_interior", None))
    ss = SceneSlots("S", 1, {}, {"kitchen_0": "kitchen"}, [a, b, c], [], {}, [])
    measured = {
        a.slot_id: {"ok": True, "floor_top_z": 0.7578, "walls_top_z": 0.87,
                    "inner_xy": [[0.95, 6.56], [1.27, 6.97]],  # world: 0.32 x 0.41
                    "clearance_m": 0.1231, "blocker": "countertop:base_link"},
        b.slot_id: {"ok": False, "reason": "no slab"},
    }
    counts = apply_measured_geometry(ss, measured, margin=0.015)
    assert counts == {"measured": 1, "failed": 1, "absent": 0}
    got = ss.by_id()
    g = got[a.slot_id].geometry
    assert g.source == MEASURED_SOURCE and g.floor_z == 0.7578
    # yaw 90 deg: world y extent is the furniture-frame x extent
    assert g.usable_extent == (0.38, 0.29, 0.1081)
    assert got[b.slot_id].geometry.usable_extent == (0.0, 0.0, 0.0)  # nothing fits
    assert got[b.slot_id].geometry.source == "sim_collision_v1_failed"
    assert got[c.slot_id].geometry == c.geometry
    # open top: the drawer's own walls bound the height
    ss2 = SceneSlots("S", 1, {}, {"kitchen_0": "kitchen"}, [a], [], {}, [])
    apply_measured_geometry(ss2, {a.slot_id: {**measured[a.slot_id], "clearance_m": None}},
                            margin=0.015)
    assert ss2.slots[0].geometry.usable_extent[2] == round(0.87 - 0.7578 - 0.015, 4)


def test_manual_priors_open_rooms_without_bddl_evidence():
    # plan E: BDDL has no bedroom evidence; a manual group approves it with
    # provenance, without inventing evidence or touching mined entries
    mined = mine_priors([("x", BDDL)], FakeTaxonomy(), RULES, {})
    rev = apply_review(mined, {"reviewer": "t", "status": "s", "approve": []})
    manual = {"reviewer": "m", "status": "s", "groups": [
        {"name": "bedroom", "room_types": ["bedroom"],
         "slot_classes": ["cabinet_storage", "table"], "categories": ["bowl"]},
        {"name": "kitchen", "room_types": ["kitchen"],
         "slot_classes": ["cabinet_storage"], "categories": ["bowl"]}]}
    out = apply_manual(rev, manual, FakeTaxonomy())
    for slot_type in ("drawer", "cabinet_interior", "table"):
        assert out.is_approved("bowl", slot_type, "bedroom")
    e = out.get("bowl", "drawer", "bedroom")
    assert (e.evidence_count, e.relation, e.compatibility) == (0, "inside", "manual")
    assert e.covers_slot_types == ["cabinet_interior", "drawer"]
    assert e.review["source"] == "manual"
    assert out.get("bowl", "table", "bedroom").relation == "on_top"
    # an existing mined key keeps its evidence and is approved on top
    k = out.get("bowl", "drawer", "kitchen")
    assert k.approved and k.evidence_count == 1
    assert not out.is_approved("bowl", "drawer", "bathroom")
    with pytest.raises(ValueError):
        apply_manual(rev, {"groups": [{"name": "x", "room_types": ["bedroom"],
                                       "slot_classes": ["table"],
                                       "categories": ["unknown_thing"]}]}, FakeTaxonomy())
