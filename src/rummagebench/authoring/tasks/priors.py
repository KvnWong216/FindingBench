"""PlacementPriors: approved (object, slot class, room type) placements.

Mined from BDDL :init facts — inside(obj, rec) / ontop(obj, rec) joined with
inroom(rec, room) — into (object SYNSET, slot class, room type) evidence.

Evidence granularity follows the SOURCE on both sides, so one BDDL fact is
counted exactly once:
  * object side: BDDL names synsets; one synset may map to several BEHAVIOR
    asset categories (chopping_board.n.01 -> chopping_board, cutting_board).
    They are aliases of ONE semantic object: the entry lists them in
    `object_categories` instead of duplicating the evidence per category.
  * receptacle side: BDDL never distinguishes a drawer from a door cabinet's
    interior (both inside(x, cabinet.n.01)), so they share one slot class
    (`prior_classes` in slot_rules); `covers_slot_types` lists them.
A reviewer narrows an approval with exclude_slot_types / exclude_categories
instead of inventing finer-grained evidence.

Four quantities are kept strictly apart:
  evidence_count     raw BDDL corpus occurrences (NOT a real-world probability)
  approved           explicit review decision (only approved entries generate)
  compatibility      semantic plausibility label (high / medium / low)
  generation_weight  the benchmark's own sampling weight
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

import yaml

PRIORS_VERSION = 3

_FACT = re.compile(r"\(\s*(inside|ontop|inroom)\s+([^\s()]+)\s+([^\s()]+)\s*\)")
_INSTANCE = re.compile(r"_\d+$")


class Taxonomy(Protocol):
    def get_categories(self, synset: str) -> list[str]: ...


@dataclass
class PriorEntry:
    object_synset: str
    slot_class: str
    room_type: str
    relation: str
    evidence_count: int
    activity_count: int
    # BEHAVIOR asset categories that are aliases of this synset
    object_categories: list[str] = field(default_factory=list)
    # concrete SceneSlots slot types this class covers (evidence is shared,
    # never duplicated across them)
    covers_slot_types: list[str] = field(default_factory=list)
    approved: bool = False
    compatibility: str = "unreviewed"  # high | medium | low | unreviewed
    generation_weight: float = 0.0
    # reviewer narrowing of an approval (e.g. ["drawer"]): evidence unchanged
    exclude_slot_types: list[str] = field(default_factory=list)
    exclude_categories: list[str] = field(default_factory=list)
    receptacle_synsets: list[str] = field(default_factory=list)
    example_activities: list[str] = field(default_factory=list)
    review: Optional[dict[str, Any]] = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.object_synset, self.slot_class, self.room_type)

    def allows(self, category: str, slot_type: str) -> bool:
        return (self.approved and self.generation_weight > 0
                and category in self.object_categories
                and category not in self.exclude_categories
                and slot_type in self.covers_slot_types
                and slot_type not in self.exclude_slot_types)


@dataclass
class PlacementPriors:
    version: int
    source: dict[str, Any]
    entries: list[PriorEntry]
    # slot type -> slot class (from slot_rules prior_classes); identity if absent
    slot_classes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self._index = {e.key: e for e in self.entries}
        self._synset_of: dict[str, str] = {}
        for e in self.entries:
            for c in e.object_categories:
                prev = self._synset_of.setdefault(c, e.object_synset)
                if prev != e.object_synset:
                    raise ValueError(f"category {c!r} under two synsets: "
                                     f"{prev}, {e.object_synset}")

    def slot_class(self, slot_type: str) -> str:
        return self.slot_classes.get(slot_type, slot_type)

    def synset_of(self, category: str) -> Optional[str]:
        return self._synset_of.get(category)

    def get(self, category: str, slot_type: str, room_type: str) -> Optional[PriorEntry]:
        """Entry covering a concrete (category, slot type) via synset + class."""
        syn = self._synset_of.get(category)
        if syn is None:
            return None
        return self._index.get((syn, self.slot_class(slot_type), room_type))

    def is_approved(self, category: str, slot_type: str, room_type: str) -> bool:
        e = self.get(category, slot_type, room_type)
        return bool(e and e.allows(category, slot_type))

    def approved_entries(self) -> list[PriorEntry]:
        return [e for e in self.entries if e.approved and e.generation_weight > 0]

    def approved_categories(self) -> set[str]:
        return {c for e in self.approved_entries() for c in e.object_categories
                if c not in e.exclude_categories}

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "source": self.source,
                "slot_classes": dict(sorted(self.slot_classes.items())),
                "entries": [asdict(e) for e in self.entries]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlacementPriors":
        return cls(version=int(d["version"]), source=dict(d.get("source", {})),
                   entries=[PriorEntry(**e) for e in d["entries"]],
                   slot_classes=dict(d.get("slot_classes") or {}))

    def save(self, path: Path, header: str = "") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + yaml.safe_dump(self.to_dict(), sort_keys=False,
                                                width=100), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PlacementPriors":
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# mining
# ---------------------------------------------------------------------------

def synset_of(instance: str) -> str:
    return _INSTANCE.sub("", instance.lstrip("?"))


def init_section(text: str) -> str:
    """The (:init ...) block only — goal facts are not placement evidence."""
    start = text.find("(:init")
    if start < 0:
        return ""
    end = text.find("(:goal", start)
    return text[start:end if end > 0 else len(text)]


def parse_init_facts(text: str) -> list[tuple[str, str, str]]:
    return [(m.group(1), m.group(2), m.group(3))
            for m in _FACT.finditer(init_section(text))]


def mine_problem(facts: list[tuple[str, str, str]]
                 ) -> list[tuple[str, str, str, str]]:
    """(relation, obj_synset, receptacle_synset, room_type) per placement.

    A receptacle's room comes from its own inroom fact; objects ontop the
    floor are not slot placements and are skipped.
    """
    room_of: dict[str, str] = {}
    for rel, a, b in facts:
        if rel == "inroom":
            room_of[a] = b
    out = []
    for rel, a, b in facts:
        if rel == "inroom" or b not in room_of:
            continue
        rec_syn = synset_of(b)
        if rec_syn.startswith("floor.") or rec_syn.startswith("agent."):
            continue
        out.append(("inside" if rel == "inside" else "on_top",
                    synset_of(a), rec_syn, room_of[b]))
    return out


def slot_classes_of(rules: dict[str, Any]) -> dict[str, str]:
    return dict(rules.get("prior_classes") or {})


def receptacle_slot_classes(rules: dict[str, Any], taxonomy: Taxonomy,
                            rec_synset: str, relation: str) -> dict[str, set[str]]:
    """BDDL receptacle synset -> {slot class: concrete slot types it covers},
    via the furniture categories it maps to (slot_rules furniture table)."""
    furn = rules["furniture"]
    classes = slot_classes_of(rules)
    types: set[str] = set()
    for cat in taxonomy.get_categories(rec_synset) or []:
        rule = furn.get(cat)
        if not rule:
            continue
        if relation == "inside" and rule.get("container"):
            types |= {"drawer", "cabinet_interior"}
        if relation == "on_top" and rule.get("surface"):
            types.add(rule["surface"])
    out: dict[str, set[str]] = {}
    for t in types:
        out.setdefault(classes.get(t, t), set()).add(t)
    return out


def mine_priors(problems: Iterable[tuple[str, str]], taxonomy: Taxonomy,
                rules: dict[str, Any], source: dict[str, Any]) -> PlacementPriors:
    """problems: (activity_name, bddl_text). One fact counts ONCE per slot
    class, however many concrete slot types that class covers."""
    counts: dict[tuple, int] = defaultdict(int)
    acts: dict[tuple, set[str]] = defaultdict(set)
    meta: dict[tuple, dict[str, Any]] = {}
    for activity, text in problems:
        for relation, obj_syn, rec_syn, room in mine_problem(parse_init_facts(text)):
            by_class = receptacle_slot_classes(rules, taxonomy, rec_syn, relation)
            cats = taxonomy.get_categories(obj_syn) or []
            if not cats:
                continue  # abstract synset with no asset category
            for cls, covers in by_class.items():
                key = (obj_syn, cls, room)
                counts[key] += 1
                acts[key].add(activity)
                m = meta.setdefault(key, {"relation": relation, "categories": set(),
                                          "receptacles": set(), "covers": set()})
                m["categories"] |= set(cats)
                m["receptacles"].add(rec_syn)
                m["covers"] |= covers
    entries = []
    for key in sorted(counts):
        syn, cls, room = key
        m = meta[key]
        entries.append(PriorEntry(
            object_synset=syn, slot_class=cls, room_type=room,
            relation=m["relation"], evidence_count=counts[key],
            activity_count=len(acts[key]), object_categories=sorted(m["categories"]),
            covers_slot_types=sorted(m["covers"]),
            receptacle_synsets=sorted(m["receptacles"]),
            example_activities=sorted(acts[key])[:5]))
    return PlacementPriors(version=PRIORS_VERSION, source=source, entries=entries,
                           slot_classes=slot_classes_of(rules))


def apply_review(mined: PlacementPriors, review: dict[str, Any]) -> PlacementPriors:
    """Merge an explicit review table into mined evidence.

    review = {reviewer, status, compatibility_from_evidence: {high: n, medium: m},
              approve: [{object_synset, slot_class, room_type, weight?,
                         exclude_slot_types?, exclude_categories?, note?,
                         categories? / covers_slot_types? (informational)}]}
    Only listed (synset, slot_class, room_type) triples are approved; every
    approval must be backed by mined evidence (no invented priors).
    """
    thresholds = review.get("compatibility_from_evidence", {"high": 10, "medium": 3})
    approve = {(a["object_synset"], a["slot_class"], a["room_type"]): a
               for a in review.get("approve", [])}
    missing = [k for k in approve if k not in {e.key for e in mined.entries}]
    if missing:
        raise ValueError(f"approved priors without BDDL evidence: {missing}")
    out = []
    for e in mined.entries:
        e = PriorEntry(**asdict(e))
        e.compatibility = ("high" if e.evidence_count >= thresholds["high"] else
                           "medium" if e.evidence_count >= thresholds["medium"] else "low")
        a = approve.get(e.key)
        if a is not None:
            excl = sorted(a.get("exclude_slot_types") or [])
            bad = set(excl) - set(e.covers_slot_types)
            if bad:
                raise ValueError(f"{e.key}: exclude_slot_types {sorted(bad)} not "
                                 f"covered by {e.covers_slot_types}")
            excl_c = sorted(a.get("exclude_categories") or [])
            bad = set(excl_c) - set(e.object_categories)
            if bad:
                raise ValueError(f"{e.key}: exclude_categories {sorted(bad)} not "
                                 f"in {e.object_categories}")
            e.approved = True
            e.generation_weight = float(a.get("weight", 1.0))
            e.exclude_slot_types = excl
            e.exclude_categories = excl_c
            e.review = {"reviewer": review.get("reviewer"),
                        "status": review.get("status"), "note": a.get("note")}
        out.append(e)
    source = dict(mined.source)
    source["review"] = {"reviewer": review.get("reviewer"),
                        "status": review.get("status")}
    return PlacementPriors(version=mined.version, source=source, entries=out,
                           slot_classes=dict(mined.slot_classes))
