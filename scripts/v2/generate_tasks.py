"""Generate vla_difficulty_v2 base tasks from certified locations (CPU).

Inputs
  runs/vla_difficulty_v2/_survey/<scene>.candidates.json   starts (analyze_survey.py)
  runs/vla_difficulty_v2/_locations/<scene>.json           certified locations (certify_locations.py)
  build/vla_difficulty_v2/object_pool_candidates.json      salient graspable models (score_object_pool.py)
  assets/tasks/vla_difficulty_v2/phrases.yaml              hand-checked room / furniture names

For every start pose, the spatial conditions are drawn from the probe
certification (the real object is re-gated per variant by run_variant.py):

  S1  on_top location in the start's room, probe RGB-visible from the start
  S2  drawer location in the start's room, invisible from the start, hidden
      AND un-graspable while closed, graspable once open
  S3  on_top location in another room, invisible from the start, with a
      MOVE/TURN route (CPU plan) from the start to a certified grasp pose

Families (goal identical across L and S):
  fetch         holding(target)
  place_on      on_top(target, G)   G: certified PLACE surface, not an S1/S3 surface
  place_in      inside(target, G)   G: certified PLACE container, not the S2 cabinet

Every combination is enumerated; base tasks are picked greedily for novelty
(start, locations, goal, object category) with the three families balanced.

    python scripts/v2/generate_tasks.py --count 125
Writes assets/tasks/vla_difficulty_v2/base_tasks/B###.yaml and generation_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visibility import S1_MIN_PX  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SURVEY = ROOT / "runs/vla_difficulty_v2/_survey"
LOCS = ROOT / "runs/vla_difficulty_v2/_locations"
POOL = ROOT / "build/vla_difficulty_v2/object_pool_candidates.json"
DESIGN = ROOT / "assets/tasks/vla_difficulty_v2"
MAX_ROUTE_STEPS = 25

# category -> (noun phrase, intent group, L3 need phrase)
OBJECTS = {
    "apple": ("apple", "fruit", "something fresh to snack on"),
    "orange": ("orange", "fruit", "a piece of fruit to snack on"),
    "lemon": ("lemon", "sour", "something sour for my tea"),
    "lime": ("lime", "sour", "something sour for my drink"),
    "pear": ("pear", "fruit", "something fresh to snack on"),
    "banana": ("banana", "fruit", "a quick snack"),
    "bell_pepper": ("bell pepper", "vegetable", "a vegetable for the salad"),
    "mango": ("mango", "fruit", "some fruit for my smoothie"),
    "peach": ("peach", "fruit", "a piece of fruit"),
    "avocado": ("avocado", "vegetable", "the avocado for my toast"),
    "mug": ("mug", "hot_drink", "something to drink my coffee from"),
    "teacup": ("teacup", "hot_drink", "something to drink my tea from"),
    "coffee_cup": ("coffee cup", "hot_drink", "something to hold my coffee"),
    "bowl": ("bowl", "dish", "something to eat my cereal from"),
    "can_of_soda": ("can of soda", "cold_drink", "a fizzy drink"),
    "water_bottle": ("water bottle", "cold_drink", "some water to drink"),
    "bottle_of_water": ("bottle of water", "cold_drink", "some water to drink"),
    "bottle_of_shampoo": ("bottle of shampoo", "toiletry", "something to wash my hair with"),
    "spray_bottle": ("spray bottle", "cleaning", "something to spray the plants with"),
    "box_of_cereal": ("box of cereal", "breakfast", "something for breakfast"),
    "carton_of_milk": ("carton of milk", "dairy", "some milk for my coffee"),
    "box_of_crackers": ("box of crackers", "salty_snack", "a salty snack"),
    "teddy_bear": ("teddy bear", "toy", "something for my daughter to cuddle"),
    "toy_car": ("toy car", "toy", "something for my son to play with"),
    "toy_figure": ("toy figure", "toy", "something for my kid to play with"),
    "tennis_ball": ("tennis ball", "ball", "something to play fetch with the dog"),
    "baseball": ("baseball", "ball", "something to play catch with"),
    "softball": ("softball", "ball", "something to play catch with"),
    "alarm_clock": ("alarm clock", "clock", "something to wake me up tomorrow"),
    "sponge": ("sponge", "cleaning", "something to wipe the spill with"),
    "pencil_case": ("pencil case", "school", "something to take my pens to class in"),
    "lunch_box": ("lunch box", "school", "something to pack my lunch in"),
}


def load_scene(scene: str) -> dict | None:
    lp = LOCS / f"{scene}.json"
    if not lp.exists():
        return None
    locs = json.loads(lp.read_text())
    gp = LOCS / f"{scene}.goals.json"   # goal phase re-run (certify_locations.py --only-goals)
    if gp.exists():
        locs["goals"] = json.loads(gp.read_text())["goals"]
    cands = json.loads((SURVEY / f"{scene}.candidates.json").read_text())
    survey = json.loads((SURVEY / f"{scene}.json").read_text())
    return {"scene": scene, "locs": locs, "starts": cands["starts"], "survey": survey}


class RouteEstimator:
    """Cheap route check on the survey's reachable-floor grid: a grasp pose is
    reachable when its cell is connected to the start's cell; the step count
    is estimated as geodesic distance / MOVE max (1 m) + 3 turns. The exact
    MOVE / TURN route is found by the route oracle in run_variant.py."""

    def __init__(self, chk, survey: dict, scene: str):
        import numpy as np
        from analyze_survey import KEEP_SURVEY_START, largest_component_seed, reachable
        boxes = [e["aabb"] for e in survey["entities"].values() if e.get("aabb")]
        lo = np.min([b[0][:2] for b in boxes], 0) - 0.5
        hi = np.max([b[1][:2] for b in boxes], 0) + 0.5
        seed = survey["start"][:2]
        if scene not in KEEP_SURVEY_START:
            seed = largest_component_seed(chk, (lo, hi), seed, scene)
        self.xs, self.ys, self.reach = reachable(chk, seed, (lo, hi))
        self.cache: dict = {}

    def _cell(self, x, y):
        i = int(round((x - self.xs[0]) / (self.xs[1] - self.xs[0])))
        j = int(round((y - self.ys[0]) / (self.ys[1] - self.ys[0])))
        ok = 0 <= i < len(self.xs) and 0 <= j < len(self.ys) and self.reach[i, j]
        return (i, j) if ok else None

    def _dist(self, start):
        from collections import deque
        import numpy as np
        if start in self.cache:
            return self.cache[start]
        d = np.full(self.reach.shape, np.inf)
        d[start] = 0
        q = deque([start])
        while q:
            i, j = q.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (i + di, j + dj)
                if 0 <= n[0] < d.shape[0] and 0 <= n[1] < d.shape[1] and self.reach[n] and d[n] == np.inf:
                    d[n] = d[i, j] + 1
                    q.append(n)
        self.cache[start] = d
        return d

    def steps(self, start: dict, poses: list) -> int | None:
        s = self._cell(start["x"], start["y"])
        if s is None:
            return None
        d = self._dist(s)
        best = min((d[c] for c in (self._cell(p[0], p[1]) for p in poses) if c is not None), default=None)
        if best is None or best == float("inf"):
            return None
        grid = float(self.xs[1] - self.xs[0])
        return int(math.ceil(best * grid / 1.0)) + 3


def enumerate_combos(sd: dict, phrases: dict, allow_partial=False) -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts/v2"))
    from analyze_survey import checker_from
    from rummagebench.authoring.tasks.room_map import RoomMap
    root = Path(os.environ["OMNIGIBSON_DATA_PATH"]) / "behavior-1k-assets"
    rm = RoomMap.load(root / "scenes" / sd["scene"], root / "metadata" / "room_categories.txt")
    chk = checker_from(sd["survey"])
    routes = RouteEstimator(chk, sd["survey"], sd["scene"])
    names = phrases.get(sd["scene"], {})
    furn = names.get("furniture", {})

    def ok_target(t):
        return t.get("relation_holds") and (t.get("grasp") or {}).get("ok")

    targets = [t for t in sd["locs"]["targets"] if ok_target(t)]
    goals = [g for g in sd["locs"]["goals"] if (g.get("place") or {}).get("ok") and g["receptacle"] in furn
             and furn[g["receptacle"]].get("goal", True)]
    combos = []
    for s in sd["starts"]:

        def vis(t, key="vis_closed"):
            return (t.get(key) or {}).get(s["name"], {"px": 0, "delta_e": None})

        # the probe mug is grey-white, so its contrast says nothing about the real
        # (salient) target: prefilter on pixels only; run_variant.py gates contrast
        s1 = [t for t in targets if t["relation"] == "on_top" and t["probe_room"] == s["room"]
              and vis(t)["px"] >= S1_MIN_PX]
        # S2 = same room, not visible from the start: inside a closed drawer
        # ("container", needs OPEN) or on a surface hidden behind furniture
        # ("occluded", needs moving / looking around). Containers are preferred.
        s2 = [{**t, "s2_kind": "container"} for t in targets
              if t["relation"] == "inside" and t["probe_room"] == s["room"]
              and vis(t)["px"] == 0 and t.get("at_pose_closed_px") == 0
              and t.get("at_pose_closed_grasp_feasible") is False
              and t.get("link")]
        if not s2:
            s2 = [{**t, "s2_kind": "occluded"} for t in targets
                  if t["relation"] == "on_top" and t["probe_room"] == s["room"] and vis(t)["px"] == 0]
        s3 = []
        for t in targets:
            if t["relation"] == "on_top" and t["probe_room"] not in (s["room"], None) and vis(t)["px"] == 0:
                steps = routes.steps(s, t["grasp"]["feasible_poses"])
                if steps is not None and steps <= MAX_ROUTE_STEPS:
                    s3.append({**t, "route_from_start": steps})
        if allow_partial:
            if not any((s1,s2,s3)):
                continue
            s1,s2,s3 = s1 or [None], s2 or [None], s3 or [None]
        for a in s1:
            for b in s2:
                if a and b and b["receptacle"] == a["receptacle"]:
                    continue
                for c in s3:
                    used = {t["receptacle"] for t in (a,b,c) if t}
                    combos.append({"scene": sd["scene"], "start": s, "S1": a, "S2": b, "S3": c,
                                   "family": "fetch", "goal": None})
                    for g in goals:
                        if g["relation"] == "on_top" and g["receptacle"] not in used:
                            combos.append({"scene": sd["scene"], "start": s, "S1": a, "S2": b, "S3": c,
                                           "family": "place_on", "goal": g})
                        if g["relation"] == "inside" and g["receptacle"] not in used:
                            combos.append({"scene": sd["scene"], "start": s, "S1": a, "S2": b, "S3": c,
                                           "family": "place_in", "goal": g})
    return combos


def room_name(phrases, scene, room):
    return phrases[scene]["rooms"][room]


def furniture_phrase(phrases, scene, rec, room):
    return f"{phrases[scene]['furniture'][rec]['noun']} in the {room_name(phrases, scene, room)}"


def location_phrase(phrases, scene, t) -> str | None:
    furniture=phrases[scene]['furniture'].get(t['receptacle'])
    if furniture is None or t['probe_room'] not in phrases[scene]['rooms']:
        return None
    if t['relation']=='inside' and t.get('link') not in furniture.get('drawers',{}):
        return None
    base = furniture_phrase(phrases, scene, t["receptacle"], t["probe_room"])
    if t["relation"] == "inside":
        drawer = phrases[scene]["furniture"][t["receptacle"]]["drawers"][t["link"]]
        return f"the {drawer} of the {base}"
    return f"the {base}"


def goal_phrase(phrases, scene, g, room) -> str:
    noun = furniture_phrase(phrases, scene, g["receptacle"], room)
    return f"in the {noun}" if g["relation"] == "inside" else f"on the {noun}"


NOUN_FIX = {"hardback": "hardback book", "paperback_book": "paperback book", "notebook": "notebook"}


def noun_of(category: str) -> str:
    if category in NOUN_FIX:
        return NOUN_FIX[category]
    if category in OBJECTS:
        return OBJECTS[category][0]
    return category.replace("_", " ")


def activity_intent(activity: str) -> str:
    """Official activity name as the user's ongoing high-level activity."""
    words = activity.split("#")[0].replace("_", " ")
    return f"I'm {words}" if words.split()[0].endswith("ing") else f"I'm working on {words}"


def language(c: dict, obj: dict, phrases: dict, rng: random.Random, activity: str) -> dict:
    """L1 explicit steps (per S), L2 goal only, L3 the official activity as intent."""
    scene, noun = c["scene"], noun_of(obj["category"])
    loc = {S: location_phrase(phrases, scene, c[S]) for S in ("S1", "S2", "S3") if c.get(S)}
    loc = {S:text for S,text in loc.items() if text is not None}
    s_room = phrases[scene]['rooms'].get(c['start']['room'])
    container = bool(c.get('S2')) and c['S2']['s2_kind'] == 'container'
    open_step = {S: (f"open {loc[S]}, " if S == 'S2' and container else '') for S in loc}
    go = {S: (f"Leave the {s_room}, go to {loc[S]}" if S == 'S3' and s_room else
               f"Go to the {loc[S].split(' of the ', 1)[1]}" if S == 'S2' and container else
               f"Go to {loc[S]}") for S in loc}
    where = {S: "inside" if S == "S2" and container else "on it" for S in go}
    intent = activity_intent(activity)
    if c["family"] == "fetch":
        l1 = {S: f"{go[S]}, {open_step[S]}pick up the {noun} {where[S]} and hold it." for S in go}
        l2 = rng.choice([f"Pick up the {noun} and keep holding it.", f"Get the {noun} and hold on to it.",
                         f"Find the {noun} and hold it."])
        l3 = rng.choice([f"{intent} and I need the {noun}. Could you get it for me?",
                         f"{intent}. Can you find the {noun} and hold it for me?"])
    else:
        room_g = c["goal"]["room"]
        gp = goal_phrase(phrases, scene, c["goal"], room_g)
        l1 = {S: f"{go[S]}, {open_step[S]}pick up the {noun} {where[S]}. "
                 f"Then put it {gp}." for S in go}
        if c["family"] == "place_in":
            # the hand must be free to pull a drawer: the goal container is opened before grasping
            first = f"First go and open the {furniture_phrase(phrases, scene, c['goal']['receptacle'], room_g)}. "
            l1 = {S: first + l1[S] for S in go}
        l2 = rng.choice([f"Put the {noun} {gp}.", f"Place the {noun} {gp}."])
        if c["family"] == "place_on":
            l3 = rng.choice([f"{intent}. The {noun} should end up {gp} -- could you take care of it?",
                             f"{intent}, so could you make sure the {noun} is {gp}?"])
        else:
            l3 = rng.choice([f"{intent}, and the {noun} belongs {gp}. Could you put it away?",
                             f"{intent}. Could you store the {noun} {gp} for me?"])
    return {"L1": l1, "L2": l2, "L3": l3}


REL = {"on_top": "ontop", "inside": "inside"}
# goal containers the robot cannot OPEN itself (route oracle found no feasible OPEN
# pose; the certification opened them with set_open). place_in needs a real OPEN.
OPEN_INFEASIBLE = {("house_double_floor_upper", "bottom_cabinet_jhymlr_0")}   # runs/vla_difficulty_v2 U001, 2026-10-26


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--count", type=int, default=125)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenes", nargs="*", default=None, help="default: every certified scene")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--exclude-models", type=Path, help="file of category/model lines that failed in "
                   "simulation (run_batch.py --failed-objects); never chosen again")
    p.add_argument("--prefix", default="B", help="task id prefix; only <prefix>###.yaml files are replaced, "
                   "so scenes can be added in separate batches without renumbering earlier tasks")
    p.add_argument('--allow-partial', action='store_true', help='Keep available S conditions; never fabricate missing ones')
    p.add_argument('--diverse-objects', action='store_true', help='Use the full accepted rigid object pool; runtime gates still apply')
    a = p.parse_args()
    if not a.dry_run and list((DESIGN / 'base_tasks').glob(f'{a.prefix}[0-9][0-9][0-9].yaml')):
        p.error('Task prefix already exists; choose a fresh prefix to preserve prior data')
    rng = random.Random(a.seed)
    phrases = yaml.safe_load((DESIGN / "phrases.yaml").read_text())
    pairs = json.loads((ROOT / "build/vla_difficulty_v2/official_pairs.json").read_text())["pairs"]
    # probe-like size (locations were certified with a ~12 cm mug): larger objects leak
    # into view in S3 and fail PLACE more often; dull ones fail the S1 contrast gate
    pool = [m for m in json.loads((ROOT / "build/vla_difficulty_v2/object_pool_official.json").read_text())["models"]
            if m["accepted"] and (a.diverse_objects or (max(m["bbox_size"]) <= 0.16 and m["bbox_size"][2] <= 0.16
            and min(m["bbox_size"]) >= 0.05 and (m.get("colourfulness") or 0) >= 40))]
    from bddl.object_taxonomy import ObjectTaxonomy
    taxonomy = ObjectTaxonomy()
    def rigid(category):
        synset = taxonomy.get_synset_from_category(category)
        return bool(synset and 'rigidBody' in taxonomy.get_abilities(synset))
    pool = [m for m in pool if rigid(m['category'])]
    exclude = set()
    if a.exclude_models:
        exclude = {tuple(x.split("/")) for x in a.exclude_models.read_text().split()}
        pool = [m for m in pool if (m["category"], m["model"]) not in exclude]
    by_cat: dict[str, list] = {}
    for m in pool:
        by_cat.setdefault(m["category"], []).append(m)
    scenes = a.scenes or sorted(p.stem for p in LOCS.glob("*.json") if "." not in p.stem)
    combos = []
    report: dict = {"scenes": {}}
    natives: dict[str, set] = {}
    for scene in scenes:
        sd = load_scene(scene)
        if sd is None or scene not in phrases:
            report["scenes"][scene] = "not certified / no phrases"
            continue
        natives[scene] = {e["category"] for e in sd["survey"]["entities"].values() if not e.get("fixed_base")}
        cs = enumerate_combos(sd, phrases, a.allow_partial)
        for c in cs:
            if c["goal"]:
                c["goal"]["category"] = sd["survey"]["entities"][c["goal"]["receptacle"]]["category"]
        report["scenes"][scene] = {"combos": len(cs), "by_family": dict(Counter(c["family"] for c in cs))}
        combos += cs

    option_cache: dict = {}

    def official_options(c):
        """(pair, object category) options: official sub-goals matching this combo."""
        key = (c["scene"], c["family"], (c["goal"] or {}).get("relation"), (c["goal"] or {}).get("category"))
        if key in option_cache:
            return option_cache[key]
        out = option_cache[key] = []
        for pr in pairs:
            if c["family"] != "fetch":
                if pr["relation"] != REL[c["goal"]["relation"]] or c["goal"]["category"] not in pr["receptacle_categories"]:
                    continue
            for oc in pr["object_categories"]:
                if oc in by_cat and oc not in natives[c["scene"]]:
                    out.append((pr, oc))
        return out

    combos = [c for c in combos if official_options(c) and not (
        c["family"] == "place_in" and (c["scene"], c["goal"]["receptacle"]) in OPEN_INFEASIBLE)]
    report["combos_with_official_subgoal"] = len(combos)
    print(json.dumps(report, indent=1))
    rng.shuffle(combos)
    chosen: list[tuple] = []
    seen: Counter = Counter()
    # Balance novelty against the retained corpus as well as this batch.
    for existing in sorted((DESIGN / 'base_tasks').glob('*.yaml')):
        old=yaml.safe_load(existing.read_text())
        seen[('obj',old['target']['category'])]+=1
        seen[('asset',old['target']['category'],old['target']['model'])]+=1

    fam_quota = {f: math.ceil(a.count / 3) for f in ("fetch", "place_on", "place_in")}
    fam_count: Counter = Counter()

    def keys(c):
        g = c["goal"]["receptacle"] if c["goal"] else None
        return [("start", c["scene"], c["start"]["name"]), ("S1", c["scene"], (c.get("S1") or {}).get("receptacle")),
                ("S2", c["scene"], (c.get("S2") or {}).get("receptacle"), (c.get("S2") or {}).get("link")), ("S3", c["scene"], (c.get("S3") or {}).get("receptacle")),
                ("goal", c["scene"], g), ("scene", c["scene"])]

    used_combo: Counter = Counter()
    while len(chosen) < a.count and combos:
        open_fams = [f for f in fam_quota if fam_count[f] < fam_quota[f]]
        cand = [c for c in combos if c["family"] in open_fams] or combos
        best = min(cand, key=lambda c: (used_combo[id(c)], sum(seen[k] for k in keys(c))))
        opts = official_options(best)
        pr, oc = min(opts, key=lambda o: (seen[("activity", o[0]["activities"][0])] + seen[("obj", o[1])],
                                          rng.random()))
        sig = (best["scene"], best["start"]["name"], (best.get("S1") or {}).get("receptacle"), (best.get("S2") or {}).get("receptacle"),
               (best.get("S3") or {}).get("receptacle"), best["family"], (best["goal"] or {}).get("receptacle"), oc)
        used_combo[id(best)] += 1
        if seen[("sig",) + sig]:
            if all(used_combo[id(c)] > len(official_options(c)) for c in cand):
                break
            continue
        seen[("sig",) + sig] += 1
        chosen.append((best, pr, oc))
        fam_count[best["family"]] += 1
        for k in keys(best) + [("activity", pr["activities"][0]), ("obj", oc)]:
            seen[k] += 1

    out_dir = DESIGN / "base_tasks"
    if not a.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob(f"{a.prefix}[0-9][0-9][0-9].yaml"):
            old.unlink()
    rows = []
    for i, (c, pr, cat) in enumerate(chosen):
        # most colourful models first: the S1 RGB-contrast gate rejects dull ones
        models = sorted(by_cat[cat], key=lambda m: -(m.get("colourfulness") or 0))
        model = min(models, key=lambda m: seen[("asset",cat,m["model"])])
        seen[("asset",cat,model["model"])]+=1
        seen[("model", cat)] += 1
        tid = f"{a.prefix}{i + 1:03d}"
        obj = {"entity": "target_obj", "category": cat, "model": model["model"]}
        activity = pr["activities"][0]
        lang = language(c, obj, phrases, rng, activity)
        spatial = {}
        for S in ("S1", "S2", "S3"):
            t = c[S]
            if t is None:
                continue
            pl = {"entity": "target_obj", "relation": t["relation"], "receptacle": t["receptacle"]}
            if t.get("link"):
                pl["link"] = t["link"]
            spatial[S] = {"placement": pl, "room": t["probe_room"],
                          "location_phrase": location_phrase(phrases, c["scene"], t),
                          "grasp_hints": t["grasp"]["feasible_poses"][:4],
                          "extra_ops_vs_S1": {"S1": [], "S2": ["OPEN"] if (c.get("S2") or {}).get("s2_kind") == "container"
                                              else ["search behind occluders"], "S3": ["cross-room navigation"]}[S],
                          **({"s2_kind": c["S2"]["s2_kind"]} if S == "S2" else {}),
                          "probe_visibility_from_start": (t.get("vis_closed") or {}).get(c["start"]["name"]),
                          "task_scope": 'single_room' if t['probe_room']==c['start']['room'] and
                              (not c['goal'] or c['goal']['room']==c['start']['room']) else 'cross_room'}
            if S == "S3":
                spatial[S]["route_from_start_steps"] = t["route_from_start"]
        goal = [{"kind": "holding", "entity": "target_obj"}] if c["family"] == "fetch" else \
            [{"kind": "on_top" if c["family"] == "place_on" else "inside", "entity": "target_obj",
              "receptacle": c["goal"]["receptacle"]}]
        initial = {(c.get("S2") or {}).get("receptacle"): {"open": False}} if (c.get("S2") or {}).get("s2_kind") == "container" else {}
        if c["family"] == "place_in":
            initial[c["goal"]["receptacle"]] = {"open": False}
        task = {"task_id": tid, "family": c["family"], "scene": c["scene"],
                "source": {
                    "official_activity": activity, "official_activities_with_same_subgoal": pr["activities"][:10],
                    "official_subgoal": f"({pr['relation']} {pr['object_synset']} {pr['receptacle_synset']})",
                    "reuse": ("official sub-goal re-used as the task goal" if c["family"] != "fetch" else
                              "official sub-goal object; goal = holding (the fetch half of the sub-goal, not an "
                              "official terminal predicate)"),
                    "scene": f"BEHAVIOR-1K {c['scene']}" + (" (+ clearance manifest)" if (ROOT / "build/vla_difficulty_v2/clearance" / f"{c['scene']}.json").exists() else ""),
                    "object": f"BEHAVIOR-1K {cat}/{model['model']}"},
                "start": {k: c["start"][k] for k in ("name", "room", "x", "y", "yaw")},
                "target": {**obj, "noun": noun_of(cat), "colour_estimate": model.get("colour")},
                "goal": goal, "place_hints": (c["goal"] or {}).get("place", {}).get("feasible_poses", [])[:4],
                "background_objects": [], "background_placements": [],
                "available_spatial_levels": list(spatial),
                "missing_spatial_levels": [S for S in ("S1","S2","S3") if S not in spatial],
                "initial_states": initial, "spatial": spatial, "language": lang,
                "available_language_levels": {S: [L for L in ('L1','L2','L3') if L != 'L1' or S in lang['L1']] for S in spatial},
                "description_provenance": {'missing_L1_reason':'source location lacks an unambiguous description',
                    'missing_L1_spatial_levels':[S for S in spatial if S not in lang['L1']]},
                "l3_ambiguity_check": {"object_category_unique_in_scene": cat not in natives[c["scene"]],
                                       "note": "the only movable task object is the target; L3 names it"}}
        rows.append({"task_id": tid, "scene": c["scene"], "family": c["family"], "object": cat,
                     "activity": activity, "start": c["start"]["name"]})
        if not a.dry_run:
            (out_dir / f"{tid}.yaml").write_text(yaml.safe_dump(task, sort_keys=False, allow_unicode=True))
    report["chosen"] = len(chosen)
    report["by_scene"] = dict(Counter(r["scene"] for r in rows))
    report["by_family"] = dict(Counter(r["family"] for r in rows))
    report["distinct_objects"] = len({r["object"] for r in rows})
    report["distinct_activities"] = len({r["activity"] for r in rows})
    if not a.dry_run:
        (DESIGN / f"generation_report_{a.prefix}.json").write_text(json.dumps({**report, "tasks": rows}, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("chosen", "by_scene", "by_family", "distinct_objects",
                                            "distinct_activities")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
