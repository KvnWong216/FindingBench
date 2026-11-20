"""Classify official BEHAVIOR-1K rigid task definitions along the
vla_difficulty_v1 axes (CPU only, reads the local BDDL copy).

For every definition the rigid audit accepts (datasets/behavior.py), derive
from :init / :goal only — no simulation:

  goal_kinds        goal predicates used (ontop/inside/open/nextto/under/...)
  quantified        goal uses forall/exists/forn/forpairs (multi-object)
  goal_movables     first arguments of relocation goal literals
  hidden_at_init    a goal movable starts INSIDE an openable container
                    (BDDL proxy for our S2: occluded / local search)
  rooms             distinct rooms in :init inroom facts (>1: proxy for S3)
  open_required     goal requires (open X) or a movable starts in a closed
                    openable (OPEN needed before GRASP)
  goal_needs_inside_furniture
                    goal inside(?, openable scene furniture) — PLACE currently
                    supports top surfaces only (see docs/phaseB_paper_tasks.md)
  closest_family    single_fetch_place / multi_object_place / state_only / other
  local_instances   pre-sampled instances installed on this host

    python scripts/analyze_official_tasks.py \
        --catalog build/behavior/official_catalog.json \
        --output build/behavior/official_task_axes.json
"""
from __future__ import annotations

import argparse
import json
import csv
import math
import hashlib
from bisect import bisect_left, bisect_right
from collections import Counter
from pathlib import Path

from rummagebench.datasets.behavior import parse_bddl

RELOCATION = {"ontop", "inside", "nextto", "under", "onfloor", "touching", "overlaid"}
QUANT = {"forall", "exists", "forn", "forpairs", "fornpairs"}


def _literals(node, out, negated=False):
    if not isinstance(node, list) or not node:
        return
    head = node[0]
    if head == "not":
        for c in node[1:]:
            _literals(c, out, not negated)
        return
    if head in QUANT or head in {"and", "or", "imply"}:
        out.append(("__quant__",) if head in QUANT else ("__logic__",))
        for c in node[1:]:
            _literals(c, out, negated)
        return
    if isinstance(head, str) and (head.startswith("?") or head.isdigit()):
        for c in node[1:]:  # quantifier variable declaration / forn count
            _literals(c, out, negated)
        return
    if isinstance(head, str) and all(isinstance(x, str) for x in node):
        out.append((("not " if negated else "") + head, *[a.lstrip("?") for a in node[1:]]))
        return
    for c in node[1:]:
        _literals(c, out, negated)


def _synset(inst: str) -> str:
    return inst.rsplit("_", 1)[0] if inst.rsplit("_", 1)[-1].isdigit() else inst


def analyze(task: dict, annotations: dict) -> dict:
    tree = parse_bddl(Path(task["definition"]).read_text())
    sec = {n[0]: n[1:] for n in tree if isinstance(n, list) and n}
    init, goal = [], []
    for e in sec.get(":init", []):
        _literals(e, init)
    for e in sec.get(":goal", []):
        _literals(e, goal)
    quantified = any(l[0] == "__quant__" for l in goal)
    goal = [l for l in goal if not l[0].startswith("__")]
    init = [l for l in init if not l[0].startswith("__")]
    kinds = sorted({l[0] for l in goal})
    movables = sorted({l[1] for l in goal if l[0] in RELOCATION and len(l) > 1})
    openable = lambda inst: "openable" in annotations.get(_synset(inst), {})
    init_inside = {l[1]: l[2] for l in init if l[0] == "inside" and len(l) > 2}
    init_open = {l[1] for l in init if l[0] == "open"}
    hidden = sorted(m for m in movables
                    if m in init_inside and openable(init_inside[m]) and init_inside[m] not in init_open)
    # quantified goals name synset variables, match by synset too
    hidden_syn = sorted({_synset(o) for o, c in init_inside.items()
                         if openable(c) and c not in init_open
                         and _synset(o) in {_synset(m) for m in movables}})
    rooms = sorted({l[2] for l in init if l[0] == "inroom" and len(l) > 2})
    open_goal = any(l[0] == "open" for l in goal)
    inside_furn = any(l[0] == "inside" and len(l) > 2 and openable(l[2])
                      and "sceneObject" in annotations.get(_synset(l[2]), {}) for l in goal)
    reloc = [l for l in goal if l[0] in RELOCATION]
    if not reloc:
        family = "state_only"
    elif len({_synset(l[1]) for l in reloc}) == 1 and not quantified and len(reloc) == 1:
        family = "single_fetch_place"
    else:
        family = "multi_object_place"
    return {
        "activity": task["activity_name"], "definition_id": task["activity_definition_id"],
        "goal_kinds": kinds, "goal_literals": len(goal), "quantified": quantified,
        "goal_movables": movables, "hidden_at_init": hidden or hidden_syn,
        "rooms": rooms, "multi_room": len(rooms) > 1,
        "open_required": bool(open_goal or hidden or hidden_syn),
        "goal_needs_inside_furniture": inside_furn,
        "closest_family": family,
        "local_instances": [f"{i['scene_model']}#{i['activity_instance_id']}" for i in task["instances"]],
    }



# Evaluator-only geometric and structural priors. Never loaded by the harness.
def declared_objects(section):
    result, pending = {}, []
    i = 0
    while i < len(section):
        if section[i] == '-':
            result.update({name: section[i + 1] for name in pending})
            pending = []
            i += 2
        else:
            pending.append(section[i])
            i += 1
    if pending:
        raise ValueError('Untyped BDDL objects')
    return result


def graph_choice(edges, object_nodes, root='house'):
    """User-defined cost on a deduplicated directed graph, averaged over objects.

    Positive costs allow Dijkstra even with cycles, without modifying the graph
    by contracting physical objects into artificial nodes.
    """
    import heapq
    adjacency = {}
    for u,v in set(tuple(e) for e in edges):
        if u != v:
            adjacency.setdefault(u,set()).add(v)
    distances, paths = {root:0.0},{root:[root]}
    queue = [(0.0,root)]
    while queue:
        cost,u = heapq.heappop(queue)
        if cost != distances[u]: continue
        children = adjacency.get(u,set())
        weight = 1+math.log2(max(1,len(children)))
        for v in sorted(children):
            candidate = cost+weight
            if candidate < distances.get(v,math.inf):
                distances[v] = candidate
                paths[v] = paths[u]+[v]
                heapq.heappush(queue,(candidate,v))
    missing = sorted(set(object_nodes)-set(distances))
    details = [dict(object=v,cost=distances[v],depth=len(paths[v])-1,path=paths[v])
               for v in sorted(set(object_nodes)) if v in distances]
    return dict(choice=sum(x['cost'] for x in details)/len(details) if details and not missing else None,
                mean_path_depth=sum(x['depth'] for x in details)/len(details) if details else None,
                object_count=len(set(object_nodes)),unreachable_objects=missing,paths=details,
                edges=sorted([u,v] for u,children in adjacency.items() for v in children))


def snapshot_graph(data, initial, geometry):
    """All native scene objects/room memberships plus recorded BDDL relations.

    No category whitelist, invented floor locations, or semantic inference.
    Snapshot relations are incomplete: full runtime Inside/OnTop edges require
    initializing the simulator. Export this provenance, never claim full graph.
    """
    infos = data['objects_info']['init_info']
    binding = data['metadata']['task']['inst_to_name']
    robots = {v for k,v in binding.items() if _synset(k)=='agent.n.01'}
    names = {k for k,v in infos.items() if k not in robots and
             'robots' not in v.get('class_module','')}
    edges,room_edges,spatial_edges = set(),set(),set()
    rooms = set(geometry.map.names.values())
    room_fallback = []
    positions = data['state']['registry']['object_registry']
    for name in sorted(names):
        in_rooms = infos[name]['args'].get('in_rooms',[]) or []
        if not in_rooms and name in positions and 'root_link' in positions[name]:
            room = geometry.map.room_at(*positions[name]['root_link']['pos'][:2])
            if room:
                in_rooms = [room]
                room_fallback.append(name)
        for room in in_rooms:
            rooms.add(room)
            room_edges.add((f'room:{room}',f'object:{name}'))
    for lit in initial:
        if lit[0] in {'inside','ontop'} and len(lit)>2:
            child,parent = binding.get(lit[1]),binding.get(lit[2])
            if child in names and parent in names and child!=parent:
                spatial_edges.add((f'object:{parent}',f'object:{child}'))
    edges = room_edges | spatial_edges | {('house',f'room:{r}') for r in rooms}
    nodes = {f'object:{n}' for n in names}
    result = graph_choice(edges,nodes)
    room_only = graph_choice(edges-spatial_edges,nodes)
    result.update(room_count=len(rooms),room_membership_edges=len(room_edges),
                  spatial_edges=len(spatial_edges),room_map_fallback_objects=room_fallback,
                  room_only_choice=room_only['choice'],
                  relation_coverage='snapshot_plus_bddl_only_not_full_runtime_graph')
    return result


def generation_choice(data, initial, geometry):
    """Reconstruct recorded placement dependencies; count category alternatives.

    Candidate counts are metadata-compatible proxies, not simulator-validated
    alternatives and not a posterior conditioned on the public instruction.
    """
    old = snapshot_graph(data, initial, geometry)
    infos = data['objects_info']['init_info']
    binding = data['metadata']['task']['inst_to_name']
    categories = {f'object:{n}':v['args'].get('category') for n,v in infos.items()}
    room_edges = {(u,v) for u,v in old['edges'] if u.startswith('room:')}
    dependencies = {(u,v) for u,v in old['edges'] if u.startswith('object:')}
    nested = {v for u,v in dependencies}
    # Explicit placement dependencies take precedence over inherited room labels.
    edges = {(u,v) for u,v in old['edges'] if not (u.startswith('room:') and v in nested)}
    adjacency, parents = {},{}
    for u,v in edges:
        adjacency.setdefault(u,set()).add(v)
        parents.setdefault(v,set()).add(u)
    memberships = {}
    for room,obj in room_edges: memberships.setdefault(obj,set()).add(room)
    all_rooms = sorted(adjacency.get('house',set()))
    explicit_rooms = {binding[x[1]]:x[2] for x in initial
                      if x[0]=='inroom' and len(x)>2 and x[1] in binding}
    node_paths = {}
    def paths_to(node,seen=frozenset()):
        if node=='house': return [['house']]
        if node in seen: raise ValueError('Placement dependency cycle')
        if node in node_paths:return node_paths[node]
        result=[]
        for p in sorted(parents.get(node,set())):
            result.extend(path+[node] for path in paths_to(p,seen|{node}))
        node_paths[node]=result
        return result
    names = sorted(set(x['object'] for x in old['paths'])|set(old['unreachable_objects']))
    details=[];missing=[]
    for obj in names:
        alternatives = paths_to(obj)
        if not alternatives:
            missing.append(obj);continue
        scores=[]
        for path in alternatives:
            room = next((n for n in path if n.startswith('room:')),None)
            ks=[];restricted=[];candidate_lists=[]
            for node in path[1:]:
                if node.startswith('room:'):
                    # Global placement-site counts below already include room choice.
                    candidates=[node] if len(path)>3 else all_rooms
                    anchor=next((n for n in path if n.startswith('object:')),None)
                    label=explicit_rooms.get(anchor.removeprefix('object:')) if anchor else None
                    filtered=candidates
                elif node==obj:
                    # Terminal object is the thing to locate, not another placement site.
                    candidates=[node];filtered=candidates
                else:
                    # Candidate sites across the whole scene; no ground-truth room filter.
                    candidates=sorted(n for n,c in categories.items() if c==categories[node])
                    if node not in candidates:candidates.append(node)
                    filtered=[n for n in candidates if room in memberships.get(n,set())]
                ks.append(len(candidates));restricted.append(len(filtered) or 1)
                candidate_lists.append(candidates)
            depth=len(path)-1
            scores.append(dict(object=obj,path=path,depth=depth,candidate_counts=ks,
                candidates=candidate_lists,cost=depth+sum(math.log2(k) for k in ks),
                private_room_conditioned_cost=depth+sum(math.log2(k) for k in restricted)))
        details.append(min(scores,key=lambda s:(s['cost'],s['path'])))
    complete=bool(details) and not missing
    return dict(choice=sum(x['cost'] for x in details)/len(details) if complete else None,
        private_room_conditioned_choice=sum(x['private_room_conditioned_cost'] for x in details)/len(details) if complete else None,
        mean_path_depth=sum(x['depth'] for x in details)/len(details) if details else None,
        max_path_depth=max((x['depth'] for x in details),default=None),
        object_count=len(names),unreachable_objects=missing,paths=details,edges=sorted(map(list,edges)),
        removed_room_shortcuts=sum(v in nested for u,v in room_edges),
        previous_choice=old['choice'],room_only_choice=old['room_only_choice'],
        relation_coverage='Recorded BDDL placement dependencies only; background nesting not recovered',
        candidate_coverage='Same native category across the whole scene. Placement-site counts include room selection; do not count room branching again. No physical feasibility filtering or public-instruction posterior.')


class SceneGeometry:
    """Static 8-neighbour point-agent map; NOT robot feasibility validation."""
    def __init__(self, scene_dir, categories):
        import numpy as np
        from scipy.sparse import coo_matrix
        from scipy.spatial import cKDTree
        from rummagebench.authoring.tasks.room_map import RoomMap
        self.map = RoomMap.load(scene_dir, categories, resolution=0.1)
        self.rc = np.argwhere(self.map.trav)
        self.xy = (self.rc[:, ::-1]-self.map.size/2)*self.map.resolution
        self.nearest = cKDTree(self.xy)
        ids = np.full(self.map.trav.shape, -1, dtype=np.int32)
        ids[tuple(self.rc.T)] = np.arange(len(self.rc))
        self.node_rooms = self.map.room_ins[tuple(self.rc.T)]
        self.room_trees = {}
        for room in self.map.names:
            ix = np.flatnonzero(self.node_rooms == room)
            if len(ix):
                self.room_trees[room] = (ix, cKDTree(self.xy[ix]))
        sources, targets, weights = [], [], []
        n = self.map.size
        for dr, dc in [(1,0),(0,1),(1,1),(1,-1)]:
            dst = self.rc + [dr, dc]
            valid = (dst[:,0]>=0)&(dst[:,0]<n)&(dst[:,1]>=0)&(dst[:,1]<n)
            src_idx = np.flatnonzero(valid)
            src, dst = self.rc[valid], dst[valid]
            good = ids[tuple(dst.T)] >= 0
            if dr and dc:
                good &= (ids[src[:,0]+dr,src[:,1]]>=0)&(ids[src[:,0],src[:,1]+dc]>=0)
            s, t = src_idx[good], ids[tuple(dst[good].T)]
            sources.extend([s,t]); targets.extend([t,s])
            weights.extend([np.full(len(s), math.hypot(dr,dc)*0.1)]*2)
        self.graph = coo_matrix((np.concatenate(weights),
            (np.concatenate(sources),np.concatenate(targets))),
            shape=(len(self.rc),len(self.rc))).tocsr()

    def snap(self, xy, limit=1.5):
        r,c = self.map.to_map(*xy)
        room = int(self.map.room_ins[r,c]) if 0<=r<self.map.size and 0<=c<self.map.size else 0
        if room in self.room_trees:
            ix, tree = self.room_trees[room]
            distance, local = tree.query(xy)
            idx = int(ix[local])
        else:
            distance, idx = self.nearest.query(xy)
        if distance > limit:
            raise ValueError(f'No map approach point within {limit} m (nearest {distance:.3f} m)')
        return int(idx), float(distance)

    def extent(self, points):
        import numpy as np
        from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
        snapped = [self.snap(p) for p in points]
        nodes = sorted({n for n,_ in snapped})
        # Unbounded field arrays are released per source to keep memory small.
        matrix = np.array([dijkstra(self.graph, directed=False, indices=n)[nodes] for n in nodes])
        if not np.isfinite(matrix).all():
            raise ValueError('Task locations disconnected in the static map')
        return dict(distance_m=float(minimum_spanning_tree(matrix).sum()),
                    unique_approach_points=len(nodes), max_snap_m=max(s for _,s in snapped),
                    approach_points=[self.xy[n].tolist() for n in nodes])


def task_sections(task):
    tree = parse_bddl(Path(task['definition']).read_text())
    sec = {n[0]: n[1:] for n in tree if isinstance(n,list) and n}
    initial, goal = [], []
    for e in sec.get(':init', []): _literals(e, initial)
    for e in sec.get(':goal', []): _literals(e, goal)
    return declared_objects(sec[':objects']), initial, goal


def instance_metrics(task, instance, geometry, move_max_m):
    objects, initial, goal = task_sections(task)
    template = Path(instance['template'])
    data = json.loads(template.read_text())
    args = {k:v['args'] for k,v in data['objects_info']['init_info'].items()}
    states = data['state']['registry']['object_registry']
    meta = data['metadata']['task']
    binding = meta['inst_to_name']
    positions = {k:v['root_link']['pos'][:2] for k,v in states.items() if 'root_link' in v}
    result = dict(**instance, template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
                  distance_m=None, choice=None, issues=[])
    structure = generation_choice(data, initial, geometry)
    result['structure'] = structure
    result['choice'] = structure['choice']
    if structure['unreachable_objects']:
        result['issues'].append('Some scene objects have no root path; Choice withheld')
    # Spatial extent includes all objects referenced in terminal goal predicates.
    # Quantified domains and alternatives are unioned, not solved as a plan.
    referenced = set()
    for literal in goal:
        if literal[0].startswith('__') or literal[0] in {'inroom','real','future'}:
            continue
        for token in literal[1:]:
            if token in objects:
                referenced.add(token)
            else:
                referenced.update(n for n,t in objects.items() if t==token)
    # BDDL '*': domain placeholder, not a physical object with a pose.
    wildcard_types = {objects[n] for n in referenced if n.endswith('_*')}
    referenced = {n for n in referenced if not n.endswith('_*')}
    referenced.update(n for n in binding if _synset(n) in wildcard_types)
    referenced = sorted(n for n in referenced if objects.get(n,_synset(n)) not in {'agent.n.01','floor.n.01'})
    missing = [n for n in referenced if binding.get(n) not in positions]
    result['goal_objects'] = referenced
    result['missing_goal_positions'] = missing
    # Compare task-conditioned averaging against the same graph's full-scene mean.
    result['choice_scene_mean'] = structure['choice']
    selected_nodes = {f'object:{binding[n]}' for n in referenced if n in binding}
    selected = [x for x in structure['paths'] if x['object'] in selected_nodes]
    result['choice_target_count'] = len(selected)
    result['choice_target_paths'] = selected
    result['choice_missing_targets'] = sorted(selected_nodes-{x['object'] for x in selected})
    result['choice'] = sum(x['cost'] for x in selected)/len(selected) if selected and not missing and not result['choice_missing_targets'] else None
    # Missing background roots no longer invalidate a task-conditioned statistic.
    result['issues'] = [e for e in result['issues'] if e!='Some scene objects have no root path; Choice withheld']
    result['goal_union_relaxation'] = any(x[0]=='__logic__' or x[0]=='__quant__' for x in goal)
    try:
        if missing or not referenced:
            raise ValueError('Goal positions unavailable or no located goal objects')
        all_poses = meta.get('robot_poses',{})
        pose_key = 'R1Pro' if all_poses.get('R1Pro') else 'robot'
        poses = all_poses.get(pose_key,[])
        result['robot_pose_key'] = pose_key
        if not poses:
            raise ValueError('Official robot start unavailable')
        points = [poses[0]['position'][:2]] + [positions[binding[n]] for n in referenced]
        result.update(geometry.extent(points))
    except ValueError as e:
        result['issues'].append(str(e))
    result['distance_normalized'] = result['distance_m']/move_max_m if result['distance_m'] is not None else None
    result['status'] = 'snapshot_graph_proxy' if result['choice'] is not None and result['distance_m'] is not None else 'incomplete_instance'
    return result


def assign_bands(rows):
    eligible = [r for r in rows if r['distance_normalized'] is not None and r['choice'] is not None]
    if not eligible:
        return None
    import numpy as np
    scales = {}
    for field in ['distance_normalized','choice']:
        values = [r[field] for r in eligible]
        lo,hi = map(float,np.quantile(values,[.1,.9]))
        scales[field] = dict(low=lo,high=hi)
        for row in eligible:
            row[field+'_scaled'] = max(0.0,(row[field]-lo)/(hi-lo)) if hi>lo else .5
    for row in eligible:
        row['difficulty_score'] = (row['distance_normalized_scaled']+row['choice_scaled'])/2
    scores = sorted(r['difficulty_score'] for r in eligible)
    low,high = scores[(len(scores)-1)//3],scores[2*(len(scores)-1)//3]
    for row in eligible:
        s = row['difficulty_score']
        row['difficulty_band'] = 'easy' if s<=low else 'medium' if s<=high else 'hard'
        row['borderline'] = min(abs(s-low),abs(s-high)) <= .03
    return dict(easy_max=low,medium_max=high,reference_count=len(eligible),scales=scales,borderline_margin=.03)


def classify_all(cat, annotations, output, assets_root, normalization_scenario):
    from rummagebench.core.scenario import load_scenario
    scenario = load_scenario(normalization_scenario)
    move_max_m = scenario.agent_protocol.max_move_cm/100.0
    geometry_cache, rows = {}, []
    for task in cat['tasks']:
        objects,initial,goal = task_sections(task)
        row = dict(activity=task['activity_name'],definition_id=task['activity_definition_id'],
            definition=task['definition'],definition_sha256=hashlib.sha256(Path(task['definition']).read_bytes()).hexdigest(),
            rigid_only=task['rigid_only'],exclusions=task.get('exclusions',[]),
            instance_count=len(task.get('instances',[])),skill_coverage=task.get('skill_coverage','unverified'),
            distance_m=None,distance_normalized=None,choice=None,difficulty_score=None,difficulty_band='unclassified',
            status='missing_official_instance',instances=[])
        for instance in task.get('instances',[]):
            scene = instance['scene_model']
            if scene not in geometry_cache:
                print(f'Loading static map: {scene}',flush=True)
                geometry_cache[scene] = SceneGeometry(assets_root/'scenes'/scene,assets_root/'metadata/room_categories.txt')
            result = instance_metrics(task,instance,geometry_cache[scene],move_max_m)
            row['instances'].append(result)
        if row['instances']:
            # No cherry-picking a successful instance from an incomplete set.
            if all(i['choice'] is not None for i in row['instances']):
                row['choice'] = sum(i['choice'] for i in row['instances'])/len(row['instances'])
            if all(i['distance_m'] is not None for i in row['instances']):
                row['distance_m'] = sum(i['distance_m'] for i in row['instances'])/len(row['instances'])
            row['distance_normalized'] = row['distance_m']/move_max_m if row['distance_m'] is not None else None
            row['status'] = 'snapshot_graph_proxy' if row['distance_m'] is not None and row['choice'] is not None else 'incomplete_instance'
        rows.append(row)
    thresholds = assign_bands(rows)
    summary = dict(total_definitions=len(rows),with_official_instances=sum(bool(r['instances']) for r in rows),
        distance_available=sum(r['distance_m'] is not None for r in rows),
        choice_available=sum(r['choice'] is not None for r in rows),
        bands=dict(Counter(r['difficulty_band'] for r in rows)),
        rigid_bands=dict(Counter(r['difficulty_band'] for r in rows if r['rigid_only'])),
        statuses=dict(Counter(r['status'] for r in rows)))
    protocol = dict(version='normalized-distance-global-candidate-task-choice-v5-pilot',
        supersedes='The former L/A/U and filtered leaf-tree priors and their bands are invalidated.',
        distance='Geodesic minimum-spanning-tree length joining official robot start and the union of goal-referenced object approach points. Spatial extent proxy, NOT optimal execution distance. Use R1Pro start when provided, otherwise legacy robot start; first official pose.',
        choice='Mean minimum placement-dependency path cost over distinct goal-referenced objects, including destination objects; quantifier/alternative union. Intermediate site K=all same-category scene objects. Room edge K=1 when global site choice already includes room choice, otherwise all scene rooms. Terminal K=1. Full-scene mean retained as comparison.',
        structure='Recorded BDDL Inside/OnTop dependencies supersede child room membership edges. Other scene objects retain native room memberships. All objects remain graph nodes; only goal-referenced objects are averaging targets. No articulation nodes or semantic whitelist.',
        candidates='Metadata category/room compatibility only. Not the actual sampler candidate distribution, not feasibility-certified, not a public-instruction-conditioned posterior. Actual-room-restricted candidate sensitivity is exported separately, never used for primary bands. Goal scope and actual parent category remain privileged evaluator information.',
        move_max_m=move_max_m, normalization_scenario=str(normalization_scenario),
        normalization_scenario_sha256=hashlib.sha256(normalization_scenario.read_bytes()).hexdigest(),
        normalized_distance='distance_m / (agent_protocol.max_move_cm / 100)',
        map='0.1 m static floor_trav_0, point agent, 8-neighbour metric, no diagonal corner cutting; nearest free approach in same segmented room when available; maximum snap 1.5 m. No footprint erosion.',
        score='Equal mean of continuous D and C scaled linearly between cohort p10 and p90, floored at 0 without upper clipping (extremes retain their magnitude); constant axis=.5. Exploratory score tertiles, ties preserved. Within .03 of either boundary flagged borderline (heuristic, not confidence interval). Frozen scales exported; future comparisons must reuse them.',
        thresholds=thresholds,
        caveats=[
            'All 1016 definitions inventoried, but missing instances or incomplete metrics stay unclassified.',
            'Available 100 official instances are a nonrandom subset; these thresholds are not full-1K population thresholds.',
            'PROVISIONAL: snapshot room memberships and BDDL relations are NOT the full simulator relationship graph. Background Inside/OnTop and internal compartments are not recovered.',
            'Task-conditioned scope is a proposed comparison, not the previously requested scene-wide metric. Goal alternatives are unioned and may include optional objects. Background objects still supply candidates, but do not dilute the mean.',
            'Static maps can differ from template geometry/door states. Distances do not certify robot reachability or manipulation feasibility.',
            'MST captures spatial extent only: no pickup-before-place constraints, return trips, repeated carrying, or manipulation effort. Quantifiers/alternatives union all referenced objects, including optional ones.',
            'Non-rigid tasks retained for analysis only, not eligible for current rigid-only runtime.',
            'No simulator or VLM run, and no data enters model observations.'],
        input_catalog_sha256=hashlib.sha256(json.dumps(cat,sort_keys=True).encode()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    protocol['maps'] = {s:{p:hashlib.sha256((assets_root/'scenes'/s/'layout'/p).read_bytes()).hexdigest()
                           for p in ['floor_trav_0.png','floor_insseg_0.png','floor_semseg_0.png']}
                        for s in geometry_cache}
    structures = [i['structure'] for r in rows for i in r['instances'] if i['structure']['choice'] is not None]
    examples = ['turning_on_radio','stacking_wood','turning_out_all_lights_before_sleep',
                'cook_hot_dogs','freeze_fruit','boxing_food_after_dinner','sorting_vegetables']
    sanity = dict(
        evaluated_structures=len(structures),
        mean_path_depth_range=[min(s['mean_path_depth'] for s in structures),max(s['mean_path_depth'] for s in structures)] if structures else None,
        identical_to_room_only=sum(s['room_only_choice'] is not None and abs(s['choice']-s['room_only_choice'])<1e-9 for s in structures),
        max_dependency_depth=max((s['max_path_depth'] for s in structures),default=None),
        removed_room_shortcuts=sum(s['removed_room_shortcuts'] for s in structures),
        findings=[
            'Placement children no longer use direct room shortcuts. Candidate counts refer to alternative placement objects, not actual children.',
            'Global intermediate candidate pools remove actual-room filtering and do not count room selection twice.',
            'Category matching does not prove physical compatibility. Actual filtered sampler candidates and public-information beliefs are not recovered.',
            'Goal-conditioned averaging removes direct background dilution; all-scene comparison retained. Parent category is still conditioned on the realized BDDL placement, not public information.',
            'Labels remain exploratory spatial proxies, not full task difficulty or model success predictions.'],
        examples=[{k:r[k] for k in ['activity','rigid_only','distance_normalized','choice','difficulty_band']} for r in rows if r['activity'] in examples])
    summary['sanity_check'] = sanity
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(protocol=protocol,summary=summary,tasks=rows),indent=2)+'\n')
    columns=['activity','definition_id','rigid_only','instance_count','distance_m','distance_normalized','choice','difficulty_score','difficulty_band','borderline','status']
    with output.with_suffix('.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        writer.writerows(sorted(rows,key=lambda r:(r['difficulty_score'] is None,r['difficulty_score'] or 0,r['activity'])))
    output.with_suffix('.summary.txt').write_text('DISTANCE + STRUCTURAL CHOICE: EXPLORATORY, INCOMPLETE COVERAGE\n\n'+json.dumps(summary,indent=2)+'\n\n'+json.dumps(protocol,indent=2)+'\n')
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path, default=Path("build/behavior/official_catalog.json"))
    p.add_argument("--annotations", type=Path, default=Path(
        "assets/behavior_official/bddl/generated_data/propagated_annots_canonical.json"))
    p.add_argument("--output", type=Path)
    p.add_argument("--difficulty-all", action="store_true", help="Inventory all definitions and classify available scenes using Distance and structural Choice")
    p.add_argument("--assets-root",type=Path,default=Path("/home/ubuntu/ygwang/behavior-datasets/behavior-1k-assets"))
    p.add_argument("--normalization-scenario",type=Path,default=Path("scenarios/behavior_groceries_pilot/scenario.yaml"))
    a = p.parse_args()
    if a.output is None:
        a.output = Path("build/behavior/task_difficulty_v1.json" if a.difficulty_all else "build/behavior/official_task_axes.json")
    cat = json.loads(a.catalog.read_text())
    ann = json.loads(a.annotations.read_text())
    if a.difficulty_all:
        print(json.dumps(classify_all(cat, ann, a.output, a.assets_root, a.normalization_scenario),indent=2))
        return 0
    rows = [analyze(t, ann) for t in cat["tasks"] if t["rigid_only"]]
    summary = {
        "rigid_definitions": len(rows),
        "with_local_instances": sum(bool(r["local_instances"]) for r in rows),
        "family": Counter(r["closest_family"] for r in rows),
        "hidden_at_init(S2-like)": sum(bool(r["hidden_at_init"]) for r in rows),
        "multi_room(S3-like)": sum(r["multi_room"] for r in rows),
        "goal_needs_inside_furniture": sum(r["goal_needs_inside_furniture"] for r in rows),
        "single_fetch_place_ontop_with_instances": sum(
            r["closest_family"] == "single_fetch_place" and r["goal_kinds"] == ["ontop"]
            and bool(r["local_instances"]) for r in rows),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"summary": summary, "tasks": rows}, indent=1, default=list) + "\n")
    print(json.dumps(summary, indent=1, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
