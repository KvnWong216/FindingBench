"""Enumerate every original BDDL definition, with a conservative rigid audit.

Articulated furniture consists of rigid links and is allowed. All declared
objects (not only goal objects) must have an explicit rigidBody annotation.
Unknown predicates/types fail closed. Initial nonmechanical states are also
excluded: no washing, heating, cutting, substances or deformable manipulation.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

POLICY = 'rigid-manipulation-v1'
PREDICATES = {'inside', 'ontop', 'under', 'nextto', 'touching', 'onfloor',
              'overlaid', 'open', 'inroom', 'real', 'future'}
LOGIC = {'and', 'or', 'not', 'imply', 'exists', 'forall', 'forn', 'forpairs', 'fornpairs'}


def parse_bddl(text):
    tokens = re.findall(r'\(|\)|[^\s()]+', re.sub(r';[^\n]*', '', text))
    stack, roots = [], []
    for token in tokens:
        if token == '(':
            node = []
            (stack[-1] if stack else roots).append(node)
            stack.append(node)
        elif token == ')':
            if not stack:
                raise ValueError('unbalanced BDDL')
            stack.pop()
        else:
            if not stack:
                raise ValueError('token outside BDDL expression')
            stack[-1].append(token)
    if stack or len(roots) != 1:
        raise ValueError('incomplete BDDL')
    return roots[0]


def audit_definition(path, annotations):
    path = Path(path)
    raw = path.read_bytes()
    tree = parse_bddl(raw.decode())
    sections = {n[0]: n[1:] for n in tree if isinstance(n, list) and n}
    reasons, synsets, predicates = [], [], set()
    objects = sections.get(':objects', [])
    for i, token in enumerate(objects):
        if token == '-':
            synsets.append(objects[i + 1])
    for synset in sorted(set(synsets)):
        props = annotations.get(synset, {})
        if 'rigidBody' not in props or set(props) & {'substance', 'cloth', 'softBody', 'rope', 'deformable'}:
            reasons.append('non_rigid_or_unknown:' + synset)
    def visit(node):
        if not isinstance(node, list) or not node:
            return
        head = node[0]
        if isinstance(head, str) and not head.startswith('?') and head not in LOGIC and not head.isdigit():
            predicates.add(head)
        for child in node[1:]:
            visit(child)
    for section in (':init', ':goal'):
        if section not in sections:
            reasons.append('missing:' + section)
        for expr in sections.get(section, []):
            visit(expr)
    reasons.extend('unsupported_predicate:' + p for p in sorted(predicates - PREDICATES))
    # Future/creation and overlaid require executors we do not have; keep the
    # rigid inventory separate from skill coverage rather than dropping tasks.
    return dict(activity_name=path.parent.name,
                activity_definition_id=int(path.stem.removeprefix('problem')),
                definition=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                synsets=sorted(set(synsets)), predicates=sorted(predicates),
                rigid_only=not reasons, exclusions=reasons,
                skill_coverage='unverified')


def catalog(bddl_root, assets_root=None):
    root = Path(bddl_root)
    ann_path = root / 'generated_data/propagated_annots_canonical.json'
    annotations = json.loads(ann_path.read_text())
    paths = sorted((root / 'activity_definitions').glob('*/problem*.bddl'))
    if not paths:
        raise ValueError('no official BDDL definitions found')
    tasks = [audit_definition(p, annotations) for p in paths]
    templates = []
    if assets_root is not None:
        assets = Path(assets_root)
        roots = [assets / 'scenes', assets / 'behavior-1k-assets/scenes',
                 assets / '2026-challenge-task-instances/scenes',
                 assets.parent / '2026-challenge-task-instances/scenes']
        templates = sorted({p.resolve() for folder in roots for p in folder.glob('*/json/*_template.json')})
    for task in tasks:
        task['instances'] = []
        if assets_root is not None:
            suffix = f"_task_{task['activity_name']}_{task['activity_definition_id']}_"
            for path in templates:
                scene = path.parents[1].name
                prefix = scene + suffix
                if path.name.startswith(prefix):
                    instance = path.name[len(prefix):-len('_template.json')]
                    if instance.isdigit():
                        task['instances'].append(dict(scene_model=scene, activity_instance_id=int(instance),
                                                      template=str(path.resolve())))
    return dict(policy=POLICY, total_definitions=len(tasks),
                rigid_definitions=sum(t['rigid_only'] for t in tasks),
                assets_scanned=assets_root is not None,
                annotations_sha256=hashlib.sha256(ann_path.read_bytes()).hexdigest(), tasks=tasks)


def assert_rigid_task(spec):
    root = Path(spec.bddl_root)
    annotations = json.loads((root / 'generated_data/propagated_annots_canonical.json').read_text())
    result = audit_definition(root / 'activity_definitions' / spec.activity_name /
                              f'problem{spec.activity_definition_id}.bddl', annotations)
    if not result['rigid_only']:
        raise ValueError('not a rigid manipulation task: ' + ', '.join(result['exclusions']))
    return result
