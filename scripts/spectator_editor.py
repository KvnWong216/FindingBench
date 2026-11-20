"""Native authoring controls for spectator.py --edit.

HTTP threads only enqueue commands. All scene reads/mutations happen on the
render thread. A save publishes a new frozen revision only after cache export
and a separate-process native reload; the selected dataset row is the commit.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlsplit
import uuid

from spectator import ROOT, SharedState, Handler

ALIASES = {
    '酱油': ('bottle_of_soy_sauce', 'soy_sauce_bottle'),
    '纸巾盒': ('box_of_tissues', 'tissue_dispenser'), '纸巾': ('box_of_tissues', 'tissue_dispenser'),
    '卷纸': ('toilet_paper',), '厨房纸': ('paper_towel',),
    '洗发水': ('bottle_of_shampoo', 'shampoo_bottle'), '洗发露': ('bottle_of_shampoo', 'shampoo_bottle'),
    '沐浴露': ('bottle_of_body_wash', 'bottle_of_shower_gel'),
    '肥皂': ('bar_of_soap', 'soap', 'soap_dispenser'), '洗手液': ('soap_dispenser',),
    '洗面奶': ('bottle_of_cleanser', 'bottle_of_face_wash'), '梳子': ('comb', 'hairbrush'),
    '牙刷': ('toothbrush',), '牙膏': ('tube_of_toothpaste', 'toothpaste'),
    '杯子': ('mug', 'water_glass', 'cup'), '马克杯': ('mug',), '玻璃杯': ('water_glass',),
    '碗': ('bowl',), '盘子': ('plate', 'saucer'), '勺子': ('spoon', 'soup_spoon', 'teaspoon'),
    '叉子': ('fork', 'table_fork'), '筷子': ('chopstick', 'chopsticks'),
    '锅': ('saucepan', 'saucepot', 'stockpot', 'frying_pan'),
    '盐': ('salt_shaker', 'jar_of_salt', 'salt'), '糖': ('bag_of_sugar', 'sugar'),
    '醋': ('bottle_of_vinegar', 'vinegar_bottle'), '胡椒': ('pepper_shaker', 'pepper_grinder'),
    '蜂蜜': ('jar_of_honey', 'bottle_of_honey', 'honey_bottle'),
    '可乐': ('can_of_cola', 'bottle_of_cola', 'soda_can', 'can_of_soda', 'bottle_of_soda'),
    '牛奶': ('carton_of_milk', 'bottle_of_milk'), '酸奶': ('yogurt', 'cup_of_yogurt', 'yogurt_cup'),
    '果汁': ('carton_of_juice', 'bottle_of_juice'), '矿泉水': ('bottle_of_water',),
    '苹果': ('apple',), '香蕉': ('banana',), '橘子': ('orange',), '梨': ('pear',),
    '红辣椒': ('chili_pepper', 'red_chili_pepper'), '青椒': ('bell_pepper', 'green_pepper'),
    '猪肉': ('pork', 'pork_chop'), '牛肉': ('beef', 'steak'),
    '书': ('hardback', 'paperback', 'paperback_book'), '书本': ('hardback', 'paperback', 'paperback_book'),
    '笔记本': ('notebook', 'notepad'), '铅笔': ('pencil',), '笔筒': ('pencil_holder',),
    '闹钟': ('alarm_clock',), '时钟': ('clock', 'alarm_clock'),
    '花盆': ('pot_plant',), '盆栽': ('pot_plant',), '花瓶': ('vase',),
    '药': ('pill_bottle', 'bottle_of_medicine', 'medicine_bottle'),
    '手柄': ('game_controller',), '游戏手柄': ('game_controller',), '手机': ('cell_phone', 'smartphone'),
    '遥控器': ('remote_control', 'remote'), '篮子': ('shopping_basket', 'wicker_basket', 'basket'),
    '毛巾': ('towel', 'hand_towel'), '抹布': ('rag', 'dishtowel'),
    '桌子': ('coffee_table', 'breakfast_table', 'desk', 'dining_table'),
    '椅子': ('chair', 'straight_chair', 'swivel_chair'), '台灯': ('table_lamp', 'desk_lamp'),
    '柜子': ('bottom_cabinet', 'top_cabinet', 'cabinet'), '抽屉': ('nightstand', 'dresser'),
    '收纳盒': ('storage_box', 'box', 'tupperware'), '购物袋': ('shopping_bag', 'paper_bag'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, doc):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, path)


class EditError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def explain_error(error, op='edit'):
    """User-facing explanation; the original exception stays in the debug log."""
    raw = str(error)
    code = getattr(error, 'code', None)
    if code:
        return dict(code=code, message=raw)
    rules = [
        (('原生 OPEN/CLOSE 能力',), 'not_openable',
         '选中的物品不能开合。请先点击柜门或抽屉，再使用“打开”或“关闭”。'),
        (('未安装这个 BEHAVIOR',), 'asset_not_installed',
         '这个型号不在本机资产库中。请从搜索结果重新选择一个已安装的型号。'),
        (('footprint overlaps', 'invalid spawn'), 'robot_blocked',
         '物品挡住了机器人起始位置。请把刚放的物品移远一点，给机器人脚下和周围留出空间。'),
        (('physics state is invalid', 'nonfinite', 'infinite'), 'physics_invalid',
         '仿真算出了无效的位置或关节状态，这次编辑不能采用。请换一个落点或型号；反复出现时保留错误编号供排查。'),
        (('out of memory', 'cuda error', 'device lost'), 'gpu_resource',
         '显卡内存不足或渲染设备失去响应。请先关闭不用的仿真窗口，再重新打开编辑器。'),
        (('no space left', 'disk quota'), 'disk_full',
         '磁盘空间不足，无法写入配置或缓存。请清理空间后重新保存。'),
        (('permission denied',), 'permission', '无法写入保存目录。请检查目录的写入权限后重试。'),
        (('no such file', 'file not found'), 'missing_file',
         '场景或资产依赖的文件找不到了。请恢复文件或换一个已安装的资产型号，再重新打开场景。'),
        (('其他编辑器',), 'concurrent_edit',
         '这条数据已在另一个窗口保存过。当前修改没有覆盖它；请保留当前窗口，核对两个版本后再决定采用哪个。'),
        (('独立重载', '重载姿态', '重载后的', '柜门初始状态'), 'reload_mismatch',
         '保存后的场景重新打开时与当前画面不一致，因此没有覆盖原数据。当前编辑仍保留，请查看错误记录。'),
        (('缓存',), 'cache_failed',
         '渲染缓存没有通过检查，因此本次保存没有覆盖原数据。当前编辑仍保留，请重试保存或查看错误记录。'),
        (('固定', 'sha256', 'hash mismatch', 'fingerprint', 'dependency'), 'source_changed',
         '这个场景依赖的文件与原先记录的版本不同，不能可靠地恢复或保存。请保留错误记录，修复文件版本后再试。'),
        (('validation error', 'field required', 'extra inputs', 'scenario'), 'configuration_invalid',
         '编辑结果无法写成有效的任务配置。这通常是编辑器兼容性问题，请保留错误编号供修复。'),
        (('布局关系不成立', '原生物品未放稳'), 'saved_relation_lost',
         '有物品已经离开了原先的桌面或容器。请重新摆放该物品，确认没有掉落，再保存。'),
        (('物体不存在',), 'no_selection', '没有选中可操作的物体。请先点击画面里的物品或柜门。'),
    ]
    lower = raw.lower()
    for matches, category, message in rules:
        if any(term.lower() in lower for term in matches):
            return dict(code=category, message=message)
    if re.search('[\u4e00-\u9fff]', raw) and not re.search('Traceback|Error:|Exception:', raw):
        return dict(code='edit_rejected', message=raw[:500])
    action = {'add':'添加物品', 'move':'移动物品', 'open':'打开柜门', 'close':'关闭柜门',
              'delete':'删除物品', 'save':'保存场景', 'search':'搜索资产'}.get(op, '编辑')
    return dict(code='unexpected_error', message=action+'遇到了程序错误，无法完成。请保留错误编号供排查；原始错误可在“技术详情”里查看。')


def stable_orientation(orientations, yaw=None, current=None):
    """Deterministic native rest pose + world-Z heading, never the clicked normal.

    This follows DatasetObject.sample_orientation's empty-metadata identity
    convention without randomizing yaw. Physics still validates equilibrium.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    entries = list(orientations.values()) if isinstance(orientations, dict) else list(orientations or [])
    entries = sorted(entries, key=lambda item: -float(item.get('prob', 0)))
    base = Rotation.identity()
    if entries:
        q = np.asarray(entries[0]['rotation'], float)
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
            raise EditError('asset_pose_invalid', '这个资产的稳定姿态记录不完整。请换一个型号，并保留错误记录供修复。')
        base = Rotation.from_quat(q)
    if yaw is None:
        delta = Rotation.from_quat(current if current is not None else [0, 0, 0, 1])*base.inv()
        matrix = delta.as_matrix()
        yaw = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    if not math.isfinite(yaw):
        raise EditError('invalid_heading', '水平旋转角度无效。请从左侧选择一个角度后再放置。')
    return (Rotation.from_euler('z', yaw, degrees=True)*base).as_quat().tolist()


def gravity_anchor(point, normal, half_extent):
    """Clear a near-vertical clicked face by the object's horizontal radius.

    The subsequent cast is still strictly vertical. This prevents the object
    center from being put directly inside a wall/back panel on a side click.
    """
    import numpy as np
    point, normal = np.asarray(point, float), np.asarray(normal, float)
    if abs(normal[2]) < .5 and np.linalg.norm(normal[:2]) > .5:
        direction = normal[:2]/np.linalg.norm(normal[:2])
        radius = float(np.dot(np.abs(direction), np.asarray(half_extent)[:2]))+.008
        point = point.copy(); point[:2] += direction*radius
    return point.tolist()


def downward_surface(point, raycast, owner, ignored):
    """First upward-facing mesh surface below a click, irrespective of click normal."""
    point = list(map(float, point))
    hits = raycast([point[0], point[1], point[2]+.05],
                   [point[0], point[1], point[2]-10.], only_closest=False, ignore_bodies=ignored)
    hits = sorted((h for h in hits if h.get('hit')), key=lambda h: float(h['distance']))
    for hit in hits:
        if float(hit.get('normal', [0,0,0])[2]) < .8:
            continue
        support = owner(hit.get('rigidBody') or hit.get('collision') or '')
        if support is not None:
            return support, hit
    raise EditError('no_support_below', '点击位置的正下方没有能托住物品的平面。请换到桌面、层板或地板上方；放进柜子时可先打开柜门。')


class AssetIndex:
    """Installed models only; Chinese aliases never invent a missing asset."""
    def __init__(self, objects_root):
        self.root = Path(objects_root).resolve()
        self.categories = {p.name: sorted(m.name for m in p.iterdir()
                           if (m/'misc/metadata.json').is_file() and (m/'usd').is_dir())
                           for p in self.root.iterdir() if p.is_dir()}

    def item(self, category, model):
        if model not in self.categories.get(category, ()):
            raise ValueError('未安装这个 BEHAVIOR 类别／型号')
        meta = json.loads((self.root/category/model/'misc/metadata.json').read_text())
        label = next((name for name, cats in ALIASES.items() if category in cats), category.replace('_', ' '))
        return dict(category=category, model=model, label=label, size=meta.get('bbox_size'))

    def search(self, query, offset=0, limit=36):
        query = str(query).strip().lower()[:100]
        terms = query.replace(' ', '_').split()
        aliases = set(ALIASES.get(query, ()))
        for name, cats in ALIASES.items():
            if query and query in name:
                aliases.update(cats)
        matches = []
        for category, models in self.categories.items():
            score = 0 if category in aliases or category == query else 1
            for model in models:
                if not query or category in aliases or all(t in category+' '+model for t in terms):
                    matches.append((score, category, model))
        matches.sort()
        return dict(total=len(matches), offset=offset, items=[self.item(c, m) for _, c, m in matches[offset:offset+limit]])


def screen_ray(camera, xy, fov, aspect):
    if (not isinstance(xy, list) or len(xy) != 2 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in xy)):
        raise ValueError('点击坐标必须位于画面内部')
    forward, right, up = camera.basis()
    tangent = math.tan(math.radians(fov)/2)
    direction = [forward[i]+right[i]*(2*xy[0]-1)*tangent+up[i]*(1-2*xy[1])*tangent/aspect for i in range(3)]
    norm = math.sqrt(sum(x*x for x in direction))
    return list(camera.position), [v/norm for v in direction]


def calibrated_point(position, basis, intrinsic, u, v, depth):
    """Renderer optical Z-depth with its actual, possibly non-square K."""
    import numpy as np
    f, r, up = (np.asarray(x, float) for x in basis)
    return (np.asarray(position, float)+depth*(f+r*(u-intrinsic[0][2])/intrinsic[0][0]
                                             -up*(v-intrinsic[1][2])/intrinsic[1][1]))


def placement_record(entity, relation, support, xy, link=None):
    # The original ScenarioSpec forbids position_xy on Inside recipes. The
    # exact inner-shelf pose is persisted in the native frozen snapshot.
    placement = dict(entity=entity, relation=relation, receptacle=support)
    if relation == 'on_top':
        placement['position_xy'] = list(xy)
    if link is not None and relation == 'inside':
        placement['link'] = link
    return placement


def protected_entities(document):
    names = {document['robot']['name']}
    if document.get('target'):
        names.add(document['target']['entity'])
    for goal in document.get('goal', []):
        names.add(goal['entity'])
        if goal.get('receptacle'):
            names.add(goal['receptacle'])
    for constraint in document.get('temporal', []):
        names.update(constraint.get(k) for k in ('entity', 'before_entity') if constraint.get(k))
    return names


def delete_from_document(document, name):
    """Keep task semantics valid; a referenced support must be cleared first."""
    doc = deepcopy(document)
    if name in protected_entities(doc):
        raise ValueError('这是任务目标、终点或机器人，不能删除；可拖动任务物品调整布局')
    if any(p['receptacle'] == name and p['entity'] != name for p in doc['placements']):
        raise ValueError('这个物体仍被用作支撑或容器，请先移走／删除其中的物品')
    doc['objects'] = [o for o in doc['objects'] if o['name'] != name]
    doc['placements'] = [p for p in doc['placements'] if p['entity'] != name]
    doc['initial_states'].pop(name, None)
    # Old demonstration/oracle sequences no longer describe the authored layout.
    doc['agent'] = {} if 'agent' in doc else doc.get('agent', {})
    if 'agent' not in document:
        doc.pop('agent', None)
    return doc


def compose_named_state(before, after, changed):
    """SDK native named states: preserve unrelated bodies across edit steps."""
    result = deepcopy(after)
    old = before[0]['registry']['object_registry']
    current = result[0]['registry']['object_registry']
    result[0]['registry']['object_registry'] = {
        name: deepcopy(value if name in changed or name not in old else old[name]) for name, value in current.items()}
    # Preserve scene frame and systems as well as original object states.
    for key in ('pos', 'ori'):
        result[0][key] = deepcopy(before[0][key])
    for key, value in before[0]['registry'].items():
        if key != 'object_registry':
            result[0]['registry'][key] = deepcopy(value)
    return result


def freeze_runtime(base, output):
    """Small derived runtime supports deletion of native scene props on reload."""
    base, output = Path(base), Path(output)
    env_file = 'src/rummagebench/authoring/tasks/household_state.py'
    marker = 'findingbench_native_editor_removed_objects_v2'
    # If the selected member already owns this hook, reuse its immutable tree.
    if marker in (base/env_file).read_text():
        return base
    key = hashlib.sha256((str(base)+digest(base/env_file)+marker).encode()).hexdigest()[:20]
    target = output/key
    if (target/'editor_runtime.json').exists():
        manifest = json.loads((target/'editor_runtime.json').read_text())
        if any(digest(target/p) != sha for p, sha in manifest['sha256'].items()):
            raise ValueError('已固定的编辑器运行版本发生变化')
        return target
    temporary = output/(key+'.'+uuid.uuid4().hex+'.tmp'); temporary.mkdir(parents=True)
    for directory in ('src', 'configs'):
        shutil.copytree(base/directory, temporary/directory, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for directory in ('assets',):
        if (base/directory).exists():
            (temporary/directory).symlink_to((base/directory).resolve(), target_is_directory=True)
    with (temporary/env_file).open('a') as stream:
        stream.write('''

# findingbench_native_editor_removed_objects_v2
_editor_original_restore = restore_frozen_member
def restore_frozen_member(backend, path, scenario):
    import json
    removed = json.loads(Path(path).read_text()).get('builder', {}).get('editor_removed_scene_objects', [])
    objects = [backend._env.scene.object_registry('name', name) for name in removed]
    objects = [obj for obj in objects if obj is not None]
    if objects:
        backend._sim.stop()
        for obj in objects:
            backend._env.scene.remove_object(obj)
        backend._sim.play()
        backend._sim.step()
    return _editor_original_restore(backend, path, scenario)
''')
    hashes = {str(p.relative_to(temporary)): digest(p) for p in temporary.rglob('*') if p.is_file() and not p.is_symlink() and 'assets' not in p.relative_to(temporary).parts}
    atomic(temporary/'editor_runtime.json', dict(version=1, parent=str(base), sha256=hashes))
    output.mkdir(parents=True, exist_ok=True)
    temporary.rename(target)
    return target


def publish_revision(directory, original, replacement, revision):
    """Publish one row last, with conflict detection and recoverable metadata."""
    from spectator_cache import identity, lookup
    directory, revision = Path(directory), Path(revision)
    with (directory/'.editor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory/'human_ready.json'; before = path.read_bytes(); document = json.loads(before)
        indices = [i for i, t in enumerate(document['tasks']) if t['parent_task_id'] == original['parent_task_id'] and t['id'] == original['id']]
        if len(indices) != 1:
            raise ValueError('数据条目不再唯一，保存已取消')
        index = indices[0]; previous = document['tasks'][index]
        if identity(previous) != identity(original):
            raise ValueError('这个条目已被其他编辑器更新，请重新打开；没有覆盖别人的修改')
        if lookup(replacement['editor_cache_dir'], replacement) is None:
            raise ValueError('新的渲染缓存校验失败，未覆盖原数据')
        backup = revision/'backup'; backup.mkdir(exist_ok=True)
        snapshots = {p.name: p.read_bytes() for p in (path, directory/'manifest.json', directory/'verification.json') if p.exists()}
        for name, content in snapshots.items():
            (backup/name).write_bytes(content)
        document['tasks'][index] = {k: v for k, v in replacement.items() if k != 'index'}
        try:
            manifest_path = directory/'manifest.json'
            if manifest_path.exists():
                manifest = json.loads(snapshots['manifest.json'])
                for row in manifest.get('tasks', []):
                    if row.get('parent_task_id') == original['parent_task_id']:
                        row.update(play_scenario=replacement['play_scenario'], scenario=replacement['scenario'],
                            scenario_sha256=digest(replacement['play_scenario']), runtime_root=replacement['runtime_root'],
                            frozen_manifest=str(revision/'frozen_member.private.json'),
                            frozen_manifest_sha256=json.loads((revision/'frozen_member.private.json').read_text())['manifest_sha256'],
                            native_initialization_report=str(revision/'validation.json'), native_initialization_report_sha256=digest(revision/'validation.json'),
                            native_reload_report=str(revision/'restore_validation.json'), native_reload_report_sha256=digest(revision/'restore_validation.json'),
                            editor_revision=str(revision))
                manifest.setdefault('human_editor_revisions', {})[original['parent_task_id']] = str(revision)
                runtime_manifest = Path(replacement['runtime_root'])/'editor_runtime.json'
                if runtime_manifest.exists():
                    manifest.setdefault('runtime_sources', {})[replacement['runtime_root']] = json.loads(runtime_manifest.read_text())['sha256']
                atomic(manifest_path, manifest)
            verification_path = directory/'verification.json'
            if verification_path.exists():
                verification = json.loads(snapshots['verification.json'])
                verification['status'] = 'historical_release_plus_individually_validated_human_edits'
                verification.setdefault('human_editor_revisions', {})[original['parent_task_id']] = dict(
                    revision=str(revision), independent_reload=True, cache=True, full_skill_completion='unverified')
                verification['historical_release_verification'] = str(backup/'verification.json')
                verification['full_release_revalidation'] = 'not rerun; edited rows have individual native reload/cache validation'
                verification['authored_asset_model_count'] = None
                verification['authored_asset_model_count_status'] = 'needs_recompute_after_human_edit'
                verification['pinned_runtime_count'] = len({row['runtime_root'] for row in document['tasks']})
                # The old file ledger must not claim that the edited release
                # still has its original human_ready/manifest SHA.
                new_ready = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False).encode()
                verification.setdefault('files', {})['human_ready.json'] = hashlib.sha256(new_ready).hexdigest()
                if manifest_path.exists():
                    verification['files']['manifest.json'] = digest(manifest_path)
                for name in list(verification['files']):
                    if name not in ('human_ready.json', 'manifest.json') and (directory/name).is_file():
                        verification['files'][name] = digest(directory/name)
                atomic(verification_path, verification)
            atomic(path, document)  # authoritative commit marker
        except Exception:
            for name, content in snapshots.items():
                (directory/name).write_bytes(content)
            raise
        atomic(revision/'publication.json', dict(published_at=time.time(), parent=original['parent_task_id'],
            dataset=str(directory), previous=previous, replacement=replacement, backup=str(backup)))


class EditorState(SharedState):
    def __init__(self, task):
        super().__init__(task)
        self.commands = deque()
        self.catalog = None
        self.status['editor'] = dict(enabled=True, dirty=False, selected=None, saving=False, message='正在加载原生编辑场景')

    def enqueue(self, body):
        if not isinstance(body, dict) or body.get('op') not in ('select', 'add', 'move', 'delete', 'open', 'close', 'rotate', 'save'):
            raise ValueError('未知编辑命令')
        if len(json.dumps(body)) > 4096:
            raise ValueError('编辑命令过长')
        if body['op'] in ('select', 'add', 'move'):
            screen_ray(type('C', (), dict(position=[0,0,0], basis=lambda _: ([1,0,0],[0,-1,0],[0,0,1])))(), body.get('xy'), 75, 1)
        with self.condition:
            if len(self.commands) >= 8:
                raise ValueError('请等待上一项编辑完成')
            identifier = uuid.uuid4().hex
            self.commands.append(dict(body, command_id=identifier))
            return identifier


class EditorHandler(Handler):
    def do_GET(self):
        route = urlsplit(self.path)
        if route.path == '/':
            return self.reply(Path(__file__).with_suffix('.html').read_bytes(), 'text/html; charset=utf-8')
        if route.path == '/editor/search':
            try:
                if self.state.catalog is None:
                    return self.reply(dict(error='资产索引尚在加载'), code=503)
                query = parse_qs(route.query)
                offset = max(0, min(100000, int(query.get('offset', ['0'])[0])))
                return self.reply(self.state.catalog.search(query.get('q', [''])[0], offset))
            except (ValueError, OSError) as error:
                return self.reply(dict(error=explain_error(error, 'search')['message']), code=400)
        return super().do_GET()

    def do_POST(self):
        if urlsplit(self.path).path != '/editor/command':
            return super().do_POST()
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 4096:
                raise ValueError('编辑命令大小非法')
            command_id = self.state.enqueue(json.loads(self.rfile.read(size)))
            self.reply(dict(ok=True, command_id=command_id), code=202)
        except (ValueError, TypeError) as error:
            self.reply(dict(error=explain_error(error, 'edit')['message']), code=400)


class NativeEditor:
    def __init__(self, args, task, spec, backend, state, camera, report, sensor):
        from rummagebench.authoring.tasks.household_state import load_frozen_member
        self.args, self.task, self.spec, self.backend, self.state, self.report = args, dict(task), spec, backend, state, report
        self.source_frozen = load_frozen_member(Path(spec.scene.frozen_member_manifest), spec)
        self.removed = set(self.source_frozen.get('builder', {}).get('editor_removed_scene_objects', []))
        self.native_placements = dict(self.source_frozen.get('builder', {}).get('editor_native_placements', {}))
        self.selected = None; self.dirty = False; self.pending = None; self.logs = []
        self.error_log = ROOT/"runs/spectator_editor_errors"/(time.strftime("%Y%m%d_%H%M%S")+f"_{os.getpid()}.jsonl")
        self.camera = camera
        self.sensor = sensor; self.render_frame = None
        import omni.replicator.core as rep
        if 'depth_linear' not in sensor.modalities:
            sensor.add_modality('depth_linear')
        self.instance_annotator = rep.AnnotatorRegistry.get_annotator('instance_id_segmentation_fast')
        with backend._sim.editing_usd():
            self.instance_annotator.attach([sensor.render_product])
        state.catalog = AssetIndex(Path(os.environ['OMNIGIBSON_DATA_PATH'])/'behavior-1k-assets/objects')
        self.accept_world()
        self.status(message='编辑模式：拖放资产，或点选场景里的物体', fov=args.fov, aspect=args.width/args.height,
                    width=args.width, height=args.height, tool_sha256=digest(__file__), placement_mode="gravity_rest_v2")

    def status(self, **values):
        current = self.state.status['editor']
        current.update(dirty=self.dirty, saving=self.pending is not None, **values)

    def failure(self, error, op):
        import traceback
        explanation = explain_error(error, op)
        identifier = uuid.uuid4().hex[:10]
        entry = dict(at=time.time(), id=identifier, op=op, **explanation,
                     detail=str(error), traceback=traceback.format_exc())
        self.error_log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.error_log.open('a') as stream:
                stream.write(json.dumps(entry, ensure_ascii=False, default=str)+'\n')
        except OSError:
            traceback.print_exc()  # still show the useful explanation if disk is full
        public = dict(id=identifier, **explanation, detail=str(error), log=str(self.error_log))
        history = self.state.status['editor'].get('errors', [])[-9:]+[public]
        self.status(message=explanation['message'], last_error=public, errors=history)
        print('EDITOR_ERROR '+json.dumps(entry, ensure_ascii=False, default=str), flush=True)
        return explanation['message']

    def object(self, name):
        obj = self.backend._env.scene.object_registry('name', name)
        if obj is None:
            raise ValueError('物体不存在')
        return obj

    def snapshot(self):
        return deepcopy(self.backend._sim.dump_state(serialized=False))

    def accept_world(self):
        from spectator import physical_fingerprint
        self.tick_value = self.backend._sim.current_time_step_index
        self.fingerprint = physical_fingerprint(self.backend)
        self.report['physics_tick'] = self.tick_value

    def audit(self, full):
        from spectator import physical_fingerprint
        if self.backend._sim.current_time_step_index != self.tick_value or full and physical_fingerprint(self.backend) != self.fingerprint:
            raise RuntimeError('编辑命令之外的物理状态发生变化')

    def ray(self, xy, ignore=None):
        from omnigibson.utils.sampling_utils import raytest
        origin, direction = screen_ray(self.camera, xy, self.args.fov, self.args.width/self.args.height)
        if self.render_frame is not None:
            import numpy as np
            frame = self.render_frame
            origin = frame['position']
            point = calibrated_point(origin, frame['basis'], frame['intrinsic'],
                                     xy[0]*self.args.width, xy[1]*self.args.height, 1.)
            vector = point-np.asarray(origin); direction = vector/np.linalg.norm(vector)
        ignored = None
        if ignore:
            obj = self.object(ignore); ignored = [obj.prim_path, *[link.prim_path for link in obj.links.values()]]
        hit = raytest(origin, [origin[i]+direction[i]*100 for i in range(3)], ignore_bodies=ignored)
        visible = self.render_hit(xy) if ignore is None else None
        if visible is not None:
            # Selection must match the image. A moved/sleeping cabinet's
            # PhysX query proxy may still refer to its pre-restore pose.
            return visible
        if not hit or not hit.get('hit'):
            raise ValueError('这里没有可选中的表面')
        obj = self.backend._owning_object(hit.get('rigidBody') or hit.get('collision') or '')
        if obj is None:
            raise ValueError('这个表面不属于可编辑物体')
        return obj, hit

    def frame(self, data):
        import numpy as np
        from rummagebench.sim.omnigibson.raw_instance import _array, canonical_labels
        raw = self.instance_annotator.get_data()
        depth = data.get('depth_linear')
        if not isinstance(raw, dict) or depth is None or 'data' not in raw:
            self.render_frame = None; return
        ids, depth = np.asarray(raw['data']), np.asarray(_array(depth)).squeeze()
        if ids.shape != (self.args.height,self.args.width) or depth.shape != ids.shape:
            self.render_frame = None; return
        roots = {str(obj.prim_path):obj.name for obj in self.backend._env.scene.objects}
        self.render_frame = dict(ids=ids.copy(), depth=depth.copy(),
            labels=canonical_labels(raw.get('info',{}).get('idToLabels'),roots),
            position=list(self.camera.position), basis=self.camera.basis(), intrinsic=_array(self.sensor.intrinsic_matrix).tolist())
        self.status(intrinsics=self.render_frame['intrinsic'])

    def render_hit(self, xy):
        import numpy as np
        if self.render_frame is None:
            return None
        frame = self.render_frame; width,height=self.args.width,self.args.height
        x,y=min(width-1,int(xy[0]*width)),min(height-1,int(xy[1]*height))
        name=frame['labels'].get(str(int(frame['ids'][y,x])))
        obj=self.backend._env.scene.object_registry('name',name) if name else None
        if obj is None:
            return None
        def point(px,py):
            depth=float(frame['depth'][py,px])
            if not math.isfinite(depth) or not 0 < depth < 100:
                raise ValueError('当前像素深度不可用，请换一个点')
            return calibrated_point(frame['position'],frame['basis'],frame['intrinsic'],px+.5,py+.5,depth)
        p=point(x,y);normal=np.cross(point(min(width-1,x+2),y)-p,point(x,min(height-1,y+2))-p)
        length=np.linalg.norm(normal);normal=normal/length if length>1e-10 else np.zeros(3)
        # The visible face normal points towards the camera. Flipping every
        # negative-Z normal would reverse a vertical cabinet back arbitrarily.
        if np.dot(normal, p-np.asarray(frame['position'])) > 0:normal=-normal
        return obj,dict(hit=True,position=p.tolist(),normal=normal.tolist(),source='same-camera renderer instance/depth')

    def selection(self):
        from omnigibson.object_states import Open
        if self.selected is None:
            return None
        obj = self.object(self.selected)
        box = self.backend.entity_aabb(obj.name)
        return dict(name=obj.name, category=obj.category, model=getattr(obj, 'model', ''), aabb=box,
            can_open=Open in obj.states, is_open=bool(obj.states[Open].get_value()) if Open in obj.states else False,
            can_delete=obj.name not in protected_entities(self.spec.model_dump(mode='json')) and obj.category not in ('floors','walls','ceilings'))

    def retain(self, before, changed):
        sim = self.backend._sim
        sim.load_state(compose_named_state(before, sim.dump_state(serialized=False), changed), serialized=False)
        self.backend._collision_body_cache = None

    def place(self, obj, support, hit, yaw=0):
        import numpy as np
        import torch
        from omnigibson.utils.sampling_utils import raytest
        from omnigibson.object_states import OnTop, Inside
        from rummagebench.sim.omnigibson.raw_instance import _array
        clicked = list(map(float, hit['position']))
        current = _array(obj.get_position_orientation()[1]).tolist()
        orientation = stable_orientation(getattr(obj, 'orientations', None), yaw, current)
        obj.set_position_orientation(orientation=orientation)
        # Fabric restores poses before PhysX updates its query proxies. This
        # step belongs to the edit transaction; unrelated bodies are restored.
        self.backend._sim.step()
        ignored = [obj.prim_path, *[link.prim_path for link in obj.links.values()]]
        lo, hi = (np.asarray(_array(v), float) for v in obj.aabb)
        half = (hi-lo)/2
        anchor = gravity_anchor(clicked, hit.get('normal', [0,0,1]), half)
        support, hit = downward_surface(anchor, raytest, self.backend._owning_object, ignored)
        if support is obj or support in self.backend._env.scene.robots:
            raise EditError('invalid_support', '这里是机器人或正在移动的物体，不能作为落点。请换到桌面、层板或地板上方。')
        point = np.asarray(_array(hit['position']), float)
        # Do not require all four bounding-box corners to hit the support:
        # round objects/tables can be stable without that conservative proxy.
        # Actual gravity, native relations, drift and velocity decide below.
        box = [point+[-half[0],-half[1],.004], point+[half[0],half[1],2*half[2]+.004]]
        for other in self.backend._env.scene.objects:
            if other is obj or other is support or other.category in ('walls','floors','ceilings'):
                continue
            bounds = self.backend.entity_aabb(other.name)
            if bounds and all(box[0][i]+.002 < bounds[1][i] and bounds[0][i] < box[1][i]-.002 for i in range(3)):
                label = next((name for name, cats in ALIASES.items() if other.category in cats), other.category.replace('_',' '))
                raise EditError('space_occupied', f'这里与{label}占用的空间重叠，按当前姿态放不下。请移开旁边的物品，或换一个更空的落点。')
        position = np.asarray(_array(obj.get_position_orientation()[0]), float)
        destination = position+point+[0,0,float(half[2])+.005]-(lo+hi)/2
        obj.set_position_orientation(position=destination)
        obj.sleep_threshold = 0.
        obj.set_linear_velocity(torch.zeros(3)); obj.set_angular_velocity(torch.zeros(3))
        self.backend.settle(30)
        stable = False
        for _ in range(3):
            previous = np.asarray(_array(obj.get_position_orientation()[0]), float)
            self.backend.settle(15)
            drift = float(np.linalg.norm(np.asarray(_array(obj.get_position_orientation()[0]), float)-previous))
            speed = float(np.linalg.norm(_array(obj.get_linear_velocity())))
            spin = float(np.linalg.norm(_array(obj.get_angular_velocity())))
            stable = drift < .005 and speed < .05 and spin < .3
            if stable:
                break
        on_top = bool(obj.states[OnTop].get_value(support)); inside = bool(obj.states[Inside].get_value(support))
        real = self.backend.entity_aabb(obj.name)
        print('EDITOR_PLACE '+json.dumps(dict(entity=obj.name, support=support.name, clicked=clicked,
            anchor=anchor, projected=point.tolist(), rest_orientation=orientation, final_aabb=real,
            on_top=on_top, inside=inside, stable=stable, drift=drift, speed=speed, spin=spin)), flush=True)
        if not stable or not (on_top or inside):
            raise EditError('not_settled', '物品落下后仍在滑动、翻倒，或没有被托住。请换到更宽、更平的落点；也可以换个型号。')
        center = (np.asarray(real[0])+np.asarray(real[1]))/2
        # A long thin asset can settle onto its side around its bottom contact.
        # Keep that valid equilibrium if it remains on this surface nearby;
        # a fixed 4 cm center limit incorrectly rejects such native rest poses.
        if abs(real[0][2]-point[2]) > .04 or np.linalg.norm(center[:2]-point[:2]) > max(.04, float(np.linalg.norm(half))*1.5):
            raise EditError('fell_from_surface', '物品从预期落点滑走或掉到下一层了。请往承托面的中间放，或先移开挡住它的物品。')
        relation = 'inside' if inside else 'on_top'
        link = self.backend._containing_moving_link(support, obj)
        link_name = next((name for name, value in support.links.items() if value is link), None) if link is not None else None
        return placement_record(obj.name, relation, support.name, point[:2].tolist(), link_name)

    def mutate(self, command):
        from rummagebench.core.scenario import ScenarioSpec
        op = command['op']; scene = self.backend._env.scene; sim = self.backend._sim
        if op == 'select':
            self.selected = self.ray(command['xy'])[0].name
            return '已选中 '+self.selected
        if op == 'save':
            return self.save()
        before = self.snapshot(); old_spec = self.spec; doc = self.spec.model_dump(mode='json', exclude_none=True)
        added = None; changed = set(); old_removed = set(self.removed); old_native_placements = deepcopy(self.native_placements)
        try:
            if op == 'add':
                item = self.state.catalog.item(command.get('category'), command.get('model'))
                support, hit = self.ray(command['xy'])
                from omnigibson.objects import DatasetObject
                name = 'editor_'+uuid.uuid4().hex[:12]
                sim.stop()
                added = DatasetObject(name=name, category=item['category'], model=item['model'], fixed_base=False)
                scene.add_object(added); added.set_position_orientation(position=[100,100,5], orientation=[0,0,0,1])
                sim.play(); sim.step(); self.retain(before, {name})
                placement = self.place(added, support, hit, float(command.get('yaw', 0)))
                doc['objects'].append(dict(name=name, category=item['category'], model=item['model'], fixed_base=False))
                doc['placements'].append(placement); self.selected = name; changed.add(name)
            elif op == 'move':
                obj = self.ray(command['from_xy'])[0] if command.get('from_xy') is not None else self.object(self.selected)
                if obj.name == self.spec.robot.name or obj.category in ('walls','floors','ceilings') or obj.n_joints:
                    raise ValueError('请拖动物品；柜体和机器人不能直接拖动')
                support, hit = self.ray(command['xy'], ignore=obj.name)
                yaw = command.get('yaw')
                placement = self.place(obj, support, hit, float(yaw) if yaw is not None else None)
                if not any(o['name'] == obj.name for o in doc['objects']):
                    # Native scene props retain their original scale and scene
                    # identity; their exact pose lives in the frozen snapshot.
                    self.native_placements[obj.name] = placement
                else:
                    doc['placements'] = [p for p in doc['placements'] if p['entity'] != obj.name]+[placement]
                self.selected = obj.name; changed.add(obj.name)
            elif op == 'delete':
                obj = self.object(self.selected)
                if obj.category in ('walls','floors','ceilings'):
                    raise ValueError('不能删除房屋结构')
                authored = any(o['name'] == obj.name for o in doc['objects'])
                doc = delete_from_document(doc, obj.name)
                if not authored:
                    self.removed.add(obj.name)
                self.native_placements.pop(obj.name, None)
                sim.stop(); scene.remove_object(obj); sim.play(); sim.step(); self.selected = None
            elif op in ('open','close'):
                obj = self.object(self.selected)
                from omnigibson.object_states import Open
                if Open not in obj.states:
                    raise ValueError('选中的物体没有原生 OPEN/CLOSE 能力，请点选柜门或抽屉')
                carried = {item.name for item in scene.objects if self.backend._containing_moving_link(obj, item) is not None}
                if not self.backend.set_open(obj.name, op == 'open'):
                    raise ValueError('柜门开关失败，已恢复原状')
                sim.step(); changed = carried|{obj.name}
                doc['initial_states'][obj.name] = dict(open=op == 'open')
            elif op == 'rotate':
                raise ValueError('旋转角度请在拖放前设置')
            self.spec = ScenarioSpec.model_validate(doc); self.backend._scenario = self.spec
            self.retain(before, changed)
            self.backend.validate_physics_state(); self.backend._validate_spawn_clearance()
            self.dirty = True
            self.logs.append(dict(at=time.time(), command=command))
            return {'add':'已添加物体','move':'已移动物体','delete':'已删除物体','open':'柜门已打开','close':'柜门已关闭'}[op]
        except Exception:
            import traceback
            traceback.print_exc()
            if not sim.is_playing():
                sim.play(); sim.step()
            if added is not None and scene.object_registry('name', added.name) is not None:
                sim.stop(); scene.remove_object(added); sim.play(); sim.step()
            # Deletion is guarded before native removal; a failed post-delete
            # audit requires a scene reload, so never attempt mismatched states.
            current = sim.dump_state(serialized=False)
            if set(before[0]['registry']['object_registry']) == set(current[0]['registry']['object_registry']):
                sim.load_state(before, serialized=False)
            self.spec = old_spec; self.backend._scenario = old_spec; self.removed = old_removed; self.native_placements = old_native_placements
            if self.selected and scene.object_registry('name', self.selected) is None:
                self.selected = None
            raise
        finally:
            self.accept_world()

    def save(self):
        from rummagebench.authoring.tasks.household_state import export_native_member
        from rummagebench.evaluation.goal_checker import check_goal
        from types import SimpleNamespace
        from omnigibson.object_states import OnTop, Inside, Open
        from spectator_cache import export, identity
        from spectator import physical_fingerprint
        if not self.dirty:
            return '当前没有未保存的修改'
        self.backend.validate_physics_state(); self.backend._validate_spawn_clearance()
        for placement in self.spec.placements:
            obj, support = self.object(placement.entity), self.object(placement.receptacle)
            if not self.backend._verify_relation(obj, support, placement.relation, Inside, OnTop):
                raise ValueError('布局关系不成立：'+placement.entity+'；请重新摆放后保存')
        for placement in self.native_placements.values():
            if not self.backend._verify_relation(self.object(placement['entity']), self.object(placement['receptacle']), placement['relation'], Inside, OnTop):
                raise ValueError('原生物品未放稳：'+placement['entity'])
        goal = check_goal(SimpleNamespace(held_object=None), self.spec, self.backend)
        if goal.satisfied or goal.fault:
            raise ValueError('当前布局会使任务初始即成功或目标无效，请调整任务物品位置')
        revision = self.args.asset_run/'editor_revisions'/self.task['parent_task_id']/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        revision.mkdir(parents=True)
        runtime = freeze_runtime(self.task['runtime_root'], ROOT/'build/spectator_editor_runtimes')
        deps = dict(self.source_frozen['dependencies'])
        manifest = runtime/'editor_runtime.json'
        if manifest.exists():
            for ref, sha in json.loads(manifest.read_text())['sha256'].items():
                deps[str(runtime/ref)] = sha
            deps[str(manifest)] = digest(manifest)
        archive = revision/'authoring_sources'; archive.mkdir()
        for file in (Path(__file__), Path(__file__).with_suffix('.html')):
            target = archive/file.name; shutil.copyfile(file, target); deps[str(target)] = digest(target)
        document = self.spec.model_dump(mode='json', exclude_none=True)
        for name in list(document['initial_states']):
            obj = self.object(name)
            if Open in obj.states:
                document['initial_states'][name] = dict(open=bool(obj.states[Open].get_value()))
        from rummagebench.core.scenario import ScenarioSpec
        exported_spec = ScenarioSpec.model_validate(document)
        expected = world_poses(self.backend)
        atomic(revision/'expected_world.private.json', expected)
        atomic(revision/'edit_log.json', self.logs)
        builder = dict(source='human_spectator_editor_v1', parent_frozen_sha256=self.source_frozen['manifest_sha256'],
            editor_removed_scene_objects=sorted(self.removed), editor_native_placements=self.native_placements,
            parent_runtime_root=self.task['runtime_root'])
        export_native_member(self.backend, exported_spec, revision, builder, deps)
        replacement = dict(self.task, runtime_root=str(runtime), play_scenario=str(revision/'play.yaml'),
            scenario=str(revision/'play.yaml'), editor_revision=str(revision), editor_cache_dir=str(self.args.cache_dir),
            stage='human_edited_native_pair_validated', full_skill_completion='unverified')
        fingerprint = physical_fingerprint(self.backend); tick = self.backend._sim.current_time_step_index
        cache, metadata = export(self.args.cache_dir, replacement, self.backend, self.camera,
            dict(physics_tick=tick, initial_physical_sha256=fingerprint))
        if physical_fingerprint(self.backend) != fingerprint or self.backend._sim.current_time_step_index != tick:
            raise ValueError('缓存导出改变了物理场景，未覆盖原数据')
        atomic(revision/'validation.json', dict(status='initialization_validated', source='interactive_native_editor',
            physics_checked=True, relations_checked=True, goals_checked=True, native_motion_test='not run',
            parent_task=self.task, cache=str(cache), physics_tick=tick, process_id=os.getpid()))
        atomic(revision/'pending_task.json', replacement)
        env = os.environ.copy(); env['PYTHONPATH'] = str(runtime/'src'); env['RUMMAGEBENCH_ROOT'] = str(runtime)
        log = (revision/'reload.log').open('w')
        child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--validate', str(revision)],
            env=env, cwd=runtime, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        self.pending = dict(process=child, revision=revision, replacement=replacement, started=time.monotonic())
        self.status(message='正在保存：缓存已生成，独立重载校验中；完成前原数据保持可用')
        return '正在独立重载验证，您可以继续移动相机'

    def tick(self, camera):
        self.camera = camera
        if self.pending:
            pending = self.pending; process = pending['process']
            if time.monotonic()-pending['started'] > 900 and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            if process.poll() is not None:
                try:
                    report_path = pending['revision']/'restore_validation.json'
                    restored = json.loads(report_path.read_text()) if report_path.exists() else {}
                    if process.returncode != 0 or restored.get('status') != 'restore_validated':
                        raise ValueError('独立重载未通过，原数据未覆盖：'+restored.get('error', '详见 '+str(pending['revision']/'reload.log')))
                    publish_revision(self.args.asset_run, self.task, pending['replacement'], pending['revision'])
                    self.task = pending['replacement']; self.state.task = self.task; self.state.status['task'] = self.task
                    from rummagebench.core.scenario import load_scenario
                    from rummagebench.adapters.python_api import prepare_renderer_grounding
                    from rummagebench.authoring.tasks.household_state import load_frozen_member
                    self.spec = load_scenario(self.task['play_scenario']); prepare_renderer_grounding(self.spec)
                    self.backend._scenario = self.spec
                    self.source_frozen = load_frozen_member(Path(self.spec.scene.frozen_member_manifest), self.spec)
                    self.dirty = False; self.logs = []
                    message = '保存完成：已覆盖当前条目，缓存已更新，旧版已备份'
                except Exception as error:
                    message = self.failure(error, 'save')
                self.pending = None; self.status(message=message)
        with self.state.condition:
            command = self.state.commands.popleft() if self.state.commands else None
        if command:
            try:
                if self.pending:
                    raise ValueError('正在保存，请等待完成后再编辑')
                message = self.mutate(command)
                self.status(message=message, last_error=None, last_command=dict(id=command['command_id'], ok=True, op=command['op']))
            except Exception as error:
                self.failure(error, command['op'])
                self.status(last_command=dict(id=command['command_id'], ok=False, op=command['op']))
        self.status(selected=self.selection())

    def close(self):
        if self.pending and self.pending['process'].poll() is None:
            os.killpg(self.pending['process'].pid, signal.SIGTERM)
            try:
                self.pending['process'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.pending['process'].pid, signal.SIGKILL)
                self.pending['process'].wait()


def world_poses(backend):
    def values(value):
        return [float(v) for v in value]
    return {obj.name: dict(links={name: [values(v) for v in link.get_position_orientation()] for name, link in obj.links.items()},
        joints=values(obj.get_joint_positions()) if obj.n_joints else []) for obj in backend._env.scene.objects}


def compare_world(expected, actual):
    import numpy as np
    if set(expected) != set(actual):
        raise ValueError('重载后的物体集合不一致')
    maxima = dict(position_m=0., rotation_rad=0., joint=0.)
    for name, row in expected.items():
        if set(row['links']) != set(actual[name]['links']):
            raise ValueError('重载后的 link 集合不一致：'+name)
        for link, (position, quaternion) in row['links'].items():
            p, q = actual[name]['links'][link]
            maxima['position_m'] = max(maxima['position_m'], float(np.linalg.norm(np.array(position)-p)))
            cosine = abs(np.dot(quaternion,q))/(np.linalg.norm(quaternion)*np.linalg.norm(q))
            maxima['rotation_rad'] = max(maxima['rotation_rad'], float(2*np.arccos(np.clip(cosine,0,1))))
        if row['joints']:
            maxima['joint'] = max(maxima['joint'], float(np.max(abs(np.array(row['joints'])-actual[name]['joints']))))
    if maxima['position_m'] > .001 or maxima['rotation_rad'] > .01 or maxima['joint'] > .005:
        raise ValueError('重载姿态与保存不一致：'+json.dumps(maxima))
    return maxima


def validate_revision(revision):
    """Fresh native process. Never rewrites the public dataset or old caches."""
    revision = Path(revision).resolve(); task = json.loads((revision/'pending_task.json').read_text())
    sys.path.insert(0, str(Path(task['runtime_root'])/'src'))
    os.environ['RUMMAGEBENCH_ROOT'] = task['runtime_root']; os.chdir(task['runtime_root'])
    from rummagebench.core.scenario import load_scenario
    from rummagebench.adapters.python_api import prepare_renderer_grounding
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    backend = OmniGibsonBackend(seed=task.get('seed',0))
    report = dict(status='failed', process_id=os.getpid())
    try:
        spec = load_scenario(task['play_scenario']); prepare_renderer_grounding(spec)
        backend.setup(spec)
        report['world_match'] = compare_world(json.loads((revision/'expected_world.private.json').read_text()), world_poses(backend))
        backend.validate_physics_state(); backend._validate_spawn_clearance()
        from omnigibson.object_states import Open
        for name, state in spec.initial_states.items():
            if state.open is not None and bool(backend._env.scene.object_registry('name', name).states[Open].get_value()) != state.open:
                raise ValueError('柜门初始状态不一致：'+name)
        report.update(status='restore_validated', runtime=task['runtime_root'], native_motion_test='not run', full_skill_completion='unverified')
    except Exception as error:
        import traceback
        traceback.print_exc(); report['error'] = str(error)
    atomic(revision/'restore_validation.json', report)
    print('EDITOR_RELOAD', report, flush=True)
    os._exit(0 if report['status']=='restore_validated' else 1)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--validate':
        validate_revision(sys.argv[2])
    else:
        raise SystemExit('Use: python scripts/spectator.py --edit --task F001')
