import copy,json
from pathlib import Path
import pytest
from rummagebench.datasets.scene_derivation import digest,validate_scene_derivation

def fixture(tmp_path,removed='chair'):
    data={'metadata':{'task':{'inst_to_name':{'jar':'honey'}}},
          'objects_info':{'init_info':{'chair':{},'honey':{}}},
          'state':{'registry':{'object_registry':{'chair':{'pos':1},'honey':{'pos':2}}}}}
    source=tmp_path/'source.json';source.write_text(json.dumps(data))
    edited=copy.deepcopy(data)
    del edited['objects_info']['init_info'][removed]
    del edited['state']['registry']['object_registry'][removed]
    derived=tmp_path/'derived.json';derived.write_text(json.dumps(edited))
    manifest=tmp_path/'manifest.json';doc=dict(scene_model='test',activity_name='store_honey',
        source_template=str(source),source_sha256=digest(source),derived_template=str(derived),
        derived_sha256=digest(derived),removed_objects=[removed]);manifest.write_text(json.dumps(doc))
    return manifest,derived,doc

def test_declared_removal_only(tmp_path):
    m,d,doc=fixture(tmp_path)
    assert validate_scene_derivation(m,'test','store_honey')[0]==str(d)
    with pytest.raises(ValueError,match='binding'):validate_scene_derivation(m,'other','store_honey')
    data=json.loads(d.read_text());data['metadata']['task']['inst_to_name']['jar']='elsewhere';d.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='hash'):validate_scene_derivation(m,'test','store_honey')
    doc['derived_sha256']=digest(d);m.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='beyond'):validate_scene_derivation(m,'test','store_honey')

def test_task_object_protected(tmp_path):
    m,_,_=fixture(tmp_path,'honey')
    with pytest.raises(ValueError,match='task-bound'):validate_scene_derivation(m,'test','store_honey')
