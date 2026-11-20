"""Synthetic source/loaded-state fixtures only; no data or physics acceptance."""
from copy import deepcopy

import pytest

from rummagebench.authoring.tasks.household_locations import search_location_inventory


def fixture():
    def item(tags,generator=None,parent='kitchen'):
        return {'tags':tags,'generator':generator,'relations':[{'parent':parent}] if parent else []}
    objects={'kitchen':item(['Semantics.Kitchen','Semantics.Room'],parent=None),
        'other_room':item(['Semantics.Room'],parent=None),
        'cab':item(['Semantics.Storage','Semantics.Furniture'],'KitchenCabinetFactory'),
        'counter':item(['Semantics.KitchenCounter','Semantics.Furniture'],'KitchenSpaceFactory'),
        'chair':item(['Semantics.Furniture'],'ChairFactory'),
        'outside':item(['Semantics.Storage'],'SingleCabinetFactory',parent='other_room')}
    manifest={'objects':[{'source_id':k,'name':'live_'+k} for k in objects if 'room' not in k and k!='kitchen'],
        'candidate_bindings':[{'source_id':'cab','location_id':'storage_0','name':'live_cab'}]}
    loaded={o['name']:{'openable_joints':['door'] if o['source_id']=='cab' else [],'fillable_volumes':[],'error':None} for o in manifest['objects']}
    return manifest,{'objects':objects},loaded


def test_static_counter_without_joints_does_not_prove_no_search_cavity():
    manifest,source,loaded=fixture()
    report=search_location_inventory(manifest,source,loaded)
    assert report['status']=='pending' and report['unknown_sources']==['counter']
    assert 'outside' not in {entry['source_id'] for entry in report['entries']}
    assert report['candidate_distribution_changed'] is False
    assert report['required_location_ids']==['storage_0']


@pytest.mark.parametrize('evidence',['openable_joints','fillable_volumes'])
def test_unplanned_actual_drawer_rejects_whole_distribution(evidence):
    manifest,source,loaded=fixture()
    loaded['live_counter'][evidence]=['actual_loaded_drawer_or_volume']
    report=search_location_inventory(manifest,source,loaded)
    assert report['status']=='failed' and report['rejections']==['ADDITIONAL_SEARCH_LOCATION:counter']
    assert report['required_location_ids']==['storage_0']  # not pruned/reweighted


def test_no_live_measurements_never_qualify_cpu_inventory():
    manifest,source,_=fixture()
    report=search_location_inventory(manifest,source)
    assert report['gpu_executed'] is False and report['status']=='pending'


def test_required_live_articulation_error_remains_unknown():
    manifest,source,loaded=fixture()
    loaded['live_cab']={'error':'real query unavailable'}
    report=search_location_inventory(manifest,source,loaded)
    assert 'cab' in report['unknown_sources'] and report['status']=='pending'


def test_missing_or_duplicate_source_cannot_be_silently_skipped():
    manifest,source,loaded=fixture()
    manifest['objects']=[o for o in manifest['objects'] if o['source_id']!='counter']
    assert search_location_inventory(manifest,source,loaded)['status']=='failed'
    manifest['candidate_bindings']*=2
    with pytest.raises(ValueError,match='BIJECTIVE'):search_location_inventory(manifest,source,loaded)
