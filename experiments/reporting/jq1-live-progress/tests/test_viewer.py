"""Allocated CPU structural contracts for the new saved-only HTML viewer.

These dependency-free fixtures inspect the actual template and its declared
browser contract. They do not claim browser execution, scientific rendering or
probability validation. Real slider, keyboard, playback, empty-step and pointer
update interactions remain mandatory in separate browser presentation QA.
"""

from html.parser import HTMLParser
import json
from pathlib import Path
import re

import pytest


TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'


class Document(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.ids, self.tags, self.scripts, self.text = {}, [], [], []
        self.current_script = None
        self.feed(source)
        self.close()

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        self.tags.append((tag, attrs))
        if 'id' in attrs:
            assert attrs['id'] not in self.ids, 'Duplicate DOM id: ' + attrs['id']
            self.ids[attrs['id']] = (tag, attrs)
        if tag == 'script':
            self.current_script = {'attrs': attrs, 'parts': []}
            self.scripts.append(self.current_script)

    def handle_endtag(self, tag):
        if tag == 'script':
            self.current_script = None

    def handle_data(self, data):
        self.text.append(data)
        if self.current_script is not None:
            self.current_script['parts'].append(data)


@pytest.fixture(scope='module')
def source():
    return TEMPLATE.read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def document(source):
    return Document(source)


@pytest.fixture(scope='module')
def javascript(document):
    scripts = [s for s in document.scripts if s['attrs'].get('type') != 'application/json']
    assert len(scripts) == 1
    return ''.join(scripts[0]['parts'])


@pytest.mark.parametrize('identifier, tag', (
    ('arm-cards', 'nav'), ('arm-select', 'select'), ('global-step', 'input'),
    ('frame-slider', 'input'), ('previous', 'button'), ('next', 'button'),
    ('play', 'button'), ('latest', 'button'), ('metric-select', 'select'),
    ('loss-plot', 'svg'), ('raw-graph', 'svg'), ('atom-heatmap', 'svg'),
    ('bond-heatmap', 'svg'), ('report-error', 'div'),
))
def test_real_template_declares_required_unique_accessible_controls(document, identifier, tag):
    assert document.ids[identifier][0] == tag
    attrs = document.ids[identifier][1]
    if identifier in ('arm-select', 'frame-slider', 'previous', 'next', 'metric-select',
                      'loss-plot', 'raw-graph', 'atom-heatmap', 'bond-heatmap'):
        assert attrs.get('aria-label')


def test_template_has_only_inline_styles_and_scripts_with_no_external_assets(document, source):
    assert source.startswith('<!doctype html>')
    assert '<html lang="en">' in source and '<meta charset="utf-8">' in source
    assert not any(tag in ('iframe', 'object', 'embed') for tag, _ in document.tags)
    for tag, attrs in document.tags:
        if tag in ('script', 'img', 'link'):
            assert 'src' not in attrs and 'href' not in attrs
        assert not any(name.lower().startswith('on') for name in attrs)
    assert '@import' not in source and 'url(' not in source


def test_immutable_edition_configuration_does_not_assume_or_rewrite_a_latest_pointer(document):
    row = next(s for s in document.scripts if s['attrs'].get('id') == 'report-config')
    assert row['attrs']['type'] == 'application/json'
    assert json.loads(''.join(row['parts'])) == {'manifest': 'manifest.json', 'latest': None}


def test_dom_references_are_all_actual_unique_template_elements(document, javascript):
    references = set(re.findall(r'\$\("([A-Za-z0-9-]+)"\)', javascript))
    references |= set(re.findall(r'(?:text|clear)\("([A-Za-z0-9-]+)"', javascript))
    assert references and references <= set(document.ids)


def test_global_numeric_input_is_distinct_from_retained_frame_slider(document, javascript):
    number = document.ids['global-step'][1]
    slider = document.ids['frame-slider'][1]
    assert number['type'] == 'number' and number['step'] == '1'
    assert slider['type'] == 'range' and slider['step'] == '1'
    assert 'arm.frames.steps[Number(e.target.value)]' in javascript
    assert 'arm.frames.steps.indexOf(step)' in javascript
    assert 'r.global_step===value' in javascript


def test_continuation_progress_uses_absolute_target_and_explicit_offset(javascript):
    assert 'a.global_target===a.target_updates' in javascript
    assert 'a.latest_global_step===a.global_start+a.committed_updates' in javascript
    assert 'arm.global_target-arm.global_start' in javascript
    assert 'absolute target' in javascript and 'since ${a.global_start.toLocaleString()}' in javascript
    assert '${a.latest_global_step.toLocaleString()} / ${a.global_target.toLocaleString()}' in javascript


def test_four_arm_manifest_keeps_presentation_and_scientific_outcomes_separate(javascript):
    assert 'value.arms.length===4' in javascript
    assert 'ids.has(a.id)' in javascript
    assert 'pill("Scientific",arm.scientific_status)' in javascript
    assert 'pill("Presentation",arm.presentation_status)' in javascript
    assert 'pill("Progress",arm.progress_status' in javascript
    assert 'statusClass(arm.scientific_status)==="failed"' in javascript


def test_unsaved_global_step_clears_every_previous_molecule_view(document, javascript):
    start = javascript.index('function clearFrame(')
    end = javascript.index('function validateFrame(', start)
    clear = javascript[start:end]
    for identifier in ('atom-heatmap', 'bond-heatmap', 'raw-graph', 'matrix-empty',
                       'graph-empty', 'raw-record', 'download-frame'):
        assert identifier in clear
    assert 'if(index<0)' in javascript
    assert 'No molecule matrix was saved at global update' in javascript
    assert 'Scalar records remain available' in javascript
    assert 'renderScalars();await renderFrame()' in javascript
    assert 'hidden' in document.ids['matrix-empty'][1]


def test_probability_shapes_and_precomputed_hpc_argmax_are_required_without_recalculation(javascript):
    assert 'frame.atoms.length===25' in javascript and 'r.length===6' in javascript
    assert 'frame.bonds.length===300' in javascript and 'r.length===5' in javascript
    assert 'frame.atom_labels?.length===25' in javascript
    assert 'frame.bond_labels?.length===300' in javascript
    assert 'matrix.every(r=>r.every(v=>finite(v)&&v>=0&&v<=1))' in javascript
    assert not re.search(r'\b(?:softmax|argmax|sigmoid|sanitize|repair_graph)\s*\(', javascript)


def test_every_loss_row_is_plotted_without_index_decimation(javascript):
    assert 'data-points":rows.length' in javascript
    assert 'const path=rows.map(' in javascript
    assert 'scalarRows.filter(r=>finite(r.losses?.[metric]))' in javascript
    assert not re.search(r'\b(?:stride|downsample|resample)\s*\(', javascript)
    assert 'r.global_step>scalarRows[i-1].global_step' in javascript


def test_saved_frame_chunks_are_bounded_and_asset_cache_is_bounded(javascript):
    assert 'c.count<=100' in javascript
    assert 'payload.frames.length<=100' in javascript
    assert 'cache.size>6' in javascript and 'cache.delete(cache.keys().next().value)' in javascript
    assert 'epoch!==frameEpoch||arm!==selectedArm' in javascript
    assert 'selection!==armEpoch' in javascript


def test_slider_numeric_focus_keyboard_and_explicit_actions_pause_playback(javascript):
    assert '["pointerdown","keydown","focus"]' in javascript
    assert '["focus","input","keydown"]' in javascript
    assert 'if(manual)pause()' in javascript
    assert 'function adjacent(direction){pause()' in javascript
    assert 'if(document.hidden)pause()' in javascript
    assert 'setInterval(' in javascript and ',650)' in javascript
    assert 'setAttribute("aria-pressed","true")' in javascript
    assert 'setAttribute("aria-pressed","false")' in javascript


def test_saved_selection_is_exported_for_stable_wrapper_pointer_updates(javascript):
    assert 'url.searchParams.set("arm",arm.id)' in javascript
    assert 'url.searchParams.set("step",String(step))' in javascript
    assert 'history.replaceState(null,"",url)' in javascript
    assert 'selection:()=>({arm:arm?.id,step})' in javascript
    assert 'setInterval(pollLatest,60000)' in javascript
    assert 'latest.report_url||latest.report_html||latest.url' in javascript


def test_assets_are_same_origin_relative_paths_and_data_is_not_inserted_as_html(javascript):
    assert '!path.startsWith("/")' in javascript
    assert '!path.split("/").includes("..")' in javascript
    assert 'url.origin===location.origin' in javascript
    assert 'textContent' in javascript
    assert 'innerHTML' not in javascript and 'eval(' not in javascript
    assert 'new Function(' not in javascript and 'document.write(' not in javascript


def test_raw_graph_draws_supplied_edges_and_keeps_invalid_or_disconnected_data_visible(source, javascript):
    assert 'for(const edge of frame.graph.edges)' in javascript
    assert 'nodes=frame.graph.nodes' in javascript
    assert 'Raw graph edge endpoint missing' in javascript
    assert 'without repair' in source
    assert 'Invalid valences, disconnected fragments and absent bonds remain visible.' in source
    assert 'not a predicted pose' in source


def test_all_raw_scalar_frame_manifest_and_provenance_downloads_exist(document, javascript):
    for identifier in ('download-scalars', 'download-frame', 'download-manifest'):
        tag, attrs = document.ids[identifier]
        assert tag == 'a' and 'download' in attrs
    assert document.ids['frame-downloads'][0] == 'div'
    assert 'link.download=""' in javascript
    assert 'JSON.stringify(arm.provenance||arm,null,2)' in javascript
    assert 'JSON.stringify(row,null,2)' in javascript


def test_scope_and_interpretation_caveats_are_short_and_visible(source):
    assert 'contact, affinity and the DBMol objective are different metrics' in source
    assert 'does not turn a failed or incomplete gradient check into a scientific pass' in source
    assert 'no new inference, interpolation, molecule repair or rescoring' in source
    assert 'does not establish binding or activity' in source
    assert 'coefficient changes' in source


def test_loading_errors_remain_visible_without_fabricating_new_progress(javascript, document):
    assert document.ids['report-error'][1]['role'] == 'alert'
    assert 'Report could not load verified saved data' in javascript
    assert 'current verified edition remains displayed' in javascript
    assert 'start().catch(showError)' in javascript


def test_progress_bar_is_a_bounded_block_so_span_fill_cannot_overlay_following_panel(source):
    rule = re.search(r'\.bar\{([^}]+)\}', source)
    assert rule is not None
    declarations = dict(item.split(':', 1) for item in rule.group(1).split(';') if ':' in item)
    assert declarations['display'] == 'block'
    assert declarations['height'] == '5px' and declarations['overflow'] == 'hidden'
    assert '.bar>span{display:block;height:100%;' in source


def test_visible_primary_metrics_are_bounded_without_removing_full_scalar_curves_or_records(source, javascript, document):
    assert 'for(const [name,value]of primaryMetrics(row?.losses||{}))' in javascript
    primary = javascript[javascript.index('function primaryMetrics('):javascript.index('function pretty(')]
    assert '.slice(0,8)' in primary
    assert '!name.startsWith("components.")' in primary and '!name.startsWith("update.")' in primary
    assert 'function availableMetrics(){const keys=new Set();for(const row of scalarRows)' in javascript
    assert 'JSON.stringify(row,null,2)' in javascript and 'const path=rows.map(' in javascript
    assert 'All saved values remain available in the metric selector, full record and download.' in source
    label = next(attrs for tag, attrs in document.tags if tag == 'label' and attrs.get('for') == 'metric-select')
    assert 'metric-picker' in label['class']
    assert '.top{flex-wrap:wrap}' in source
    assert '.metric-picker select{width:100%;max-width:100%;min-width:0;' in source
