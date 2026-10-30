(function () {
  'use strict';
  const model = globalThis.IoTBenchRanking;
  const $ = id => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  const state = { board: params.get('board') === 'qa' ? 'qa' : 'imu', mode: params.get('tab') === 'cross-dataset' ? 'cross' : params.get('tab') === 'standard' ? 'unified' : 'reported', dataset: params.get('dataset') || params.get('target') || '', protocol: params.get('protocol') || 'all', metric: params.get('metric') || 'primary', view: params.get('view') === 'overall' ? 'overall' : params.get('view') === 'overview' || params.get('dataset') === 'core' ? 'overview' : 'ranking', aggregateScope: params.get('aggregate') || '', basket: params.get('basket') || 'shared', search: '', order: 'best' };
  let data = null, rows = [], shown = [];
  const CORE_DATASETS = ['uci_har', 'wisdm', 'pamap2', 'hhar', 'opportunity_a'];
  const labels = { uci_har: 'UCI-HAR', wisdm: 'WISDM · 6 classes', pamap2: 'PAMAP2', hhar: 'HHAR', opportunity_a: 'OPPORTUNITY · 17 gestures · no Null', opportunity_gesture_18c: 'OPPORTUNITY · 17 gestures + Null', opportunity_gesture_17c: 'OPPORTUNITY · 17 gestures · no Null', realworld_har: 'RealWorld', usc_had: 'USC-HAD' };
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const safeUrl = value => { try { const u = new URL(IoTBenchPath(value)); return ['http:', 'https:'].includes(u.protocol) && !u.username && !u.password ? u.href : ''; } catch { return ''; } };
  const link = (url, text) => safeUrl(url) ? `<a href="${esc(safeUrl(url))}" target="_blank" rel="noopener noreferrer">${esc(text)} <span aria-hidden="true">↗</span></a>` : '';
  const nameOf = id => labels[id] || data?.qa.benchmarks?.find(b => b.id === id)?.name || data?.benchmarks.find(b => b.id === id)?.name || rows.find(r => r.dataset === id)?.datasetLabel || id;
  function score(row) {
    if (!model.finite(row.value)) return '—';
    if (row.unit === 'percent') return `${(row.value * 100).toFixed(2)}<span class="score-unit">%</span>`;
    if (row.unit === 'percentage-points') return `${(row.value * 100).toFixed(2)}<span class="score-unit">pp</span>`;
    const native = row.metric === row.sourceRecord?.primary_metric && row.displayValue != null ? Number(row.displayValue) : row.value;
    return (Number.isFinite(native) ? native : row.value).toLocaleString('en-US', { maximumFractionDigits: 3 });
  }
  const sourceBadge = row => `<span class="source-badge ${row.source === 'run' ? 'source-run' : 'source-paper'}">${row.source === 'run' ? '<i aria-hidden="true"></i>Our run' : 'Paper'}</span>`;
  async function getJson(path, optional = false) {
    const response = await fetch(path, { cache: 'no-store' });
    if (!response.ok) { if (optional) return null; throw new Error(`Unable to load ${path}`); }
    return response.json();
  }
  async function getApi(name) {
    let response = await fetch(`api/${name}.json`, { cache: 'no-store' });
    if (!response.ok) response = await fetch(`api/${name}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Unable to load ${name}`);
    return response.json();
  }
  async function load() {
    $('refresh').disabled = true;
    try {
      const [benchmarks, registry, evidence, reproduction, qa, transfer] = await Promise.all([getApi('benchmarks'), getApi('results'), getJson('data/imu_evidence.json'), getJson('data/imu_reproduction_20261028.json', true), getJson('data/sensor_qa_leaderboard.json'), getJson('data/paper_transfer_evidence.json', true)]);
      data = { benchmarks, registry, evidence, reproduction: reproduction || { cells: [] }, qa, transfer: transfer || { groups: [] } };
      data.imuRows = [...model.imuObservations(registry, evidence, data.reproduction), ...model.mainPaperObservations(data.transfer)];
      data.qaRows = model.qaObservations(qa);
      data.crossRows = model.transferObservations(data.transfer);
      $('last-updated').textContent = data.reproduction.updated_at ? `Runs updated ${new Date(data.reproduction.updated_at).toLocaleString('en-GB', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })}` : 'Latest saved results';
      render();
    } catch (error) {
      $('leaderboard-wrap').innerHTML = '<div class="empty-state" role="alert"><strong>Results could not be loaded.</strong><p>Check the connection and use Refresh to try again.</p></div>';
      $('result-summary').textContent = 'Connection unavailable';
      console.error(error);
    } finally { $('refresh').disabled = false; }
  }
  function syncUrl() {
    const u = new URL(location.href);
    for (const key of ['board', 'tab', 'dataset', 'target', 'protocol', 'metric', 'view', 'track', 'modality', 'aggregate', 'basket']) u.searchParams.delete(key);
    u.searchParams.set('board', state.board);
    if (state.mode !== 'reported' && state.board === 'imu') u.searchParams.set('tab', state.mode === 'cross' ? 'cross-dataset' : 'standard');
    if (state.view === 'overall') { u.searchParams.set('view', 'overall'); u.searchParams.set('aggregate', state.aggregateScope); u.searchParams.set('basket', state.basket); }
    else if (state.view === 'overview' && state.board === 'imu' && state.mode === 'reported') u.searchParams.set('view', 'overview');
    else if (state.dataset) u.searchParams.set('dataset', state.dataset);
    if (state.protocol !== 'all' && state.view !== 'overall') u.searchParams.set('protocol', state.protocol);
    if (state.metric !== 'primary') u.searchParams.set('metric', state.metric);
    history.replaceState({}, '', u);
  }
  function option(value, label) { return `<option value="${esc(value)}">${esc(label)}</option>`; }
  const matchesSearch = method => `${method.name} ${state.mode === 'cross' ? '' : `${method.venue || ''} ${method.modality || ''}`} ${method.id === 'onehar' ? 'One Model to Fit Them All oneHAR CDNet' : ''}`.toLowerCase().includes(state.search.toLowerCase());
  function renderCrossPapers() {
    const controls = $('cross-paper-controls');
    if (!controls) return;
    controls.hidden = state.board !== 'imu' || state.mode !== 'cross';
    if (controls.hidden) return;
    const selected = state.view === 'overall' ? state.aggregateScope : state.protocol;
    const active = data.transfer.groups.find(g => g.id === selected)?.method_id;
    const papers = [...new Map(data.transfer.groups.map(g => [g.method_id, g])).values()];
    controls.innerHTML = '<span>Paper</span>' + papers.map(g => `<button type="button" data-cross-paper="${esc(g.method_id)}" class="${active === g.method_id ? 'active' : ''}" aria-pressed="${active === g.method_id}">${esc(g.method_id === 'onehar' ? 'One Model to Fit Them All (oneHAR)' : g.method_name)}</button>`).join('');
  }
  function render() {
    if (!data) return;
    if (state.view === 'overview' && (state.board !== 'imu' || state.mode !== 'reported')) state.view = 'ranking';
    $('board-switch').querySelectorAll('button').forEach(b => { b.classList.toggle('active', b.dataset.board === state.board); b.setAttribute('aria-pressed', String(b.dataset.board === state.board)); });
    $('mode-controls').hidden = state.board !== 'imu';
    $('mode-description').hidden = state.board !== 'imu';
    $('mode-description').textContent = {
      reported: 'Literature reference: published scores take priority. Our completed runs fill gaps; evaluation settings may differ between papers.',
      unified: 'Measured comparison: our runs use shared dataset splits and metrics. Only completed runs are shown; implementation and paper-calibration status remain in Details.',
      cross: 'Transfer results from the cited papers, grouped by target label set, training sources and label budget.'
    }[state.mode];
    $('view-controls').hidden = false;
    $('mode-controls').querySelectorAll('button').forEach(b => { b.classList.toggle('active', b.dataset.mode === state.mode); b.setAttribute('aria-pressed', String(b.dataset.mode === state.mode)); });
    $('view-controls').querySelectorAll('button').forEach(b => { b.hidden = b.dataset.view === 'overview' && (state.board !== 'imu' || state.mode !== 'reported'); b.classList.toggle('active', b.dataset.view === state.view); b.setAttribute('aria-pressed', String(b.dataset.view === state.view)); });
    $('board-title').textContent = state.board === 'qa' ? 'Sensor question answering' : 'IMU activity recognition';
    rows = state.board === 'qa' ? data.qaRows : state.mode === 'cross' ? data.crossRows : data.imuRows;
    if (state.mode === 'unified' && state.board === 'imu') rows = rows.filter(r => r.protocol === 'raw-v2');
    $('basket-field').hidden = state.view !== 'overall';
    $('dataset-field').hidden = false;
    $('protocol-field').hidden = false;
    $('metric-field').hidden = false;
    $('protocol-label').textContent = state.view === 'overall' ? 'Score source' : 'Protocol';
    $('sort-results').hidden = false;
    if (state.view === 'overall') { renderOverall(); renderCrossPapers(); syncUrl(); return; }
    // Old released-test links continue to resolve to their distinct measured track.
    if (state.dataset === 'onehar-released') { state.dataset = 'uci_har'; state.protocol = 'onehar_released_checkpoint'; }
    const cross = state.board === 'imu' && state.mode === 'cross';
    const crossProtocols = cross ? data.transfer.groups.map(g => [g.id, g.label]) : [];
    if (cross && state.protocol !== 'all' && !crossProtocols.some(([id]) => id === state.protocol)) state.protocol = 'all';
    const datasetRows = cross && state.protocol !== 'all' ? rows.filter(r => r.protocol === state.protocol) : rows;
    const ids = [...new Set(datasetRows.map(r => r.dataset))];
    const defaultId = state.board === 'qa' ? 'quants_binary' : state.mode === 'cross' ? 'uci_har_6c' : 'uci_har';
    if (!ids.includes(state.dataset)) state.dataset = ids.includes(defaultId) ? defaultId : ids[0] || '';
    ids.sort((a, b) => {
      const ai = CORE_DATASETS.indexOf(a), bi = CORE_DATASETS.indexOf(b);
      return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi) || nameOf(a).localeCompare(nameOf(b));
    });
    $('dataset-select').innerHTML = ids.map(id => option(id, nameOf(id))).join('');
    $('dataset-select').value = state.dataset;
    const overview = state.board === 'imu' && state.mode === 'reported' && state.view === 'overview';
    $('dataset-field').hidden = overview;
    $('protocol-field').hidden = overview;
    $('metric-field').hidden = overview;
    $('sort-results').hidden = overview;
    if (overview) { renderOverview(); syncUrl(); return; }
    let visible = rows.filter(r => r.dataset === state.dataset);
    const protocols = cross ? crossProtocols : [...new Map(visible.map(r => [r.protocol, r.protocolLabel])).entries()];
    if (!protocols.some(([id]) => id === state.protocol)) state.protocol = 'all';
    $('protocol-select').innerHTML = option('all', 'All protocols · grouped') + protocols.map(([id, name]) => option(id, name)).join('');
    $('protocol-select').value = state.protocol;
    if (state.protocol !== 'all') visible = visible.filter(r => r.protocol === state.protocol);
    const metrics = [...new Set(visible.flatMap(r => state.board === 'qa' ? [r.metric] : [r.metric, ...Object.keys(r.metrics || {}).filter(k => !k.endsWith('_std') && ['accuracy', 'f1_macro', 'f1_weighted', 'f1_score'].includes(k))]))];
    if (!metrics.includes(state.metric)) state.metric = 'primary';
    $('metric-select').innerHTML = option('primary', 'Reported primary metric') + metrics.map(k => option(k, model.metricName(k))).join('');
    $('metric-select').value = state.metric;
    if (state.metric !== 'primary') visible = visible.map(r => model.forMetric(r, state.metric)).filter(r => r && model.finite(r.value));
    if (state.search) visible = visible.filter(r => matchesSearch(r.method));
    shown = visible;
    const groups = model.groupRows(visible, state.order);
    const count = new Set(visible.map(r => r.method.id)).size;
    const measured = visible.filter(r => r.source === 'run').length;
    $('result-summary').textContent = `${count} models · ${groups.length} protocol${groups.length === 1 ? '' : 's'} · ${measured} measured result${measured === 1 ? '' : 's'}`;
    $('dataset-heading').textContent = nameOf(state.dataset);
    $('context-note').textContent = state.mode === 'cross' && state.board === 'imu' ? 'Scores are sorted within each paper’s comparison setting. Training sources and label budgets stay separate.' : 'Ranked within the same protocol and metric. Open a result for the paper value and run details.';
    if (state.board === 'imu' && state.dataset === 'opportunity_a' && visible.some(r => r.protocol === 'raw-v2')) {
      $('context-note').textContent = 'Custom subject-held-out evaluation: 2 train / 1 validation / 1 test subject. No Null output class; input windows retain Null frames. Paper protocols differ. Open Details for the audit.';
    }
    $('sort-results').textContent = state.order === 'best' ? 'Score ↓' : 'Score ↑';
    $('leaderboard-wrap').innerHTML = groups.length ? groups.map(groupTable).join('') : '<div class="empty-state"><strong>No matching results.</strong><p>Try another dataset, protocol or model name.</p></div>';
    renderCrossPapers();
    syncUrl();
  }
  function groupTable(group) {
    let previous = null, rank = 0;
    return `<section class="ranking-group"><div class="group-heading"><h3>${esc(group.label)}</h3><span>${esc(group.metric)} ${group.rows[0]?.direction === 'lower' ? '↓' : '↑'}</span></div><div class="ranking-scroll"><table class="ranking-table"><caption class="sr-only">${esc(nameOf(state.dataset))}, ${esc(group.label)}, ${esc(group.metric)}</caption><thead><tr><th scope="col" class="rank-col">Rank</th><th scope="col">Model</th><th scope="col" class="score-col">Score</th><th scope="col">Source</th><th scope="col"><span class="sr-only">Result details</span></th></tr></thead><tbody>${group.rows.map((r, i) => {
      if (previous === null || Math.abs(r.value - previous) > 1e-12) rank = i + 1;
      previous = r.value;
      return `<tr class="result-row${rank === 1 && state.order === 'best' ? ' leading' : ''}"><td class="rank-col"><span class="rank-number">${rank}</span></td><th scope="row" class="model-cell"><span class="model-title">${esc(r.method.name)}</span><span class="model-subtitle">${esc([r.method.venue, r.method.year].filter(Boolean).join(' · '))}${r.method.modality ? ` · ${esc(r.method.modality)}` : ''}</span></th><td class="score-col"><span class="main-score">${score(r)}</span>${model.finite(r.uncertainty) ? `<span class="score-uncertainty">± ${(r.uncertainty * (r.unit === 'percent' ? 100 : 1)).toFixed(2)}</span>` : ''}</td><td class="source-col">${sourceBadge(r)}</td><td class="details-col"><button class="details-button" type="button" data-result="${esc(r.id)}" aria-label="Details for ${esc(r.method.name)}">Details <span aria-hidden="true">↗</span></button></td></tr>`;
    }).join('')}</tbody></table></div>${group.rows.some(r => r.review === 'Calibration pending') ? '<p class="group-footnote">Complete measured runs; paper calibration is pending for some implementations. See Details.</p>' : ''}</section>`;
  }
  function renderOverview() {
    let entries = model.overviewRows(data.imuRows, CORE_DATASETS, data.registry);
    if (state.search) entries = entries.filter(r => matchesSearch(r.method));
    shown = entries.flatMap(r => Object.values(r.scores).filter(Boolean));
    $('dataset-heading').textContent = 'Core dataset overview';
    $('result-summary').textContent = `${entries.length} models · 5 datasets · ${shown.filter(r => r.source === 'run').length} measured scores`;
    $('context-note').textContent = 'Paper scores take precedence. Class variants and metrics are shown in each cell; protocols vary. Open Details before comparing.';
    $('leaderboard-wrap').innerHTML = `<div class="overview-scroll"><table class="overview-table"><caption class="sr-only">Selected paper or measured result for each model and dataset. Coverage order; no aggregate rank.</caption><thead><tr><th scope="col">Model</th>${CORE_DATASETS.map(id => `<th scope="col"><button class="dataset-link" data-dataset="${id}">${esc(nameOf(id))} <span aria-hidden="true">↗</span></button></th>`).join('')}</tr></thead><tbody>${entries.map(entry => `<tr><th scope="row" class="model-cell"><span class="model-title">${esc(entry.method.name)}</span><span class="model-subtitle">${esc([entry.method.venue, entry.method.year].filter(Boolean).join(' · '))}</span></th>${CORE_DATASETS.map(id => {
      const r = entry.scores[id];
      return `<td>${r ? `<button class="overview-score" data-result="${esc(r.id)}" aria-label="Details for ${esc(entry.method.name)} on ${esc(r.datasetLabel || nameOf(id))}"><span class="main-score">${score(r)}</span><span class="cell-metric">${esc(r.metricLabel)}</span>${r.datasetLabel ? `<span class="cell-metric">${esc(r.datasetLabel)}</span>` : ''}${r.paperGroup ? `<span class="cell-metric">${esc(r.protocol === 'onehar-joint' ? 'Joint training' : 'Full-shot')}</span>` : ''}${sourceBadge(r)}</button>` : '<span class="score-missing" aria-label="No reported result">—</span>'}</td>`;
    }).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  function overallMissingCell(entry, dataset) {
    if (state.board !== 'imu' || state.mode !== 'reported' || state.aggregateScope !== 'selected') return '<span class="score-missing" aria-label="Missing result">—</span>';
    const variants = { wisdm: ['wisdm_18c'], opportunity_a: ['opportunity_gesture_18c', 'opportunity_4c'], pamap2: ['pamap2_9c'] };
    const candidates = data.imuRows.filter(r => r.method.id === entry.method.id && r.source === 'paper' && variants[dataset]?.includes(r.dataset))
      .map(r => model.forMetric(r, state.metric)).filter(Boolean)
      .sort((a, b) => Number(!['onehar-joint', 'mosaic-full-shot'].includes(a.protocol)) - Number(!['onehar-joint', 'mosaic-full-shot'].includes(b.protocol)));
    const unique = [...new Map(candidates.map(r => [r.dataset, candidates.find(c => c.dataset === r.dataset)])).values()];
    if (!unique.length) return '<span class="score-missing">—</span><span class="cell-metric">Not reported</span>';
    for (const r of unique) if (!shown.some(s => s.id === r.id)) shown.push(r);
    return `<div class="other-label-set">${unique.map(r => `<button class="variant-result" data-result="${esc(r.id)}" aria-label="Different label set: ${esc(entry.method.name)} on ${esc(r.datasetLabel || nameOf(r.dataset))}">${esc(r.sourceRecord?.num_classes ? `${r.sourceRecord.num_classes} classes` : r.datasetLabel || nameOf(r.dataset))} ↗</button>`).join('')}<span class="cell-metric">Other label set<br>excluded from mean</span></div>`;
  }
  function renderOverall() {
    const cross = state.board === 'imu' && state.mode === 'cross';
    let pool = rows;
    let scopes;
    if (state.board === 'qa') scopes = [['qa', 'Published QA results']];
    else if (cross) scopes = data.transfer.groups.map(g => [g.id, g.label]);
    else if (state.mode === 'unified') scopes = [['raw-v2', 'Unified evaluation · 5 seeds']];
    else {
      const protocols = new Map();
      for (const r of data.imuRows) {
        if (!protocols.has(r.protocol)) protocols.set(r.protocol, { name: r.protocolLabel, ids: new Set() });
        protocols.get(r.protocol).ids.add(r.dataset);
      }
      scopes = [['selected', 'Paper priority · measured fallback'], ['raw-v2', 'Unified evaluation · 5 seeds'], ['ad-paper', 'A&D paper comparison']];
      for (const [id, p] of protocols) if (id !== 'raw-v2' && p.ids.size >= 2) scopes.push([id, p.name]);
    }
    if (!scopes.some(([id]) => id === state.aggregateScope)) state.aggregateScope = scopes[0]?.[0] || 'selected';
    $('protocol-select').innerHTML = scopes.map(([id, label]) => option(id, label)).join('');
    $('protocol-select').value = state.aggregateScope;
    if (state.aggregateScope === 'selected') pool = pool.filter(r => model.CORE.includes(r.method.id));
    else if (state.aggregateScope === 'ad-paper') pool = pool.filter(r => ['ad_pamap', 'ad_opp'].includes(r.protocol));
    else if (state.aggregateScope !== 'qa') pool = pool.filter(r => r.protocol === state.aggregateScope);
    const keys = state.board === 'qa' ? [...new Set(pool.map(r => r.metric))] : ['accuracy', 'f1_macro', 'f1_weighted', 'f1_score'];
    const available = keys.filter(key => new Set(pool.map(r => model.forMetric(r, key)).filter(Boolean).map(r => r.dataset)).size >= 2);
    if (!available.includes(state.metric)) state.metric = available.includes('accuracy') ? 'accuracy' : available[0] || 'accuracy';
    $('metric-select').innerHTML = available.map(key => option(key, model.metricName(key))).join('');
    $('metric-select').value = state.metric;
    const metricPool = pool.map(r => model.forMetric(r, state.metric)).filter(Boolean);
    let basis = metricPool;
    // The default reference panel follows the datasets jointly completed by the
    // unified cohort, rather than changing the denominator for each model.
    if (state.aggregateScope === 'selected') {
      const raw = metricPool.filter(r => r.protocol === 'raw-v2');
      if (model.sharedDatasets(raw).length >= 2) basis = raw;
    }
    const shared = model.sharedDatasets(basis);
    const allIds = [...new Set(metricPool.map(r => r.dataset))].sort((a, b) => nameOf(a).localeCompare(nameOf(b)));
    const baskets = [['shared', `Shared datasets · ${shared.length}`]];
    if (state.board === 'imu' && !cross) baskets.push(['core', 'Core 5 datasets']);
    baskets.push(['all', `All available · ${allIds.length}`]);
    if (!baskets.some(([id]) => id === state.basket)) state.basket = 'shared';
    $('basket-select').innerHTML = baskets.map(([id, label]) => option(id, label)).join('');
    $('basket-select').value = state.basket;
    const ids = [...(state.basket === 'core' ? CORE_DATASETS : state.basket === 'all' ? allIds : shared)];
    if (state.board === 'imu' && !cross) ids.sort((a, b) => (CORE_DATASETS.includes(a) ? CORE_DATASETS.indexOf(a) : 99) - (CORE_DATASETS.includes(b) ? CORE_DATASETS.indexOf(b) : 99) || nameOf(a).localeCompare(nameOf(b)));
    $('dataset-field').hidden = true;
    $('dataset-heading').textContent = 'Overall leaderboard';
    let entries;
    try { entries = model.aggregateRows(pool, ids, state.metric, state.order, state.board === 'imu' ? model.paperAverages(data.transfer, !cross) : []); }
    catch { $('leaderboard-wrap').innerHTML = '<div class="empty-state"><strong>These scores use different units.</strong><p>Choose a matching metric and evaluation setting to calculate a mean.</p></div>'; $('result-summary').textContent = 'No comparable mean'; return; }
    // Keep the requested two methods in their dataset and full-coverage views.
    if (state.board === 'imu') entries = entries.filter(r => r.complete || !['onehar', 'mosaic'].includes(r.method.id));
    if (state.search) entries = entries.filter(r => matchesSearch(r.method));
    shown = entries.flatMap(r => Object.values(r.scores));
    const complete = entries.filter(r => r.complete).length;
    const published = entries.some(e => e.averageKind === 'paper');
    $('result-summary').textContent = `${complete} ranked models · ${ids.length} datasets · ${published ? 'paper-reported averages' : 'equal dataset weight'}`;
    $('context-note').textContent = state.aggregateScope === 'selected'
      ? 'Paper values take precedence. Reference means may combine publication protocols; choose Unified evaluation for shared dataset splits and metrics.'
      : published ? 'The full target set matches the original table, so its published average is displayed. Details retain per-dataset scores.'
      : 'Arithmetic mean of the same metric across a fixed dataset set. Only models covering every selected dataset receive a rank.';
    $('sort-results').textContent = state.order === 'best' ? 'Average ↓' : 'Average ↑';
    if (ids.length < 2) {
      $('leaderboard-wrap').innerHTML = '<div class="empty-state"><strong>At least two datasets are needed.</strong><p>Choose another source or dataset set with results for the same metric.</p></div>'; return;
    }
    if (!entries.length) {
      $('leaderboard-wrap').innerHTML = '<div class="empty-state"><strong>No matching results for this dataset set.</strong><p>Choose another evaluation setting or dataset set.</p></div>'; return;
    }
    let last = null, rank = 0, partial = false;
    const metricLabel = model.metricName(state.metric);
    $('leaderboard-wrap').innerHTML = `<div class="overall-datasets"><span>Included datasets</span>${ids.map(id => `<span class="dataset-chip">${esc(nameOf(id))}</span>`).join('')}</div>${!complete ? '<p class="overall-empty-note">No model covers the full selected set yet. Available-only means are shown below without ranks.</p>' : ''}<div class="overview-scroll"><table class="overview-table overall-table"><caption class="sr-only">${published ? 'Paper-reported averages, where the complete target set matches; otherwise calculated' : 'Equal-weight'} ${esc(metricLabel)} across ${esc(ids.map(nameOf).join(', '))}. Incomplete coverage is unranked.</caption><thead><tr><th scope="col">Rank</th><th scope="col">Model</th><th scope="col">Avg. ${esc(metricLabel)}</th>${ids.map(id => `<th scope="col">${esc(nameOf(id))}</th>`).join('')}<th scope="col">Coverage</th></tr></thead><tbody>${entries.map((entry, i) => {
      let divider = '';
      if (!entry.complete && !partial) { partial = true; divider = `<tr class="partial-divider"><th colspan="${ids.length + 4}">Partial coverage · available-only mean · unranked</th></tr>`; }
      if (entry.complete && (last === null || Math.abs(entry.mean - last) > 1e-12)) rank = i + 1;
      if (entry.complete) last = entry.mean;
      const average = score({ value: entry.mean, metric: state.metric, unit: entry.unit });
      return divider + `<tr class="${entry.complete ? 'overall-complete' : 'overall-partial'}${entry.complete && rank === 1 && state.order === 'best' ? ' leading' : ''}"><td><span class="rank-number">${entry.complete ? rank : '—'}</span></td><th scope="row" class="model-cell"><span class="model-title">${esc(entry.method.name)}</span><span class="model-subtitle">${esc([entry.method.venue, entry.method.year].filter(Boolean).join(' · '))}</span></th><td class="overall-mean"><span class="main-score">${average}</span><span class="cell-metric">${entry.averageKind === 'paper' ? 'paper-reported average' : entry.complete ? `mean of ${entry.total} datasets` : `available ${entry.coverage} only`}</span></td>${ids.map(id => {
        const r = entry.scores[id];
        return `<td>${r ? `<button class="overview-score" data-result="${esc(r.id)}" aria-label="Details for ${esc(entry.method.name)} on ${esc(nameOf(id))}"><span class="main-score">${score(r)}</span>${sourceBadge(r)}</button>` : overallMissingCell(entry, id)}</td>`;
      }).join('')}<td><span class="coverage-badge ${entry.complete ? 'coverage-full' : ''}">${entry.coverage} / ${entry.total}</span><span class="cell-metric">${entry.sourceRuns ? `${entry.sourceRuns} run${entry.sourceRuns === 1 ? '' : 's'}` : ''}${entry.sourceRuns && entry.sourcePapers ? ' · ' : ''}${entry.sourcePapers ? `${entry.sourcePapers} paper` : ''}</span></td></tr>`;
    }).join('')}</tbody></table></div><p class="overall-formula">${published ? 'Paper-reported averages are copied from the original table when the full dataset set and protocol match; its aggregation rule is not inferred. ' : ''}Calculated means use equal dataset weight. Partial rows use only their available count and remain unranked; missing scores are never zero.</p>`;
  }
  function openResult(id) {
    const r = shown.find(x => x.id === id);
    if (!r) return;
    $('detail-title').textContent = r.method.name;
    $('detail-context').textContent = `${nameOf(r.dataset)} · ${r.metricLabel}`;
    const choice = r.choice;
    const gap = choice?.gap;
    const measuredValue = choice?.measured ? model.metricValue(choice.measured, choice.paper.primary_metric) : null;
    let comparison = '';
    if (choice) {
      comparison = `<section class="comparison-panel"><h3>Score selection</h3><dl><div><dt>Paper</dt><dd>${(choice.paper.primary_score * 100).toFixed(2)}%</dd></div><div><dt>Matching run</dt><dd>${model.finite(measuredValue) ? `${(measuredValue * 100).toFixed(2)}%` : 'Not available'}</dd></div>${model.finite(gap) ? `<div><dt>Absolute difference</dt><dd>${(gap * 100).toFixed(2)} pp</dd></div>` : ''}</dl><p>The paper value is displayed whenever the corresponding published result is available. Saved runs and their differences remain here for comparison.</p></section>`;
    }
    const record = r.sourceRecord || {};
    const artifact = record.run_source;
    const otherRuns = choice?.alternatives?.filter(a => a.id !== choice.measured?.id) || [];
    $('detail-body').innerHTML = `<div class="detail-score">${sourceBadge(r)}<span class="main-score">${score(r)}</span></div><dl class="result-facts"><div><dt>Protocol</dt><dd>${esc(r.protocolLabel)}</dd></div><div><dt>Setting</dt><dd>${esc(r.protocolNote || 'See the cited source for the original evaluation setting.')}</dd></div>${r.sourceTitle ? `<div><dt>Source</dt><dd>${esc(r.sourceTitle)}</dd></div>` : ''}${r.review ? `<div><dt>Review</dt><dd>${esc(r.review)}</dd></div>` : ''}${r.locator ? `<div><dt>Source location</dt><dd>${esc(typeof r.locator === 'string' ? r.locator : JSON.stringify(r.locator))}</dd></div>` : ''}${record.standalone_artifact_found === false ? '<div><dt>Run evidence</dt><dd>Saved registry observation; a standalone run artifact was not located. No independent verification is asserted.</dd></div>' : ''}${artifact ? `<div><dt>Saved artifact</dt><dd><code>${esc(artifact.path)}</code></dd></div>` : ''}${record.completed_folds ? `<div><dt>Complete runs</dt><dd>${esc(record.completed_folds)} / ${esc(record.required_folds)} · ${esc((record.seeds || []).length)} seeds</dd></div>` : ''}</dl>${comparison}${otherRuns.length ? `<details class="other-runs"><summary>Other saved runs (${otherRuns.length})</summary>${otherRuns.map(a => `<p><strong>${esc(data.evidence.protocols?.[a.protocol_id]?.label || a.protocol_id)}</strong><br>${esc(model.metricName(a.primary_metric))} ${(a.primary_score * 100).toFixed(2)}% · ${esc(a.status)}<br>${esc(a.saved_protocol_note || '')}</p>`).join('')}</details>` : ''}<div class="detail-links">${link(r.paperUrl || r.method.paper, 'Source paper')}${link(r.method.code, 'Code')}${r.auditUrl ? link(r.auditUrl, 'OPPORTUNITY audit · 30 Oct') : ''}<a href="${r.protocol === 'raw-v2' ? 'data/imu_reproduction_20261028.json' : state.board === 'qa' ? 'data/sensor_qa_leaderboard.json' : r.paperGroup || state.mode === 'cross' ? 'data/paper_transfer_evidence.json' : 'data/imu_evidence.json'}" target="_blank" rel="noopener">Result data ↗</a></div>`;
    $('result-dialog').showModal();
  }
  $('board-switch').addEventListener('click', event => {
    const b = event.target.closest('[data-board]'); if (!b) return;
    state.board = b.dataset.board; state.dataset = ''; state.protocol = 'all'; state.metric = 'primary'; state.aggregateScope = ''; state.basket = 'shared'; state.search = ''; $('model-search').value = ''; render();
  });
  $('mode-controls').addEventListener('click', event => {
    const b = event.target.closest('[data-mode]'); if (!b) return;
    state.mode = b.dataset.mode; state.dataset = ''; state.protocol = 'all'; state.metric = 'primary'; state.aggregateScope = ''; state.basket = 'shared'; render();
  });
  $('cross-paper-controls')?.addEventListener('click', event => {
    const button = event.target.closest('[data-cross-paper]');
    if (!button) return;
    const groups = data.transfer.groups.filter(g => g.method_id === button.dataset.crossPaper);
    const group = groups.find(g => g.category === 'zero_shot_transfer') || groups[0];
    if (!group) return;
    state.protocol = group.id;
    state.aggregateScope = group.id;
    state.metric = 'primary';
    state.basket = 'all';
    render();
  });
  $('view-controls').addEventListener('click', event => {
    const b = event.target.closest('[data-view]'); if (!b) return;
    if (state.board === 'imu' && state.mode === 'cross') {
      if (state.view === 'overall') state.protocol = state.aggregateScope;
      else if (state.protocol !== 'all') state.aggregateScope = state.protocol;
    }
    state.view = b.dataset.view; render();
  });
  $('dataset-select').addEventListener('change', event => { state.dataset = event.target.value; if (state.board !== 'imu' || state.mode !== 'cross') state.protocol = 'all'; state.metric = 'primary'; render(); });
  $('protocol-select').addEventListener('change', event => {
    if (state.view === 'overall') state.aggregateScope = event.target.value;
    else state.protocol = event.target.value;
    if (state.board === 'imu' && state.mode === 'cross' && event.target.value !== 'all') state.protocol = state.aggregateScope = event.target.value;
    state.metric = 'primary'; render();
  });
  $('basket-select').addEventListener('change', event => { state.basket = event.target.value; render(); });
  $('metric-select').addEventListener('change', event => { state.metric = event.target.value; render(); });
  $('model-search').addEventListener('input', event => { state.search = event.target.value; render(); });
  $('sort-results').addEventListener('click', () => { state.order = state.order === 'best' ? 'reverse' : 'best'; render(); });
  $('refresh').addEventListener('click', load);
  $('close-dialog').addEventListener('click', () => $('result-dialog').close());
  $('result-dialog').addEventListener('click', event => { if (event.target === $('result-dialog')) $('result-dialog').close(); });
  $('leaderboard-wrap').addEventListener('click', event => {
    const result = event.target.closest('[data-result]'); if (result) { openResult(result.dataset.result); return; }
    const dataset = event.target.closest('[data-dataset]'); if (dataset) { state.dataset = dataset.dataset.dataset; state.view = 'ranking'; state.protocol = 'all'; state.metric = 'primary'; render(); return; }
    const cross = event.target.closest('[data-cross]'); if (cross) { state.mode = 'cross'; state.view = 'ranking'; state.protocol = cross.dataset.cross; state.dataset = ''; state.metric = 'primary'; render(); }
  });
  window.addEventListener('popstate', () => location.reload());
  load();
})();
