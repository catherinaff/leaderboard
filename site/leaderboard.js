(function () {
  let benchmarks = [];
  let results = [];
  let imuEvidence = { records: [], sources: {}, protocols: {} };
  let imuEvidenceLoaded = false;
  let reproduction = { cells: [] };
  let transferEvidence = { groups: [], papers: {} };
  let crossProtocol = initialCrossProtocol();
  let crossTarget = new URLSearchParams(window.location.search).get('target') || 'all';
  function initialCrossProtocol() { return new URLSearchParams(window.location.search).get('protocol') || 'overview'; }
  let qaBenchmarks = [];
  let visibleQaBenchmarks = [];
  let qaMethods = [];
  let qaTracks = [];
  const initialParams = new URLSearchParams(window.location.search);
  let activeBoard = initialParams.get('board') === 'qa' ? 'qa' : 'imu';
  let activeQaTrack = initialParams.get('track') || 'all';
  let activeQaModality = initialParams.get('modality') || 'all';
  let activeTab = ['paper-coverage', 'standard', 'cross-dataset'].includes(initialParams.get('tab')) ? initialParams.get('tab') : 'paper-coverage';
  let imuView = initialParams.get('view') === 'all' ? 'all' : 'curated';
  let imuDataset = initialParams.get('dataset') || 'core';
  const CORE_DATASETS = ['uci_har', 'wisdm', 'pamap2', 'hhar', 'opportunity_a'];
  const ONEHAR_RELEASED_DATASETS = ['realworld_har', 'hhar', 'uci_har', 'usc_had', 'pamap2'];
  const CORE_TRANSFERS = ['cross_uci_to_hhar', 'cross_hhar_to_uci', 'cross_uci_to_pamap2', 'cross_pamap2_to_uci'];
  // Publication metadata is a display overlay; the original result registry stays intact.
  const IMU_SELECTION = {
    mosaic: { transfer_protocol: 'mosaic-zero-shot', venue: 'IMWUT / UbiComp', year: 2026, url: 'https://dl.acm.org/doi/10.1145/3831643', reason: 'Recent open-ended activity understanding through motion decomposition and semantic fusion. Synthetic-to-unseen transfer and target-supervised settings are documented separately.', score_note: 'Results in Cross-Dataset protocols' },
    onehar: { transfer_protocol: 'onehar-lodo', released_tests: true, venue: 'IMWUT / UbiComp', year: 2025, url: 'https://dl.acm.org/doi/10.1145/3749509', reason: 'Universal IMU activity recognition with LLM-assisted cross-dataset representations. Five released-checkpoint test results are saved as local observations; paper LODO and joint results are documented separately.', score_note: 'Released checkpoint · 5 test sets' },
    robusthar: { venue: 'IJCAI', year: 2025, url: 'https://www.ijcai.org/proceedings/2025/952', reason: 'Recent multi-sensor masked pretraining; missing-data robustness is a separate evaluation setting.' },
    unimts: { venue: 'NeurIPS', year: 2024, url: 'https://papers.nips.cc/paper_files/paper/2024/hash/c290d4373c495b2cad0625d6288260f0-Abstract-Conference.html', reason: 'Recent unified motion pretraining; zero-, few- and full-shot settings must remain separate.' },
    if_convtransformer: { venue: 'IMWUT', year: 2022, url: 'https://dl.acm.org/doi/10.1145/3534584', reason: 'Representative IMU fusion and convolution/Transformer architecture.' },
    tinyhar: { venue: 'ISWC', year: 2022, url: 'https://publikationen.bibliothek.kit.edu/1000150216', reason: 'Representative compact model for wearable activity recognition.' },
    limu_bert: { venue: 'SenSys', year: 2021, url: 'https://tanrui.github.io/pub/LIMU_BERT.pdf', reason: 'Representative self-supervised IMU encoder; 1% labeled-data results are explicitly marked.' },
    attend_and_discriminate: { venue: 'IMWUT', year: 2021, url: 'https://github.com/AdelaideAuto-IDLab/Attend-And-Discriminate', reason: 'Attention and discriminative feature learning; publication year is 2021 (the preprint was 2020).' },
    deepsense: { venue: 'WWW', year: 2017, url: 'https://arxiv.org/abs/1611.01942', reason: 'Classic convolutional/recurrent multisensor framework.' },
    deepconvlstm: { venue: 'Sensors', year: 2016, url: 'https://www.mdpi.com/1424-8220/16/1/115', reason: 'Classic convolutional/LSTM baseline; included for historical relevance.' }
  };
  let sortCol = 'avg';   // current sort column
  let sortDir = 'desc';  // current sort direction
  const DATA_VERSION = '20261028-onehar-released';

  // ── DOM refs ──────────────────────────────────────────
  const boardSwitch = document.getElementById('board-switch');
  const tabContainer = document.getElementById('task-tabs');
  const note = document.getElementById('leaderboard-note');
  const qaTrackControls = document.getElementById('qa-track-controls');
  const qaTrackSwitch = document.getElementById('qa-track-switch');
  const qaTrackSummary = document.getElementById('qa-track-summary');
  const wrap = document.getElementById('leaderboard-wrap');
  const imuControls = document.getElementById('imu-controls');
  const imuSummary = document.getElementById('imu-summary');
  const imuDatasetSelect = document.getElementById('imu-dataset');
  const imuViewSelect = document.getElementById('imu-view');
  const imuIntro = document.getElementById('imu-intro');
  const crossControls = document.getElementById('imu-cross-controls');
  const crossProtocolSelect = document.getElementById('cross-protocol');
  const crossTargetSelect = document.getElementById('cross-target');
  const crossTargetLabel = document.getElementById('cross-target-label');
  const imuGuide = document.getElementById('imu-guide');
  let imuLoadError = false;

  // ── Fetch data ────────────────────────────────────────
  async function load() {
    try {
      const [bResp, rResp, qaResp, evidenceResp, transferResp, reproductionResp] = await Promise.all([
        fetch(`api/benchmarks?v=${DATA_VERSION}`, { cache: 'no-store' }),
        fetch(`api/results?v=${DATA_VERSION}`, { cache: 'no-store' }),
        fetch(`data/sensor_qa_leaderboard.json?v=${DATA_VERSION}`, { cache: 'no-store' }),
        fetch(`data/imu_evidence.json?v=${DATA_VERSION}`, { cache: 'no-store' }).catch(() => null),
        fetch(`data/paper_transfer_evidence.json?v=${DATA_VERSION}`, { cache: 'no-store' }).catch(() => null),
        fetch('data/imu_reproduction_20261028.json', { cache: 'no-store' }).catch(() => null)
      ]);
      if (!bResp.ok || !rResp.ok) throw new Error('IMU data unavailable');
      benchmarks = await bResp.json();
      results = await rResp.json();
      if (reproductionResp?.ok) {
        try { const data = await reproductionResp.json(); if (Array.isArray(data.cells)) reproduction = data; } catch { /* Optional progress file. */ }
      }
      if (evidenceResp?.ok) {
        const audit = await evidenceResp.json();
        if (Array.isArray(audit.records)) { imuEvidence = audit; imuEvidenceLoaded = true; }
      }
      if (transferResp?.ok) {
        try {
          const transfer = await transferResp.json();
          if (Array.isArray(transfer.groups)) transferEvidence = transfer;
        } catch { /* Keep the existing pairwise registry usable if this optional file is unavailable. */ }
      }
      const qa = qaResp.ok ? await qaResp.json() : { benchmarks: [], methods: [] };
      qaBenchmarks = qa.benchmarks || [];
      qaTracks = qa.tracks || [];
      const qaMethodRecords = [...(qa.methods || []), ...(qa.adapted_methods || []), ...(qa.supplemental_methods || [])];
      qaMethods = normalizeQaMethods(qaMethodRecords, qa.method_taxonomy || {}, qa.leaderboard_method_ids || []);
      if (activeBoard === 'qa' && activeQaTrack !== 'all') sortCol = 'track_order';
    } catch { imuLoadError = true; benchmarks = []; results = []; }
    render();
  }

  function normalizeQaMethods(methods, taxonomy, leaderboardMethodIds) {
    const curatedIds = new Set(leaderboardMethodIds);
    return methods.flatMap(method => {
      const isLeaderboardMethod = curatedIds.size ? curatedIds.has(method.id) : method.qa_core === true;
      if (!isLeaderboardMethod) return [];
      const scores = { ...(method.scores || {}) };
      for (const result of (method.results || [])) {
        if (!result?.benchmark_id) continue;
        const current = scores[result.benchmark_id];
        if (!current || preferQaResult(result, current)) scores[result.benchmark_id] = result;
      }
      const classification = taxonomy[method.id] || taxonomy[method.method_name] || {};
      const displayName = classification.display_name || method.method_name;
      const origin = classification.origin || 'Native';
      const evaluationBenchmark = classification.evaluation_benchmark || classification.native_benchmark || '—';
      return [{ ...method, ...classification, origin, evaluation_benchmark: evaluationBenchmark, source_method_name: method.method_name, method_name: displayName, scores }];
    });
  }

  // ── Render ─────────────────────────────────────────────
  function render() {
    if (crossProtocol !== 'pairwise' && !(crossProtocol === 'overview' && transferEvidence.groups.length) && !transferEvidence.groups.some(g => g.id === crossProtocol)) crossProtocol = transferEvidence.groups.length ? 'overview' : 'pairwise';
    syncBoardChrome();
    if (activeBoard === 'qa') {
      renderQaBoard();
      return;
    }
    const config = tabConfig();
    if (imuLoadError) { wrap.innerHTML = '<p class="imu-empty" role="alert">Unable to load IMU data. Please reload the page or check the local server.</p>'; return; }
    if (config.mode === 'cross') {
      renderCrossControls();
      if (crossProtocol !== 'pairwise') { renderTransferBoard(); return; }
    }
    const categoryBenchmarks = benchmarks.filter(b => b.category === config.category);
    const coreIds = config.mode === 'cross' ? CORE_TRANSFERS : CORE_DATASETS;
    const oneharReleased = config.mode === 'paper' && imuDataset === 'onehar-released';
    if (imuDataset !== 'core' && !oneharReleased && !categoryBenchmarks.some(b => b.id === imuDataset)) imuDataset = 'core';
    const benchList = oneharReleased
      ? ONEHAR_RELEASED_DATASETS.map(id => categoryBenchmarks.find(b => b.id === id)).filter(Boolean)
      : imuDataset === 'core'
      ? coreIds.map(id => categoryBenchmarks.find(b => b.id === id)).filter(Boolean)
      : categoryBenchmarks.filter(b => b.id === imuDataset);
    renderImuControls(categoryBenchmarks);
    if (benchList.length === 0) { wrap.innerHTML = '<p style="padding:2rem;text-align:center;color:#6b7280">No benchmarks in this category.</p>'; return; }

    const methods = buildMethodRows(benchList, config).filter(m => imuView === 'all' || m.curation);
    if (imuSummary) imuSummary.textContent = `${methods.length} methods · ${benchList.length} datasets · ${imuView === 'curated' ? 'Curated selection' : 'Full method registry'}`;
    const sorted = sortMethods(methods, benchList, config);
    if (sorted.length === 0) { wrap.innerHTML = buildEmptyState(benchList, config); return; }

    wrap.innerHTML = buildTable(sorted, benchList, config);
    bindSortHeaders();
    bindExpand();
  }

  function syncBoardChrome() {
    if (boardSwitch) {
      boardSwitch.querySelectorAll('.board-option').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.board === activeBoard);
      });
    }
    const cross = activeBoard === 'imu' && activeTab === 'cross-dataset';
    if (crossControls) crossControls.hidden = !cross;
    if (imuControls) imuControls.hidden = activeBoard !== 'imu' || (cross && crossProtocol !== 'pairwise');
    if (imuGuide) imuGuide.hidden = cross || activeBoard !== 'imu';
    tabContainer?.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === activeTab));
    if (imuIntro) imuIntro.hidden = activeBoard !== 'imu';
    document.body.classList.toggle('imu-page', activeBoard === 'imu');
    wrap.classList.toggle('imu-board', activeBoard === 'imu');
    if (tabContainer) tabContainer.style.display = activeBoard === 'qa' ? 'none' : 'flex';
    if (qaTrackControls) qaTrackControls.hidden = activeBoard !== 'qa';
    if (wrap) wrap.classList.toggle('qa-board', activeBoard === 'qa');
    if (activeBoard === 'qa') renderQaTrackControls();
    if (note) {
      note.textContent = activeBoard === 'qa'
        ? 'Rows are models, agents, or explicit backbone configurations. Native methods have paper-reported results; adapted methods remain unranked until Track-specific evaluation is verified. Frameworks and datasets are metadata only.'
        : 'Paper = published table; Local = saved experiment. Protocols and metrics can differ; these are separate observations, not a shared ranking. Expand for file, table and review notes. — = unavailable.';
    }
  }

  function renderImuControls(categoryBenchmarks) {
    if (!imuDatasetSelect) return;
    imuDatasetSelect.innerHTML = '<option value="core">Common datasets</option>'
      + (activeTab === 'paper-coverage' ? '<option value="onehar-released">oneHAR released test sets · 5 datasets</option>' : '')
      + categoryBenchmarks.map(b => `<option value="${esc(b.id)}">${esc(b.name)}</option>`).join('');
    imuDatasetSelect.value = imuDataset;
    if (imuViewSelect) imuViewSelect.value = imuView;
  }

  function syncImuUrlState() {
    const url = new URL(window.location.href);
    if (activeTab !== 'paper-coverage') url.searchParams.set('tab', activeTab);
    else url.searchParams.delete('tab');
    if (activeTab === 'cross-dataset') {
      url.searchParams.set('protocol', crossProtocol);
      if (crossTarget !== 'all' && crossProtocol !== 'pairwise') url.searchParams.set('target', crossTarget);
      else url.searchParams.delete('target');
    } else { url.searchParams.delete('protocol'); url.searchParams.delete('target'); }
    if (imuView === 'all') url.searchParams.set('view', 'all');
    else url.searchParams.delete('view');
    if (imuDataset !== 'core') url.searchParams.set('dataset', imuDataset);
    else url.searchParams.delete('dataset');
    window.history.replaceState({}, '', url.toString());
  }

  if (imuDatasetSelect) imuDatasetSelect.addEventListener('change', () => {
    imuDataset = imuDatasetSelect.value;
    sortCol = 'avg';
    syncImuUrlState();
    render();
  });
  if (imuViewSelect) imuViewSelect.addEventListener('change', () => {
    imuView = imuViewSelect.value;
    sortCol = 'avg';
    syncImuUrlState();
    render();
  });

  crossProtocolSelect?.addEventListener('change', () => {
    crossProtocol = crossProtocolSelect.value;
    crossTarget = 'all';
    syncImuUrlState();
    render();
  });
  crossTargetSelect?.addEventListener('change', () => {
    crossTarget = crossTargetSelect.value;
    syncImuUrlState();
    render();
  });

  const CROSS_TARGET_SETS = {
    all: { label: 'Common targets · 5 columns', datasets: ['UCI-HAR', 'PAMAP2', 'RealWorld', 'USC-HAD', 'HHAR'] },
    mobility: { label: 'Phone & daily activity targets', datasets: ['MotionSense', 'w-HAR', 'Shoaib', 'HAR70+', 'TNDA-HAR'] },
    wearables: { label: 'Additional wearable targets', datasets: ['Opportunity', 'WISDM', 'Mhealth', 'Harth', 'DSADS'] },
    complex: { label: 'Complex activity targets', datasets: ['UT-Complex', 'Wharf', 'UTD-MHAD', 'MMAct'] }
  };
  const CROSS_CURATED_BASELINES = {
    'mosaic-zero-shot': ['deepconvlstm', 'if_convtransformer', 'imu2clip'],
    'onehar-lodo': ['deepconvlstm', 'unimts', 'hargpt']
  };

  function selectedTransferGroups() {
    return transferEvidence.groups.filter(g => crossProtocol === 'overview'
      ? g.category === 'zero_shot_transfer' : g.id === crossProtocol);
  }

  function renderCrossControls() {
    if (!crossProtocolSelect) return;
    const groups = transferEvidence.groups || [];
    crossProtocolSelect.innerHTML = (groups.length ? '<option value="overview">Zero-shot overview</option>' : '')
      + '<optgroup label="Paper comparison groups">' + groups.filter(g => g.category === 'zero_shot_transfer').map(g => `<option value="${esc(g.id)}">${esc(g.label)}</option>`).join('') + '</optgroup>'
      + '<optgroup label="Target-supervised references">' + groups.filter(g => g.category === 'target_supervised_reference').map(g => `<option value="${esc(g.id)}">${esc(g.label)}</option>`).join('') + '</optgroup>'
      + '<option value="pairwise">Existing pairwise registry</option>';
    crossProtocolSelect.value = crossProtocol;
    const selected = selectedTransferGroups();
    if (crossTargetLabel) crossTargetLabel.hidden = !selected.length;
    const methodLabel = document.getElementById('cross-methods-label');
    if (methodLabel) methodLabel.hidden = !selected.length;
    const methodSelect = document.getElementById('cross-methods');
    if (methodSelect) methodSelect.value = imuView;
    if (crossTargetSelect && selected.length) {
      const targets = [...new Map(selected.flatMap(g => g.results).map(r => [r.target_id, r])).values()];
      const sets = Object.entries(CROSS_TARGET_SETS).filter(([id, set]) => id === 'all' || targets.some(r => set.datasets.includes(r.target_dataset)));
      if (!sets.some(([id]) => id === crossTarget) && !targets.some(r => r.target_id === crossTarget)) crossTarget = 'all';
      crossTargetSelect.innerHTML = '<optgroup label="Dataset columns">' + sets.map(([id, set]) => `<option value="${id}">${esc(set.label)}</option>`).join('') + '</optgroup><optgroup label="Individual dataset variants">'
        + targets.map(r => `<option value="${esc(r.target_id)}">${esc(r.target_dataset)} · ${r.num_classes} classes</option>`).join('') + '</optgroup>';
      crossTargetSelect.value = crossTarget;
    }
  }

  document.getElementById('cross-methods')?.addEventListener('change', () => {
    imuView = document.getElementById('cross-methods').value;
    syncImuUrlState();
    render();
  });

  function transferPercent(value, group) {
    return (group.native_unit === 'fraction_0_1' ? value * 100 : value).toFixed(1) + '%';
  }

  function transferMetricPair(result, group) {
    return `<span class="score-main">${transferPercent(result.accuracy, group)}<span class="score-metric">Accuracy</span></span><span class="imu-cross-f1">${transferPercent(result.f1, group)} <span>${group.metric_names.f1 === 'macro F1' ? 'Macro F1' : 'F1*'}</span></span>`;
  }

  function transferDetails(group, entry) {
    const paper = transferEvidence.papers[group.paper_id];
    const p = group.provenance;
    const baseline = entry.role === 'paper_comparison_baseline';
    const locator = `Table ${p.table} · PDF p.${p.pdf_page_1based} · printed ${p.printed_page}`;
    let html = `<div class="imu-cross-details"><p><strong>Model:</strong> ${esc(entry.model_variant)}</p><p><strong>Source:</strong> ${esc(entry.source_description)}</p><p><strong>Uncertainty:</strong> ${esc(entry.uncertainty_note)}</p><p><strong>Metric:</strong> ${esc(group.metric_names.f1)}; native units: ${esc(group.native_unit)}. Displayed as percentages.</p>`;
    if (baseline) html += '<p>Comparison baseline evaluated/reimplemented in the cited paper. These are not independently verified scores from the baseline’s original paper. Training sources can differ across methods.</p>';
    else {
      const b = group.label_budget;
      const budget = b.samples_per_activity != null ? `${b.samples_per_activity} labeled target samples per activity`
        : b.train_fraction != null ? `${b.train_fraction * 100}% target training / ${b.test_fraction * 100}% test samples`
          : '0 target training labels · no target fine-tuning';
      html += `<p><strong>Target labels:</strong> ${esc(budget)}.</p><ul>${group.notes.map(n => `<li>${esc(n)}</li>`).join('')}</ul>`;
    }
    html += `<p><a href="${esc(paper.url)}" target="_blank" rel="noopener">${esc(paper.title)}</a> · ${esc(locator)}.</p><p><strong>Native paper average:</strong> accuracy ${entry.paper_reported_average.accuracy}; ${esc(group.metric_names.f1)} ${entry.paper_reported_average.f1} (${esc(group.native_unit)}).</p><p>${esc(entry.average_note)} Averages cover the complete paper table and are not recalculated for the visible columns.</p>`;
    if (!baseline && group.derived_unweighted_mean_not_paper_average) html += `<p><strong>Derived simple target mean (not the paper average):</strong> accuracy ${group.derived_unweighted_mean_not_paper_average.accuracy}; ${esc(group.metric_names.f1)} ${group.derived_unweighted_mean_not_paper_average.f1} (fractions).</p>`;
    if (!baseline && group.constraint_ablation) {
      const a = group.constraint_ablation;
      html += `<p><strong>Unknown-output constraint:</strong> Table ${a.table}, PDF p.${a.pdf_page_1based} / ${esc(a.printed_page)}. Main constrained averages: accuracy ${a.with_constraint.accuracy}%, F1 ${a.with_constraint.f1}%, unknown rate ${a.with_constraint.unknown_rate}%. Without constraint: accuracy ${a.without_constraint.accuracy}%, F1 ${a.without_constraint.f1}%, unknown rate ${a.without_constraint.unknown_rate}%. Unknown predictions count as incorrect; alternative per-target values are unavailable.</p>`;
    }
    html += `<div class="method-detail-grid">${entry.results.map(r => `<div class="method-detail-box"><strong>${esc(r.target_dataset)} · ${r.num_classes} classes</strong><p>${r.source_datasets.map(esc).join(' + ')} → ${esc(r.target_dataset)}</p><p>Accuracy ${transferPercent(r.accuracy, group)} · ${esc(group.metric_names.f1)} ${transferPercent(r.f1, group)}</p></div>`).join('')}</div>`;
    return html + `<p>Attachment: <code>${esc(paper.library_file_id)}</code> · version ${paper.library_file_version}. Supplied paper PDF; parent-verified table extraction. Paper observations only; no local reproduction or Standard promotion is asserted.</p><p><a href="data/paper_transfer_evidence.json" target="_blank" rel="noopener">Download full protocol evidence (JSON) ↗</a></p></div>`;
  }

  function renderTransferBoard() {
    const groups = selectedTransferGroups();
    if (!groups.length) return;
    renderCrossControls();
    const selectedTarget = groups.flatMap(g => g.results).find(r => r.target_id === crossTarget);
    const columns = CROSS_TARGET_SETS[crossTarget]?.datasets || [selectedTarget.target_dataset];
    const supervised = groups.every(g => g.category === 'target_supervised_reference');
    note.textContent = supervised
      ? 'Target-supervised reference. Target labels are used; kept separate from zero-shot transfer and Standard. Cells show Accuracy / F1; paper averages retain the original full-table values.'
      : 'Methods × target datasets. Protocol groups have different source/training settings; no shared ranking. Cells show Accuracy / F1. Class variants stay explicit. F1* = aggregation unspecified; — = not reported.';
    let html = '<table class="imu-table imu-cross-matrix"><caption class="sr-only">Cross-dataset method-by-target matrix. Separate paper protocol groups; no shared ranking.</caption><colgroup><col class="imu-method-col"><col class="imu-cross-average-col">' + columns.map(() => '<col>').join('') + '</colgroup><thead><tr><th scope="col">Method / source</th><th scope="col">Paper avg.<span class="imu-column-note">Full table · ACC / F1</span></th>'
      + columns.map(name => `<th scope="col">${esc(name)}<span class="imu-column-note">Accuracy / F1</span></th>`).join('') + '</tr></thead>';
    for (const group of groups) {
      const paper = transferEvidence.papers[group.paper_id];
      const p = group.provenance;
      const comparison = transferEvidence.comparisons?.[group.id] || [];
      const entries = [group, ...comparison.filter(e => imuView === 'all' || CROSS_CURATED_BASELINES[group.id]?.includes(e.method_id))];
      const label = group.id === 'mosaic-zero-shot' ? 'MOSAIC paper · synthetic → real for MOSAIC; baseline sources differ'
        : group.id === 'onehar-lodo' ? 'oneHAR paper · four source datasets → held-out target (LODO)'
          : group.label;
      html += `<tbody data-protocol="${esc(group.id)}"><tr class="imu-cross-group"><th scope="rowgroup" colspan="${columns.length + 2}">${esc(label)}<span>${esc(group.metric_names.f1)} · <a href="${esc(paper.url)}" target="_blank" rel="noopener">Table ${p.table}, PDF p.${p.pdf_page_1based} ↗</a></span></th></tr>`;
      for (const entry of entries) {
        const key = `${group.id}-${entry.method_id}`;
        const detailId = `imu-detail-${key}`;
        const baseline = entry.role === 'paper_comparison_baseline';
        html += `<tr class="imu-method-row" data-entry="${esc(key)}"><td class="method-name"><button class="expand-toggle" data-method="${esc(key)}" aria-controls="${detailId}" aria-expanded="false">${esc(entry.method_name)}<span aria-hidden="true" class="imu-expand-icon">+</span></button><span class="method-sub">${baseline ? 'Baseline evaluated in' : 'Primary method ·'} ${group.paper_id === 'mosaic' ? 'MOSAIC' : 'oneHAR'} · Table ${p.table}</span><a class="imu-paper-link" href="${esc(paper.url)}" target="_blank" rel="noopener">Paper ↗</a></td><td class="score imu-cross-average" data-label="Paper avg. · full table">${transferMetricPair(entry.paper_reported_average, group)}<span class="imu-column-note">${entry.results.length} targets · original avg.</span></td>`;
        for (const name of columns) {
          const r = entry.results.find(r => r.target_dataset === name && (!selectedTarget || r.target_id === crossTarget));
          html += `<td class="score" data-label="${esc(name)}"${r ? ` data-target="${esc(r.target_id)}"` : ''}>${r ? transferMetricPair(r, group) + `<span class="imu-column-note">${r.num_classes} classes</span>` : '<span class="score-na" aria-label="Not reported for this dataset variant">—</span>'}</td>`;
        }
        html += `</tr><tr class="expand-row" id="${detailId}" data-method="${esc(key)}"><td colspan="${columns.length + 2}">${transferDetails(group, entry)}</td></tr>`;
      }
      html += '</tbody>';
    }
    html += '</table><p class="imu-cross-footnote">F1* = averaging not specified in MOSAIC. oneHAR uses macro F1. Paper averages cover all original targets, not just visible columns. Expand a method for exact sources, target classes and complete results.</p>';
    wrap.innerHTML = html;
    bindExpand();
  }

  function renderQaTrackControls() {
    if (!qaTrackSwitch) return;
    const validTracks = new Set(qaTracks);
    if (activeQaTrack !== 'all' && !validTracks.has(activeQaTrack)) activeQaTrack = 'all';
    const options = [{ id: 'all', label: 'All Tracks' }, ...qaTracks.map(track => ({ id: track, label: track }))];
    qaTrackSwitch.innerHTML = options.map(option => {
      const active = option.id === activeQaTrack ? ' active' : '';
      const count = option.id === 'all' ? qaMethods.length : qaMethods.filter(method => method.track === option.id).length;
      return `<button class="qa-track-option${active}" data-track="${esc(option.id)}">${esc(option.label)}<span>${count}</span></button>`;
    }).join('');
    if (qaTrackSummary) {
      qaTrackSummary.textContent = activeQaTrack === 'all'
        ? `${qaTracks.length} tracks · ${qaMethods.length} methods`
        : `${qaMethods.filter(method => method.track === activeQaTrack).length} methods`;
    }
  }

  function tabConfig() {
    const configs = {
      'paper-coverage': {
        mode: 'paper',
        category: 'Standard',
        rankLabel: 'Coverage',
        emptyTitle: 'No paper coverage rows yet',
        emptyBody: 'No original-paper or official paper-protocol evidence has been curated for this dataset set yet.'
      },
      standard: {
        mode: 'standard',
        category: 'Standard',
        rankLabel: 'Avg',
        emptyTitle: 'No Standard leaderboard entries yet',
        emptyBody: 'Methods enter Standard only after at least two original-paper dataset calibrations pass and canonical official-code reruns exist under the unified IoTBench-HAR protocol.'
      },
      'cross-dataset': {
        mode: 'cross',
        category: 'Cross-Dataset',
        rankLabel: 'Avg',
        emptyTitle: 'No cross-dataset results yet',
        emptyBody: 'Cross-Dataset only displays methods with explicit source-target paper evidence or an official cross-dataset rerun.'
      }
    };
    return configs[activeTab] || configs['paper-coverage'];
  }

  function buildMethodRows(benchList, config) {
    const map = new Map();
    const benchmarkById = new Map(benchList.map(b => [b.id, b]));
    const allowedBenchIds = new Set(benchList.map(b => b.id));
    const calibrationCounts = paperCalibrationCountsByMethod();
    for (const r of results) {
      const key = r.method_name;
      if (!map.has(key)) {
        map.set(key, {
          id: r.id,
          curation: IMU_SELECTION[r.id],
          evidence: imuEvidence.records.filter(item => item.method_id === r.id),
          method_name: r.method_name,
          year: IMU_SELECTION[r.id]?.year || r.year || '',
          method_family: r.method_family || '',
          paper_url: r.paper_url || '',
          code_url: r.code_url || '',
          description: r.description || '',
          summary: r.summary || '',
          implementation: r.implementation || '',
          har_scope: r.har_scope || '',
          coverage_status: r.coverage_status || '',
          submission_type: r.submission_type || '',
          datasets: r.datasets || [],
          sources: r.sources || [],
          scores: {},
          score_references: {},
          calibration_count: 0,
          paper_validated: false
        });
      }
      const entry = map.get(key);
      if (r.description) entry.description = r.description;
      mergeDatasets(entry, r.datasets || []);
      mergeSources(entry, r.sources || []);
      entry.calibration_count = calibrationCounts.get(key) || 0;
      for (const item of (r.results || [])) {
        if (!allowedBenchIds.has(item.benchmark_id)) continue;
        const benchmark = benchmarkById.get(item.benchmark_id);
        if (isValidatedPaperRun(item)) entry.paper_validated = true;
        if (!entry.score_references[item.benchmark_id]) entry.score_references[item.benchmark_id] = [];
        if (isPaperReferenceResult(item)) entry.score_references[item.benchmark_id].push(item);
        if (!isDisplayableMainResult(item, benchmark, config.mode, entry.calibration_count)) continue;
        const current = entry.scores[item.benchmark_id];
        if (!current || preferResult(item, current, benchmark)) entry.scores[item.benchmark_id] = item;
      }
    }
    return Array.from(map.values()).filter(entry => {
      if (Object.keys(entry.scores).length > 0) return true;
      if (config.mode === 'paper' && entry.evidence.some(item => allowedBenchIds.has(item.benchmark_id))) return true;
      if (config.mode === 'cross') return entry.datasets.some(d => allowedBenchIds.has(d.benchmark_id));
      if (config.mode !== 'paper') return false;
      if (imuDataset === 'core' && entry.curation) return true;
      return entry.datasets.some(d => allowedBenchIds.has(d.benchmark_id));
    });
  }

  function paperCalibrationCountsByMethod() {
    const byMethod = new Map();
    for (const method of results) {
      if (!byMethod.has(method.method_name)) byMethod.set(method.method_name, new Set());
      const calibrated = byMethod.get(method.method_name);
      for (const result of (method.results || [])) {
        if (result?.official_code_used !== true && !isPaperReimplementation(result)) continue;
        if (result?.calibration_status !== 'passed') continue;
        if (result?.paper_reference_gap_status !== 'within_expected_range') continue;
        if (result?.paper_reference_score == null && !result?.paper_reference_metrics) continue;
        calibrated.add(result.benchmark_id);
      }
    }
    return new Map(Array.from(byMethod.entries()).map(([method, ids]) => [method, ids.size]));
  }

  function mergeDatasets(entry, datasets) {
    const seen = new Set(entry.datasets.map(d => `${d.benchmark_id || ''}|${d.dataset || ''}`));
    for (const d of datasets) {
      const key = `${d.benchmark_id || ''}|${d.dataset || ''}`;
      if (!seen.has(key)) {
        entry.datasets.push(d);
        seen.add(key);
      }
    }
  }

  function mergeSources(entry, sources) {
    const seen = new Set(entry.sources || []);
    for (const source of sources) {
      if (!seen.has(source)) {
        entry.sources.push(source);
        seen.add(source);
      }
    }
  }

  function preferResult(candidate, current, benchmark) {
    const candidateScore = scoreValue(candidate, benchmark);
    const currentScore = scoreValue(current, benchmark);
    if ((candidateScore == null) !== (currentScore == null)) return candidateScore != null;
    const candidateUsable = isUsableResult(candidate);
    const currentUsable = isUsableResult(current);
    if (candidateUsable !== currentUsable) return candidateUsable;
    const candidateFresh = ['official_code_standard_raw_v2','paper_reimplementation_standard_raw_v2'].includes(candidate.source);
    const currentFresh = ['official_code_standard_raw_v2','paper_reimplementation_standard_raw_v2'].includes(current.source);
    if (candidateFresh !== currentFresh && candidate.protocol_status === 'canonical' && current.protocol_status === 'canonical') return candidateFresh;
    const candidateValidatedPaperRun = isValidatedPaperRun(candidate);
    const currentValidatedPaperRun = isValidatedPaperRun(current);
    if (candidateValidatedPaperRun !== currentValidatedPaperRun) return candidateValidatedPaperRun;
    const candidatePreferredOfficial = isPreferredOfficialResult(candidate);
    const currentPreferredOfficial = isPreferredOfficialResult(current);
    if (candidatePreferredOfficial !== currentPreferredOfficial) return candidatePreferredOfficial;
    const candidatePaperProtocol = candidate.protocol_status === 'paper_protocol';
    const currentPaperProtocol = current.protocol_status === 'paper_protocol';
    if (candidatePaperProtocol !== currentPaperProtocol) return candidatePaperProtocol;
    const candidateCanonical = candidate.protocol_status === 'canonical';
    const currentCanonical = current.protocol_status === 'canonical';
    if (candidateCanonical !== currentCanonical) return candidateCanonical;
    const candidateRank = resultRank(candidate);
    const currentRank = resultRank(current);
    if (candidateRank !== currentRank) return candidateRank > currentRank;
    return true;
  }

  function resultRank(result) {
    let rank = 0;
    if (result.protocol_status === 'canonical') rank += 40;
    if (result.protocol_status === 'paper_protocol') rank += 45;
    if (result.protocol_status === 'official_extension') rank += 35;
    if (result.official_code_used === false) rank += 1;
    if (result.official_code_used === true) rank += 20;
    if (isOfficialPaperProtocolResult(result)) rank += 40;
    if (result.source === 'official_code_standard') rank += 38;
    if (result.source === 'official_code_extension') rank += 35;
    if (result.source === 'official_code_adapter') rank += 20;
    if (!isUsableResult(result)) rank -= 100;
    return rank;
  }

  function isUsableResult(result) {
    const status = [
      result?.calibration_status,
      result?.paper_reference_gap_status,
      result?.reproduction_status
    ].filter(Boolean).join(' ');
    return !/(failed|blocked|superseded|large_gap|paper_reference_not_curated|not_in_original_paper|not_directly_comparable|not_standard_main|unverified)/i.test(status);
  }

  function isPreferredOfficialResult(result) {
    if (!isUsableResult(result) || result?.official_code_used !== true) return false;
    if (result.protocol_status === 'paper_protocol' && isGapReviewedRerun(result)) return true;
    return result.source === 'official_code_adapter' ||
      isOfficialPaperProtocolResult(result) ||
      result.source === 'official_code_extension' ||
      result.source === 'official_code_standard' || result.source === 'official_code_standard_raw_v2';
  }

  function isOfficialPaperProtocolResult(result) {
    return result?.source === 'official_code_paper_protocol' ||
      result?.source === 'official_code_paper_protocol_rerun';
  }

  function isGapReviewedRerun(result) {
    if (!isUsableResult(result)) return false;
    if (result?.paper_reference_gap_status !== 'within_expected_range') return false;
    return result?.paper_reference_score != null || result?.paper_reference_metrics;
  }

  function isValidatedPaperRun(result) {
    if (!isGapReviewedRerun(result)) return false;
    return result?.official_code_used === true || isPaperReimplementation(result);
  }

  function isPaperReimplementation(result) {
    if (result?.paper_reimplementation === true) return true;
    const source = String(result?.source || '').toLowerCase();
    return source.includes('reimplementation') || source.includes('rerun') || source.includes('training');
  }

  function isPromotablePaperRerun(result, calibrationCount) {
    if (!isGapReviewedRerun(result)) return false;
    return (calibrationCount || 0) >= 2;
  }

  function isOriginalPaperValue(result) {
    if (!result || result.official_code_used === true) return false;
    if (result.runtime_seconds != null || result.epochs != null || result.folds != null) return false;
    const source = String(result.source || '').toLowerCase();
    if (source.includes('local') || source.includes('rerun') || source.includes('training') || source.includes('reimplementation')) return false;
    return source.includes('table') ||
      source.includes('figure') ||
      source.includes('paper') ||
      result.submission_type === 'paper_reported';
  }

  function isDisplayableMainResult(result, benchmark, mode, calibrationCount) {
    if (scoreValue(result, benchmark) == null || !isUsableResult(result)) return false;
    if (mode === 'paper') {
      if (result.protocol_status !== 'paper_protocol') return false;
      if (isOriginalPaperValue(result)) return true;
      if (isPaperReimplementation(result)) return isPromotablePaperRerun(result, calibrationCount);
      return isPreferredOfficialResult(result) && isPromotablePaperRerun(result, calibrationCount);
    }
    if (mode === 'cross') {
      if (isOriginalPaperValue(result)) return true;
      if (isPaperReimplementation(result)) return isPromotablePaperRerun(result, calibrationCount);
      return isPreferredOfficialResult(result) && (calibrationCount || 0) >= 2;
    }
    if (mode === 'standard') {
      if ((calibrationCount || 0) < 2) return false;
      const auditedPaper = result.source === 'paper_reimplementation_standard_raw_v2' && result.paper_reimplementation === true;
      return (isPreferredOfficialResult(result) || auditedPaper) && result.protocol_status === 'canonical';
    }
    return false;
  }

  function isPaperReferenceResult(result) {
    const source = String(result?.source || '').toLowerCase();
    if (source.includes('paper_inspired') || source.includes('local_inspired')) return false;
    return isOriginalPaperValue(result) ||
      result?.paper_reference_score != null ||
      result?.paper_reference_metrics;
  }

  function sortMethods(methods, benchList, config) {
    return methods.sort((a, b) => {
      if (config.mode === 'standard' && sortCol !== 'avg') {
        const bench = benchList.find(item => item.id === sortCol);
        const va = scoreValue(a.scores[sortCol], bench) ?? -1;
        const vb = scoreValue(b.scores[sortCol], bench) ?? -1;
        if (va !== vb) return sortDir === 'desc' ? vb - va : va - vb;
      }
      return (Number(b.year) || 0) - (Number(a.year) || 0) || a.method_name.localeCompare(b.method_name);
    });
  }

  function coverageCount(method, benchList, mode) {
    if (activeTab === 'paper-coverage' && method.curation && imuEvidenceLoaded) {
      const ids = new Set(benchList.map(b => b.id));
      return new Set(method.evidence.filter(item => item.display !== false && ids.has(item.benchmark_id)).map(item => item.benchmark_id)).size;
    }
    let count = 0;
    for (const b of benchList) {
      const fresh = mode === 'paper' && reproduction.cells.some(c => c.method === method.id && c.dataset === b.id && c.status === 'complete');
      const evidence = mode === 'paper' && method.evidence.some(r => r.benchmark_id === b.id && r.display !== false && Number.isFinite(r.primary_score));
      if (scoreValue(method.scores[b.id], b) != null || fresh || evidence) count++;
    }
    return count;
  }

  // ── Build table HTML ──────────────────────────────────
  function buildTable(methods, benchList, config) {
    const benchIds = benchList.map(b => b.id);
    let html = '<table class="imu-table"><caption class="sr-only">IMU activity recognition evidence by method and dataset. Publication order; no aggregate performance ranking.</caption><colgroup><col class="imu-method-col"><col class="imu-coverage-col">' + benchList.map(() => '<col>').join('') + '</colgroup><thead><tr><th scope="col">Method / publication</th><th scope="col">Coverage</th>';
    for (const b of benchList) {
      const label = datasetLabel(b);
      const sortable = config.mode === 'standard';
      const arrow = sortCol === b.id ? (sortDir === 'desc' ? ' ↓' : ' ↑') : '';
      const download = datasetDownloadKey(b.id);
      html += `<th scope="col"><a href="task.html?id=${esc(b.id)}" title="${esc(b.name)}" class="task-link">${esc(label)}</a><span class="imu-column-note">${config.mode === 'paper' ? 'Reported metric' : esc(metricLabel(b.metrics.primary))}</span>${download ? `<a href="datasets.html#${download}" class="dataset-download-link" aria-label="Download ${esc(label)} data and view its license">Download data ↓</a>` : ''}${sortable ? `<button class="sort-header imu-sort" data-col="${esc(b.id)}" aria-label="Sort by ${esc(label)}">Sort${arrow}</button>` : ''}</th>`;
    }
    html += '</tr></thead><tbody>';
    for (const m of methods) {
      const detailId = `imu-detail-${m.id}`;
      html += '<tr class="imu-method-row">';
      const subtitle = m.curation ? `${m.curation.venue} · ${m.year}` : [m.year, m.method_family].filter(Boolean).join(' · ');
      html += `<td class="method-name"><button class="expand-toggle" data-method="${esc(m.method_name)}" aria-controls="${esc(detailId)}" aria-expanded="false">${esc(m.method_name)}<span aria-hidden="true" class="imu-expand-icon">+</span></button>${subtitle ? `<span class="method-sub">${esc(subtitle)}</span>` : ''}${m.curation?.score_note ? `<span class="imu-result-status">${esc(m.curation.score_note)}</span><a class="imu-paper-link" href="${esc(m.curation.url)}" target="_blank" rel="noopener">Paper ↗</a>` : ''}${m.curation?.released_tests ? '<a class="imu-paper-link" href="?board=imu&amp;dataset=onehar-released">View 5 released test sets →</a>' : ''}${m.curation?.transfer_protocol ? `<a class="imu-paper-link" href="?board=imu&amp;tab=cross-dataset&amp;protocol=${esc(m.curation.transfer_protocol)}">View protocol results →</a>` : ''}</td>`;
      html += `<td class="imu-coverage" data-label="Coverage">${coverageCount(m, benchList, config.mode)}<span> / ${benchList.length}</span></td>`;
      for (const b of benchList) {
        const s = m.scores[b.id];
        const audited = m.evidence.filter(item => item.benchmark_id === b.id && item.display !== false);
        const value = config.mode === 'paper' && m.curation && imuEvidenceLoaded
          ? buildEvidenceCell(audited, m.id, b)
          : s ? scoreCell(scoreValue(s, b), s.protocol_status, s, m.score_references[b.id] || [], b) : '<span class="score-na" aria-label="No eligible result">—</span>';
        html += `<td class="score" data-label="${esc(datasetLabel(b))}"><div>${value}</div></td>`;
      }
      html += '</tr>';

      // expandable detail row
      html += `<tr class="expand-row" id="${esc(detailId)}" data-method="${esc(m.method_name)}"><td colspan="${2 + benchList.length}" style="white-space:normal">`;
      if (m.curation) html += `<p class="imu-selection-reason"><strong>Why selected:</strong> ${esc(m.curation.reason)} <a href="${esc(m.curation.url)}" target="_blank" rel="noopener">Publication source ↗</a></p>`;
      if (m.evidence.length && config.mode === 'paper') html += buildEvidenceDetails(m, benchIds);
      const separateLegacy = config.mode === 'paper' && m.curation && imuEvidenceLoaded;
      if (separateLegacy) html += '<details class="imu-legacy"><summary>Original registry details (historical)</summary>';
      const overview = m.summary || m.description;
      if (overview) html += `<div style="line-height:1.6;margin-bottom:.75rem;color:var(--text)">${esc(overview)}</div>`;
      if (m.har_scope) html += `<div style="line-height:1.6;margin-bottom:.75rem"><span class="detail-label">HAR scope:</span> ${esc(m.har_scope)}</div>`;
      if (m.implementation) html += `<div style="line-height:1.6;margin-bottom:.75rem"><span class="detail-label">Implementation:</span> ${esc(m.implementation)}</div>`;
      html += `<div style="line-height:1.6;margin-bottom:.75rem"><span class="detail-label">Paper calibration:</span> ${esc(m.calibration_count || 0)} dataset${(m.calibration_count || 0) === 1 ? '' : 's'} passed</div>`;
      html += '<div class="method-detail-grid">';
      const datasetDetails = m.datasets.filter(d => benchIds.includes(d.benchmark_id));
      for (const d of datasetDetails) {
        html += '<div class="method-detail-box">';
        html += `<div class="detail-label">${esc(d.dataset || d.benchmark_id)}</div>`;
        if (d.activity_task) html += `<div><strong>Task:</strong> ${esc(d.activity_task)}</div>`;
        if (d.subjects) html += `<div><strong>Subjects:</strong> ${esc(d.subjects)}</div>`;
        if (d.device_placement) html += `<div><strong>Placement:</strong> ${esc(d.device_placement)}</div>`;
        if (d.reported_effect) html += `<div><strong>Effect:</strong> ${esc(d.reported_effect)}</div>`;
        if (d.protocol_note) html += `<div><strong>Protocol:</strong> ${esc(d.protocol_note)}</div>`;
        html += '</div>';
      }
      html += '</div>';
      html += '<div style="display:flex;gap:2rem;flex-wrap:wrap;margin:.75rem 0 .5rem">';
      for (const b of benchList) {
        const s = m.scores[b.id];
        if (!s || !s.metrics) continue;
        html += '<div>';
        html += `<div class="detail-label">${b.name}${s.protocol_status ? ` <span class="protocol-badge">${statusLabel(s.protocol_status)}</span>` : ''}</div>`;
        for (const [mk, mv] of Object.entries(s.metrics)) {
          html += `<div>${mk}: ${typeof mv === 'number' ? (mv * 100).toFixed(1) : mv}</div>`;
        }
        const runBits = [];
        if (typeof s.runtime_seconds === 'number') runBits.push(`runtime ${s.runtime_seconds.toFixed(1)}s`);
        if (s.epochs != null) runBits.push(`${esc(s.epochs)} epochs`);
        if (s.seed != null) runBits.push(`seed ${esc(s.seed)}`);
        if (s.folds != null) runBits.push(`${esc(s.folds)} folds`);
        if (s.device) runBits.push(`device ${esc(s.device)}`);
        if (s.gpu_name) runBits.push(esc(s.gpu_name));
        if (s.official_code_used === true) runBits.push('official code');
        if (s.official_code_used === false) runBits.push('official code not used');
        if (s.source) runBits.push(`source ${esc(s.source)}`);
        if (s.paper_training_setting) runBits.push(`paper setting ${esc(s.paper_training_setting)}`);
        if (s.official_repo) runBits.push(`<a href="${esc(s.official_repo)}" target="_blank" rel="noopener">official repo</a>`);
        if (s.official_commit) runBits.push(`commit ${esc(String(s.official_commit).slice(0, 8))}`);
        if (s.implementation_variant) runBits.push(esc(s.implementation_variant));
        if (s.feature_selector) runBits.push(esc(s.feature_selector));
        if (s.selected_features != null) runBits.push(`${esc(s.selected_features)} selected features`);
        if (s.optimizer) runBits.push(esc(s.optimizer));
        if (s.alpha != null) runBits.push(`alpha ${esc(s.alpha)}`);
        if (s.max_iter != null) runBits.push(`max_iter ${esc(s.max_iter)}`);
        if (runBits.length) html += `<div class="run-meta">${runBits.join(' · ')}</div>`;
        if (s.protocol_note) html += `<div class="run-meta"><strong>Protocol:</strong> ${esc(s.protocol_note)}</div>`;
        if (s.review_note) html += `<div class="run-meta"><strong>Review:</strong> ${esc(s.review_note)}</div>`;
        const references = paperReferencesFor(s, m.score_references[b.id] || [], b);
        if (references.length) {
          html += '<div class="reference-list"><span class="detail-label">Original paper result:</span>';
          for (const ref of references) {
            const refBits = paperReferenceBits(ref, b);
            html += `<div class="reference-item">${refBits.join(' · ')}</div>`;
          }
          html += '</div>';
        }
        html += '</div>';
      }
      html += '</div>';
      if (m.submission_type) {
        const label = submissionLabel(m.submission_type);
        const color = m.submission_type === 'platform' ? '#059669' : '#9ca3af';
        html += `<div style="margin-top:.35rem"><span style="font-size:.78rem;color:${color}">${label}</span></div>`;
      }
      if (m.coverage_status) html += `<div style="margin-top:.15rem"><span class="detail-label">Coverage:</span> ${esc(m.coverage_status)}</div>`;
      if (m.paper_url) html += `<div style="margin-top:.15rem"><span class="detail-label">Paper:</span> <a href="${esc(m.paper_url)}" target="_blank" rel="noopener">${esc(m.paper_url)}</a></div>`;
      if (m.code_url) html += `<div><span class="detail-label">Code:</span> <a href="${esc(m.code_url)}" target="_blank" rel="noopener">${esc(m.code_url)}</a></div>`;
      if (m.sources.length) html += `<div><span class="detail-label">Sources:</span> ${m.sources.map(u => `<a href="${esc(u)}" target="_blank" rel="noopener">${esc(sourceLabel(u))}</a>`).join(' · ')}</div>`;
      if (separateLegacy) html += '</details>';
      html += '</td></tr>';
    }
    html += '</tbody></table>';
    return html;
  }

  function evidenceMetric(metric) {
    return metric === 'f1_score' ? 'F1 (type n/a)' : metricLabel(metric);
  }

  function evidenceStatus(item) {
    const labels = { released_checkpoint_evaluated: 'Measured · released test set', registry_only: 'Registry only', registry_only_review: 'Registry · review', metric_mismatch_registry: 'Different metric · registry', protocol_mismatch_registry: 'Different protocol · registry', protocol_review: 'Review needed', metric_review: 'Review needed', inspired_baseline: 'Inspired baseline' };
    return labels[item.status] || '';
  }

  function buildEvidenceCell(items, methodId, benchmark) {
    const run = reproduction.cells.find(c => c.method === methodId && c.dataset === benchmark.id);
    let fresh = '';
    if (run) {
      const href = `reproduction.html#${encodeURIComponent(methodId)}`;
      if (run.status === 'complete' && Number.isFinite(run.metrics?.[benchmark.metrics.primary])) {
        const metric = benchmark.metrics.primary;
        fresh = `<div class="imu-evidence-item imu-evidence-local"><span class="imu-evidence-kind">Fresh rerun · 5 seeds</span><strong>${(run.metrics[metric] * 100).toFixed(2)}%<small>± ${(run.metrics[metric + '_std'] * 100).toFixed(2)} · ${esc(metricLabel(metric))}</small></strong><span class="imu-evidence-review">${run.required_folds} runs complete · ${run.validation_status === 'standard_eligible' ? 'Standard eligible' : 'calibration pending'}</span><a class="imu-paper-link" href="${href}">Run details →</a></div>`;
      } else {
        const state = { running: 'Training', queued: 'Queued', failed: 'Run failed · review', source_unavailable: 'Code not located', adapter_audit_pending: 'Adapter audit', paper_fulltext_needed: 'Paper full text needed', training_assets_missing: 'Training assets needed' }[run.status] || 'Audit pending';
        fresh = `<div class="imu-evidence-item"><a class="imu-paper-link" href="${href}">${esc(state)} · ${run.completed_folds}/${run.required_folds} runs →</a></div>`;
      }
    }
    if (!items.length) return '<span class="score-na" aria-label="No verified paper table or saved run">—</span>' + fresh;
    return ['paper', 'local'].map(kind => {
      const item = items.find(record => record.kind === kind);
      const label = kind === 'paper' ? 'Paper' : 'Local';
      if (!item) return `<div class="imu-evidence-item imu-evidence-missing"><span class="imu-evidence-kind">${label}</span>—</div>`;
      const protocol = imuEvidence.protocols[item.protocol_id] || {};
      const status = evidenceStatus(item);
      return `<div class="imu-evidence-item imu-evidence-${kind}"><span class="imu-evidence-kind">${label}</span><strong>${(item.primary_score * 100).toFixed(2)}%<small>${esc(evidenceMetric(item.primary_metric))}</small></strong><span class="imu-evidence-setting">${esc(protocol.label || '')}</span>${status ? `<span class="imu-evidence-review">${esc(status)}</span>` : ''}</div>`;
    }).join('') + fresh;
  }

  function buildEvidenceDetails(method, benchIds) {
    const records = method.evidence.filter(item => benchIds.includes(item.benchmark_id));
    if (!records.length) return '';
    return `<section class="imu-audit"><h3>Paper tables &amp; saved experiments</h3><p>Historical observations only. These entries do not bypass the Standard calibration gate.</p>${records.map(item => {
      const p = imuEvidence.protocols[item.protocol_id] || {};
      const source = imuEvidence.sources[item.source_id];
      const file = item.run_source;
      const metrics = Object.entries(item.metrics).filter(([key, value]) => typeof value === 'number' && !key.endsWith('_std')).map(([key, value]) => `${esc(evidenceMetric(key))}: ${item.metric_units?.[key] === 'loss' ? value.toFixed(4) : (value * 100).toFixed(2) + '%'}`).join(' · ');
      const bits = [
        `<h4>${esc(item.benchmark_id)} · ${item.kind === 'paper' ? 'Paper-reported' : 'Saved local result'} · ${esc(p.label || '')}</h4>`,
        `<p>${metrics}${item.reported_uncertainty != null ? ` · reported ± ${(item.reported_uncertainty * 100).toFixed(2)} points` : ''}</p>`,
        `<p><strong>Dataset:</strong> ${esc(p.dataset_variant || 'See source')}</p>`,
        ...(item.model_variant ? [`<p><strong>Model variant:</strong> ${esc(item.model_variant)}</p>`] : []),
        `<p><strong>Split:</strong> ${esc(p.split || 'Not established')}</p><p>${esc(p.metric_note || '')}</p>`
      ];
      if (source) bits.push(`<p><a href="${esc(source.url)}" target="_blank" rel="noopener">${esc(source.title)}</a> · ${esc(source.locator)}</p>`);
      if (file) bits.push(`<p><strong>Saved file:</strong> <code>${esc(file.path)}#${esc(file.json_pointer)}</code></p><p class="imu-audit-hash">SHA-256: ${esc(file.sha256)}</p><p>${item.standalone_artifact_found ? 'Standalone saved result found.' : 'Saved registry record only; standalone run artifact not located.'} ${esc(evidenceStatus(item))}</p><p>Saved calibration: ${esc(item.saved_calibration_status || 'not recorded')}; gap review: ${esc(item.saved_gap_status || 'not recorded')}; seed: ${esc(item.seed ?? 'not recorded')}; epochs: ${esc(item.epochs ?? 'not recorded')}.</p>`);
      if (file && item.saved_protocol_status) bits.push(`<p><strong>Saved protocol:</strong> ${esc(item.saved_protocol_status)}. ${esc(item.saved_protocol_note || '')}</p>`);
      if (file && item.configuration_source) bits.push(`<p><strong>Configuration:</strong> ${esc(item.configuration_source)}</p>`);
      return `<article class="imu-audit-record">${bits.join('')}</article>`;
    }).join('')}</section>`;
  }

  function buildEmptyState(benchList, config) {
    const names = benchList.map(b => b.name).join(', ');
    return `<div style="padding:3rem 2rem;text-align:center;color:#6b7280">
      <h3 style="margin-bottom:.5rem">${esc(config.emptyTitle)}</h3>
      <p>${esc(config.emptyBody)}</p>
      <p style="margin-top:.65rem">Benchmarks: ${esc(names)}.</p>
    </div>`;
  }

  // ── Sensor QA board ───────────────────────────────────
  function renderQaBoard() {
    if (!qaBenchmarks.length) {
      wrap.innerHTML = '<p style="padding:2rem;text-align:center;color:#6b7280">No Sensor QA benchmarks configured.</p>';
      return;
    }
    const trackMethods = activeQaTrack === 'all'
      ? qaMethods
      : qaMethods.filter(method => method.track === activeQaTrack);
    const modalities = [...new Set(trackMethods.map(method => method.modality).filter(Boolean))].sort();
    if (activeQaModality !== 'all' && !modalities.includes(activeQaModality)) activeQaModality = 'all';
    const rows = trackMethods
      .filter(method => activeQaModality === 'all' || method.modality === activeQaModality)
      .map(method => ({ ...method }));
    visibleQaBenchmarks = qaBenchmarks.filter(benchmark => rows.some(method => qaScoreValue(method.scores?.[benchmark.id]) != null));
    if (sortCol !== 'avg' && sortCol !== 'track_order' && !visibleQaBenchmarks.some(b => b.id === sortCol)) sortCol = 'avg';
    const sorted = sortQaRows(rows);
    wrap.innerHTML = buildQaTable(sorted, modalities, trackMethods.length);
    bindQaBenchmarkGuide();
    bindQaFilters();
    bindSortHeaders();
    bindExpand();
  }

  function sortQaRows(rows) {
    return rows.sort((a, b) => {
      if (sortCol === 'track_order') return (a.track_order ?? Number.MAX_SAFE_INTEGER) - (b.track_order ?? Number.MAX_SAFE_INTEGER);
      let va, vb;
      if (sortCol === 'avg') {
        va = qaCoverageCount(a);
        vb = qaCoverageCount(b);
        if (va === vb) {
          va = qaAvgScore(a);
          vb = qaAvgScore(b);
        }
      } else {
        va = qaSortableValue(a.scores?.[sortCol]);
        vb = qaSortableValue(b.scores?.[sortCol]);
      }
      if (va === vb) return 0;
      const diff = sortDir === 'desc' ? vb - va : va - vb;
      return diff > 0 ? 1 : -1;
    });
  }

  function qaAvgScore(method) {
    let sum = 0, count = 0;
    for (const b of visibleQaBenchmarks) {
      const val = qaScoreValue(method.scores?.[b.id]);
      if (typeof val === 'number') { sum += val; count++; }
    }
    return count ? sum / count : -1;
  }

  function qaCoverageCount(method) {
    let count = 0;
    for (const b of visibleQaBenchmarks) {
      if (qaScoreValue(method.scores?.[b.id]) != null) count++;
    }
    return count;
  }

  function buildQaTable(methods, modalities, trackMethodCount) {
    let html = '<div class="qa-table-toolbar">';
    html += '<label class="qa-modality-filter"><span>Modality</span><select id="qa-modality-filter">';
    html += `<option value="all"${activeQaModality === 'all' ? ' selected' : ''}>All modalities</option>`;
    for (const modality of modalities) {
      const selected = modality === activeQaModality ? ' selected' : '';
      html += `<option value="${esc(modality)}"${selected}>${esc(modality)}</option>`;
    }
    html += '</select></label>';
    html += `<span class="qa-result-count">Showing ${methods.length} of ${trackMethodCount} methods · ${visibleQaBenchmarks.length} reported benchmarks</span>`;
    html += '</div>';
    html += buildQaBenchmarkGuide(visibleQaBenchmarks);

    if (!methods.length) {
      html += '<div class="qa-empty-state"><strong>No methods match this modality.</strong><span>Choose another modality or track.</span></div>';
      return html;
    }

    html += '<div class="qa-table-scroll"><table><thead><tr><th>#</th><th>Method</th><th>Type</th><th>Framework</th><th>Source</th><th>Scenario</th><th>Task</th><th>Modality</th><th>Track</th><th>Evaluation Benchmark</th>';
    const avgArrow = sortCol === 'avg' ? (sortDir === 'desc' ? ' ↓' : ' ↑') : '';
    html += `<th class="sort-header" data-col="avg">Coverage${avgArrow}</th>`;
    for (const b of visibleQaBenchmarks) {
      const arrow = sortCol === b.id ? (sortDir === 'desc' ? ' ↓' : ' ↑') : '';
      html += `<th class="sort-header" data-col="${esc(b.id)}">${esc(b.name)}${arrow}<br><span style="font-weight:400;font-size:.7rem;color:#9ca3af">${metricLabel(b.primary_metric)}</span></th>`;
    }
    html += '</tr></thead><tbody>';

    let rank = 0;
    for (const m of methods) {
      rank++;
      const hasScore = qaCoverageCount(m) > 0;
      html += '<tr>';
      html += `<td class="rank">${hasScore ? rank : '—'}</td>`;
      const subtitle = [m.year, m.venue].filter(Boolean).join(' · ');
      const rowId = m.id || m.method_name;
      const methodMeta = [m.entry_type || m.method_type || m.backbone, subtitle].filter(Boolean).join(' · ');
      html += `<td class="method-name"><a class="expand-toggle" data-method="${esc(rowId)}">${esc(m.method_name)}</a>${methodMeta ? `<br><span class="method-sub">${esc(methodMeta)}</span>` : ''}</td>`;
      const origin = m.origin || 'Native';
      html += '<td><div class="qa-type-tags">';
      html += `<span class="qa-origin-chip qa-origin-${String(origin).toLowerCase()}">${esc(origin)}</span>`;
      if (m.is_baseline && origin !== 'Baseline') html += '<span class="qa-baseline-marker">Baseline</span>';
      if (m.baseline_tier) html += `<span class="qa-baseline-tier">${esc(m.baseline_tier)}</span>`;
      html += '</div></td>';
      html += `<td>${m.framework ? `<span class="qa-framework-chip">${esc(m.framework)}</span>` : '<span class="score-na">—</span>'}</td>`;
      html += `<td class="qa-source">${esc(m.source_label || [m.venue, m.year].filter(Boolean).join(' ') || '—')}</td>`;
      html += `<td class="qa-scenario">${esc(m.scenario || '—')}</td>`;
      html += `<td>${esc(m.task || '—')}</td>`;
      html += `<td><span class="qa-modality-chip">${esc(m.modality || '—')}</span></td>`;
      html += `<td><span class="qa-track-chip">${esc(m.track || '—')}</span></td>`;
      html += `<td class="qa-native-benchmark">${esc(m.evaluation_benchmark || '—')}</td>`;
      const coverage = qaCoverageCount(m);
      html += `<td class="score">${coverage ? `${coverage}/${visibleQaBenchmarks.length}` : `<span class="score-badge score-badge-placeholder">${esc(m.status || 'placeholder')}</span>`}</td>`;
      for (const b of visibleQaBenchmarks) {
        const result = m.scores?.[b.id];
        html += `<td class="${result ? 'score' : 'score-na'}">${qaScoreCell(result, b)}</td>`;
      }
      html += '</tr>';

      html += `<tr class="expand-row" data-method="${esc(rowId)}"><td colspan="${11 + visibleQaBenchmarks.length}" style="white-space:normal">`;
      html += buildQaBaselineExplanation(m);
      html += buildQaMethodNarrative(m);
      html += '<div class="method-detail-grid">';
      html += '<div class="method-detail-box">';
      html += `<div><span class="detail-label">Entity:</span> ${esc(m.entry_type || m.method_type || 'Method')}</div>`;
      html += `<div><span class="detail-label">Type:</span> ${esc(m.origin || 'Native')}</div>`;
      if (m.is_baseline) html += `<div><span class="detail-label">Baseline level:</span> ${esc(m.baseline_tier || 'Baseline')}</div>`;
      if (m.baseline_role) html += `<div><span class="detail-label">Baseline role:</span> ${esc(m.baseline_role)}</div>`;
      html += `<div><span class="detail-label">Framework:</span> ${esc(m.framework || '—')}</div>`;
      html += `<div><span class="detail-label">Source:</span> ${esc(m.source_label || [m.venue, m.year].filter(Boolean).join(' ') || '—')}</div>`;
      html += `<div><span class="detail-label">Scenario:</span> ${esc(m.scenario || '—')}</div>`;
      html += `<div><span class="detail-label">Classification:</span> ${esc(m.track || '—')} · ${esc(m.task || '—')}</div>`;
      html += `<div><span class="detail-label">Modality:</span> ${esc(m.modality || '—')}</div>`;
      html += `<div><span class="detail-label">Native benchmark:</span> ${esc(m.native_benchmark || '—')}</div>`;
      html += `<div><span class="detail-label">Evaluation benchmark:</span> ${esc(m.evaluation_benchmark || '—')}</div>`;
      if (m.core_source_method && m.core_source_method !== m.method_name) {
        html += `<div><span class="detail-label">Source row:</span> ${esc(m.core_source_method)}</div>`;
      }
      html += `<div><span class="detail-label">Training data:</span> ${esc(m.training_data || '—')}</div>`;
      html += `<div><span class="detail-label">Status:</span> ${esc(m.status || 'pending')}</div>`;
      if (m.paper_url) html += `<div><span class="detail-label">Paper:</span> <a href="${esc(m.paper_url)}" target="_blank" rel="noopener">${esc(m.paper_url)}</a></div>`;
      if (m.code_url) html += `<div><span class="detail-label">Code:</span> <a href="${esc(m.code_url)}" target="_blank" rel="noopener">${esc(m.code_url)}</a></div>`;
      html += '</div>';
      for (const d of (m.datasets || [])) {
        html += '<div class="method-detail-box">';
        html += `<div class="detail-label">${esc(d.dataset || d.benchmark_id)}</div>`;
        if (d.protocol_note) html += `<div><strong>Protocol:</strong> ${esc(d.protocol_note)}</div>`;
        html += '</div>';
      }
      html += '</div>';
      html += '<div style="display:flex;gap:2rem;flex-wrap:wrap;margin:.75rem 0 .5rem">';
      for (const b of visibleQaBenchmarks) {
        const result = m.scores?.[b.id];
        if (!result) continue;
        html += '<div>';
        html += `<div class="detail-label">${esc(b.name)}</div>`;
        for (const [mk, mv] of Object.entries(result.raw_metrics || result.metrics || {})) {
          html += `<div>${esc(mk)}: ${esc(mv)}</div>`;
        }
        if (result.scale) html += `<div class="run-meta"><strong>Scale:</strong> ${esc(result.scale)}</div>`;
        if (result.protocol_note) html += `<div class="run-meta"><strong>Protocol:</strong> ${esc(result.protocol_note)}</div>`;
        if (result.review_note) html += `<div class="run-meta"><strong>Review:</strong> ${esc(result.review_note)}</div>`;
        if (Array.isArray(result.variants) && result.variants.length > 1) {
          html += '<div class="reference-list"><span class="detail-label">All reported settings:</span>';
          for (const v of result.variants) {
            const metric = esc(v.primary_metric_label || v.primary_metric || 'score');
            const value = esc(v.display_value ?? '—');
            const bits = [v.split, v.task_type, v.setting, v.source_table].filter(Boolean).map(esc);
            html += `<div class="reference-item">${metric} ${value}${v.scale ? ` (${esc(v.scale)})` : ''}${bits.length ? ` · ${bits.join(' · ')}` : ''}</div>`;
          }
          html += '</div>';
        }
        html += '</div>';
      }
      html += '</div>';
      html += '</td></tr>';
    }
    html += '</tbody></table></div>';
    return html;
  }

  function buildQaBenchmarkGuide(benchmarks) {
    const documented = benchmarks.filter(b =>
      b.data_summary && b.task_definition && b.evaluation_protocol && b.metrics_description
    );
    if (!documented.length) return '';
    let html = '<section class="qa-benchmark-guide">';
    html += `<div class="qa-benchmark-heading"><strong>Benchmark definitions</strong><span>${documented.length} benchmarks · 点击按钮查看数据 / 任务定义 / 评测协议 / 指标</span></div>`;
    html += '<div class="qa-benchmark-buttons" role="tablist" aria-label="Benchmark definitions">';
    for (const b of documented) {
      html += `<button type="button" class="qa-benchmark-button" role="tab" data-benchmark-info="${esc(b.id)}" aria-controls="benchmark-info-${esc(b.id)}" aria-expanded="false">${esc(b.name)}</button>`;
    }
    html += '</div><div class="qa-benchmark-panels">';
    for (const b of documented) {
      html += `<article class="qa-benchmark-card" id="benchmark-info-${esc(b.id)}" data-benchmark-panel="${esc(b.id)}" hidden>`;
      html += `<h3>${esc(b.name)}</h3>`;
      html += `<div><strong>数据</strong><p>${esc(b.data_summary)}</p></div>`;
      html += `<div><strong>任务定义</strong><p>${esc(b.task_definition)}</p></div>`;
      html += `<div><strong>评测协议</strong><p>${esc(b.evaluation_protocol)}</p></div>`;
      html += `<div><strong>指标</strong><p>${esc(b.metrics_description)}</p></div>`;
      if (b.source_url) html += `<a class="qa-benchmark-source" href="${esc(b.source_url)}" target="_blank" rel="noopener">Source paper ↗</a>`;
      html += '</article>';
    }
    html += '</div></section>';
    return html;
  }

  function bindQaBenchmarkGuide() {
    const buttons = [...document.querySelectorAll('.qa-benchmark-button')];
    const panels = [...document.querySelectorAll('.qa-benchmark-card[data-benchmark-panel]')];
    for (const button of buttons) {
      button.addEventListener('click', () => {
        const benchmarkId = button.dataset.benchmarkInfo;
        const wasOpen = button.getAttribute('aria-expanded') === 'true';
        for (const item of buttons) {
          item.classList.remove('active');
          item.setAttribute('aria-expanded', 'false');
        }
        for (const panel of panels) panel.hidden = true;
        if (!wasOpen) {
          button.classList.add('active');
          button.setAttribute('aria-expanded', 'true');
          const panel = panels.find(item => item.dataset.benchmarkPanel === benchmarkId);
          if (panel) panel.hidden = false;
        }
      });
    }
  }

  function bindQaFilters() {
    const modalityFilter = document.getElementById('qa-modality-filter');
    if (!modalityFilter) return;
    modalityFilter.addEventListener('change', () => {
      activeQaModality = modalityFilter.value || 'all';
      syncQaUrlState();
      renderQaBoard();
    });
  }

  function buildQaBaselineExplanation(method) {
    if (!method.is_baseline) return '';
    return `<div class="qa-baseline-explanation">
      <span class="qa-baseline-explanation-label">Baseline 分类</span>
      <strong>${esc(method.baseline_tier || 'Baseline')}</strong>
      <span>${esc(method.baseline_role || '用于统一评测协议下的对照。')}</span>
    </div>`;
  }

  function buildQaMethodNarrative(method) {
    const detail = method.details || {};
    const problem = detail.problem || method.summary || 'Paper-level description is still being curated.';
    const approach = detail.approach || method.implementation || 'Implementation details are pending manual paper extraction.';
    const effect = detail.effect || summarizeQaResults(method);
    return `<div class="method-story-grid">
      <div class="method-story-box">
        <h4>解决了什么</h4>
        <p>${esc(problem)}</p>
      </div>
      <div class="method-story-box">
        <h4>怎么做的</h4>
        <p>${esc(approach)}</p>
      </div>
      <div class="method-story-box">
        <h4>效果怎么样</h4>
        <p>${esc(effect)}</p>
      </div>
    </div>`;
  }

  function summarizeQaResults(method) {
    const reported = [];
    for (const b of visibleQaBenchmarks) {
      const result = method.scores?.[b.id];
      if (!result) continue;
      const label = result.primary_metric_label || metricLabel(result.primary_metric || b.primary_metric || 'score');
      reported.push(`${b.name}: ${label} ${result.display_value ?? '—'}${result.scale ? ` (${result.scale})` : ''}`);
      if (reported.length >= 3) break;
    }
    if (reported.length) return `论文报告了 ${reported.join('；')}。`;
    return '还没有抽取到稳定可核对的主结果表；当前仅保留论文占位，等待人工核表或作者公开结果。';
  }

  function qaScoreCell(result, benchmark) {
    if (!result) return '—';
    const val = qaScoreValue(result);
    if (val == null) return '—';
    const metric = metricLabel(result.primary_metric || benchmark.primary_metric || 'score');
    const display = result.display_value != null ? esc(result.display_value) : (val * 100).toFixed(1);
    const scale = result.scale ? `<span class="score-alt">${esc(result.scale)}</span>` : '';
    const variants = Array.isArray(result.variants) && result.variants.length > 1
      ? `<span class="score-badge score-badge-paper" title="${result.variants.length} reported settings">+${result.variants.length - 1}</span>`
      : '';
    const badge = result.verified === true ? '<span class="score-badge score-badge-official">verified</span>' : '';
    return `<span class="score-main"><span class="score-metric">${metric}</span> ${display}</span>${badge}${variants}${scale}`;
  }

  function qaScoreValue(result) {
    if (!result) return null;
    const primary = result.primary_metric;
    if (primary && result.metrics && typeof result.metrics[primary] === 'number') return result.metrics[primary];
    if (typeof result.primary_score === 'number') return result.primary_score;
    return null;
  }

  function qaSortableValue(result) {
    const value = qaScoreValue(result);
    if (value == null) return -1;
    return result.direction === 'lower' ? -value : value;
  }

  function preferQaResult(candidate, current) {
    const candidateValue = qaScoreValue(candidate);
    const currentValue = qaScoreValue(current);
    if (candidateValue == null) return false;
    if (currentValue == null) return true;
    const lowerIsBetter = candidate.direction === 'lower' || current.direction === 'lower';
    return lowerIsBetter ? candidateValue < currentValue : candidateValue > currentValue;
  }

  // ── Column sort ────────────────────────────────────────
  function bindSortHeaders() {
    // prevent task-link clicks from triggering column sort
    wrap.querySelectorAll('.task-link').forEach(a => {
      a.addEventListener('click', (e) => { e.stopPropagation(); });
    });
    wrap.querySelectorAll('.sort-header').forEach(th => {
      th.style.cursor = 'pointer';
      th.addEventListener('click', () => {
        const col = th.dataset.col;
        if (sortCol === col) { sortDir = sortDir === 'desc' ? 'asc' : 'desc'; }
        else { sortCol = col; sortDir = 'desc'; }
        render();
      });
    });
  }

  // ── Row expansion ──────────────────────────────────────
  function bindExpand() {
    wrap.querySelectorAll('.expand-toggle').forEach(a => {
      a.addEventListener('click', (e) => {
        e.preventDefault();
        const method = a.dataset.method;
        const row = wrap.querySelector(`.expand-row[data-method="${method}"]`);
        if (row) {
          const open = row.classList.toggle('show');
          a.setAttribute('aria-expanded', String(open));
        }
      });
    });
  }

  // ── Sensor QA track switching ─────────────────────────
  if (qaTrackSwitch) {
    qaTrackSwitch.addEventListener('click', (e) => {
      const btn = e.target.closest('.qa-track-option');
      if (!btn) return;
      activeQaTrack = btn.dataset.track || 'all';
      activeQaModality = 'all';
      sortCol = activeQaTrack === 'all' ? 'avg' : 'track_order';
      sortDir = 'desc';
      syncQaUrlState();
      render();
    });
  }

  function syncQaUrlState() {
    const url = new URL(window.location.href);
    if (activeBoard === 'qa') url.searchParams.set('board', 'qa');
    else url.searchParams.delete('board');
    if (activeBoard === 'qa' && activeQaTrack !== 'all') url.searchParams.set('track', activeQaTrack);
    else url.searchParams.delete('track');
    if (activeBoard === 'qa' && activeQaModality !== 'all') url.searchParams.set('modality', activeQaModality);
    else url.searchParams.delete('modality');
    window.history.replaceState({}, '', url.toString());
  }

  // ── Board switching ────────────────────────────────────
  if (boardSwitch) {
    boardSwitch.addEventListener('click', (e) => {
      const btn = e.target.closest('.board-option');
      if (!btn) return;
      activeBoard = btn.dataset.board || 'imu';
      sortCol = activeBoard === 'qa' && activeQaTrack !== 'all' ? 'track_order' : 'avg';
      sortDir = 'desc';
      syncQaUrlState();
      render();
    });
  }

  // ── Tab switching ──────────────────────────────────────
  tabContainer.addEventListener('click', (e) => {
    const tab = e.target.closest('.tab');
    if (!tab) return;
    activeTab = tab.dataset.tab;
    sortCol = 'avg';
    sortDir = 'desc';
    tabContainer.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    syncImuUrlState();
    render();
  });

  // ── Helpers ────────────────────────────────────────────
  function datasetLabel(b) { return ({ uci_har: 'UCI-HAR', wisdm: 'WISDM', pamap2: 'PAMAP2', hhar: 'HHAR', opportunity_a: 'OPPORTUNITY' })[b.id] || b.name.replace('OPPORTUNITY — ', 'OPP · ').replace('Cross-Dataset: ', ''); }
  function datasetDownloadKey(id) { return ({ uci_har: 'uci_har', wisdm: 'wisdm', pamap2: 'pamap2', hhar: 'hhar', opportunity_a: 'opportunity', opportunity_b: 'opportunity' })[id] || ''; }
  function scoreCell(val, status, result, references, benchmark) {
    if (val == null) return '—';
    const note = result?.official_code_used === true && status && status !== 'canonical' ? `<span class="score-note" title="${esc(statusLabel(status))}">*</span>` : '';
    const badge = resultBadge(result);
    const protocolBadge = lowLabelProtocol(result) ? '<span class="score-badge score-badge-lowlabel" title="Reproduced under LIMU-BERT 1% labeled-data protocol">1% labeled</span>' : '';
    const ref = isGapReviewedRerun(result) ? bestPaperReference(result, references, benchmark) : null;
    const refHtml = ref ? paperReferenceLine(ref, benchmark) : '';
    const setting = lowLabelProtocol(result) ? '1% labeled data' : result?.paper_training_setting || result?.protocol_note || result?.source || '';
    const metricPrefix = `<span class="score-metric">${metricLabel(displayMetric(result, benchmark))}</span> `;
    const fresh = activeBoard === 'imu' && ['official_code_standard_raw_v2','paper_reimplementation_standard_raw_v2'].includes(result?.source);
    const std = result?.metrics?.[displayMetric(result, benchmark) + '_std'];
    const uncertainty = fresh && Number.isFinite(std) ? `<span class="score-alt">± ${(std * 100).toFixed(2)} · 5 seeds</span>` : '';
    return `<span class="score-main" title="${esc(setting)}">${metricPrefix}${(val * 100).toFixed(fresh ? 2 : 1)}${note}</span>${uncertainty}${badge}${protocolBadge}${activeBoard === 'qa' ? refHtml : ''}`;
  }

  function resultBadge(result) {
    if (result?.official_code_used === true) return '<span class="score-badge score-badge-official">official</span>';
    if (result?.source === 'paper_reimplementation_standard_raw_v2') return '<span class="score-badge score-badge-repro">paper reproduction</span>';
    if (isGapReviewedRerun(result)) return '<span class="score-badge score-badge-repro">reproduced</span>';
    if (isOriginalPaperValue(result)) return '<span class="score-badge score-badge-paper">paper</span>';
    return '';
  }
  function lowLabelProtocol(result) {
    const text = [
      result?.source,
      result?.protocol_note,
      result?.paper_training_setting,
      result?.official_pipeline
    ].filter(Boolean).join(' ').toLowerCase();
    return text.includes('limu') && (text.includes('1%') || text.includes('lowlabel') || text.includes('low-label'));
  }
  function bestPaperReference(selected, references, benchmark) {
    const refs = paperReferencesFor(selected, references, benchmark);
    return refs[0] || null;
  }
  function paperReferencesFor(selected, references, benchmark) {
    const refs = [];
    if (selected?.paper_reference_metrics || selected?.paper_reference_score != null) refs.push({ paper_reference: selected });
    for (const ref of (references || [])) {
      if (ref === selected && selected?.official_code_used === true) continue;
      if (scoreValue(ref, benchmark) == null) continue;
      refs.push(ref);
    }
    return refs.sort((a, b) => paperReferenceRank(b) - paperReferenceRank(a)).slice(0, 3);
  }
  function paperReferenceRank(result) {
    const ref = result.paper_reference || result;
    let rank = 0;
    if (ref.paper_reference_metrics || ref.paper_reference_score != null) rank += 100;
    if (ref.protocol_status === 'paper_protocol') rank += 70;
    if (String(ref.source || '').toLowerCase().includes('table')) rank += 20;
    return rank;
  }
  function paperReferenceLine(result, benchmark) {
    const val = paperReferenceValue(result, benchmark);
    if (typeof val !== 'number') return '';
    const title = paperReferenceBits(result, benchmark).join(' · ');
    const metric = metricLabel(paperReferenceMetric(result.paper_reference || result, benchmark));
    return `<span class="score-alt" title="${esc(title)}">${metric} ${(val * 100).toFixed(1)} paper ref</span>`;
  }
  function paperReferenceBits(result, benchmark) {
    const val = paperReferenceValue(result, benchmark);
    const ref = result.paper_reference || result;
    const bits = [];
    if (typeof val === 'number') bits.push(`${metricLabel(paperReferenceMetric(ref, benchmark))} ${(val * 100).toFixed(1)}`);
    bits.push('original paper');
    if (ref.paper_reference_source) bits.push(esc(ref.paper_reference_source));
    else if (ref.source) bits.push(esc(ref.source));
    if (ref.paper_reference_gap_abs != null) bits.push(`gap ${(ref.paper_reference_gap_abs * 100).toFixed(1)} pts`);
    if (ref.paper_reference_protocol_note) bits.push(esc(ref.paper_reference_protocol_note));
    return bits;
  }
  function paperReferenceMetric(result, benchmark) {
    return result.paper_reference_metric || displayMetric(result, benchmark);
  }
  function paperReferenceValue(result, benchmark) {
    const ref = result.paper_reference || result;
    const metric = paperReferenceMetric(ref, benchmark);
    if (ref.paper_reference_metrics && typeof ref.paper_reference_metrics[metric] === 'number') return ref.paper_reference_metrics[metric];
    if (typeof ref.paper_reference_score === 'number') return ref.paper_reference_score;
    return scoreValue(ref, benchmark);
  }
  function scoreValue(result, benchmark) {
    if (!result) return null;
    const metrics = result.metrics || null;
    const resultPrimary = result.primary_metric;
    const resultMetricValue = resultPrimary && metrics && metrics[resultPrimary];
    if (typeof resultMetricValue === 'number') return resultMetricValue;
    if (typeof result.primary_score === 'number') return result.primary_score;
    const primary = benchmark?.metrics?.primary;
    const metricValue = primary && metrics && metrics[primary];
    if (typeof metricValue === 'number') return metricValue;
    if (primary && metrics && result.protocol_status !== 'paper_protocol') return null;
    return null;
  }
  function displayMetric(result, benchmark) {
    if (result?.primary_metric) return result.primary_metric;
    if (result?.protocol_status === 'paper_protocol') {
      const referenceMetric = result.paper_reference_metric;
      if (referenceMetric && typeof result.metrics?.[referenceMetric] === 'number' &&
          Math.abs(result.metrics[referenceMetric] - result.primary_score) < 1e-9) return referenceMetric;
      const matching = Object.entries(result.metrics || {}).filter(([key, value]) =>
        !key.endsWith('_std') && typeof value === 'number' && Math.abs(value - result.primary_score) < 1e-9);
      return matching.length === 1 ? matching[0][0] : 'score';
    }
    return benchmark?.metrics?.primary || 'score';
  }
  function metricLabel(metric) {
    const labels = {
      accuracy: 'Accuracy',
      f1_macro: 'Macro F1',
      f1_macro_no_null: 'Macro F1 (10-class)',
      f1_macro_all_classes: 'Macro F1 (all)',
      f1_weighted: 'Weighted F1',
      f1_weighted_all_classes: 'Weighted F1 (all)',
      accuracy_all_classes: 'Accuracy (all)',
      f1_score: 'F-score',
      exact_match: 'Exact Match',
      short_exact_match: 'Short EM',
      rouge_l: 'ROUGE-L',
      bleu: 'BLEU',
      cider: 'CIDEr',
      meteor: 'METEOR',
      sensible_answer_rate: 'Sensible Answer Rate',
      factual_correctness: 'Factual Correctness',
      zero_shot_avg_accuracy: '11-dataset Macro Accuracy'
    };
    return labels[metric] || metric;
  }
  function statusLabel(status) {
    const labels = {
      canonical: 'Canonical leaderboard protocol',
      paper_protocol: 'Original-paper protocol; used for Paper Coverage calibration',
      official_extension: 'Official code extension after paper calibration',
      dataset_variant: 'Different dataset/protocol; hidden from the main Standard leaderboard',
      implementation_variant: 'Implementation mismatch; hidden from the main Standard leaderboard'
    };
    return labels[status] || status;
  }
  function submissionLabel(type) {
    const labels = {
      platform: 'Platform verified',
      self_reported: 'Self reported',
      paper_reported: 'Paper reported',
      method_registry: 'Method registry only',
      local_checkpoint_evaluation: 'Local released-checkpoint evaluation'
    };
    return labels[type] || type;
  }
  function sourceLabel(url) {
    try { return new URL(url).hostname.replace(/^www\./, ''); }
    catch { return url; }
  }
  function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

  // ── Go ─────────────────────────────────────────────────
  load();
})();
