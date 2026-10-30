(function (root) {
  'use strict';
  const CORE = ['mosaic', 'onehar', 'robusthar', 'unimts', 'if_convtransformer', 'tinyhar', 'limu_bert', 'attend_and_discriminate', 'deepsense', 'deepconvlstm'];
  const META = {
    mosaic: ['MOSAIC', 2026, 'IMWUT / UbiComp'], onehar: ['oneHAR', 2025, 'IMWUT / UbiComp'],
    robusthar: ['RobustHAR', 2025, 'IJCAI'], unimts: ['UniMTS', 2024, 'NeurIPS'],
    if_convtransformer: ['IF-ConvTransformer', 2022, 'IMWUT'], tinyhar: ['TinyHAR', 2022, 'ISWC'],
    limu_bert: ['LIMU-BERT', 2021, 'SenSys'], attend_and_discriminate: ['Attend & Discriminate', 2021, 'IMWUT'],
    deepsense: ['DeepSense', 2017, 'WWW'], deepconvlstm: ['DeepConvLSTM', 2016, 'Sensors']
  };
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  const metricName = key => ({ accuracy: 'Accuracy', f1_macro: 'Macro F1', f1_weighted: 'Weighted F1', f1_score: 'F1 (unspecified)', f1: 'F1', completeness: 'Completeness' }[key] || String(key || 'Score').replaceAll('_', ' '));
  function methodIndex(registry) {
    const map = new Map();
    for (const m of registry) {
      const meta = META[m.id];
      if (!map.has(m.id)) map.set(m.id, { id: m.id, name: meta?.[0] || m.method_name, year: meta?.[1] || m.year || '', venue: meta?.[2] || m.venue || '', paper: m.paper_url || '', code: m.code_url || '' });
    }
    return map;
  }
  function metricValue(record, metric) {
    return finite(record.metrics?.[metric]) ? record.metrics[metric] : record.primary_metric === metric && finite(record.primary_score) ? record.primary_score : null;
  }
  function eligibleLocal(record) {
    const status = [record.status, record.saved_calibration_status, record.saved_gap_status].filter(Boolean).join(' ');
    return !/(failed|blocked|superseded|review|mismatch|inspired|unverified)/i.test(status);
  }
  function choosePaperScore(paper, locals) {
    // Compare the latest available matching observation, never the best test score.
    const matching = locals.filter(r => r.method_id === paper.method_id && r.benchmark_id === paper.benchmark_id && r.protocol_id === paper.protocol_id && metricValue(r, paper.primary_metric) !== null && eligibleLocal(r))
      .sort((a, b) => String(b.checked_at || b.created_at || '').localeCompare(String(a.checked_at || a.created_at || '')) || Number(b.standalone_artifact_found === true) - Number(a.standalone_artifact_found === true) || String(a.id).localeCompare(String(b.id)));
    const measured = matching[0];
    const score = measured ? metricValue(measured, paper.primary_metric) : null;
    const gap = measured ? Math.abs(score - paper.primary_score) : null;
    // A published score always stays primary. Runs and gaps are comparison evidence.
    return { selected: paper, source: 'paper', value: paper.primary_score, gap, paper, measured: measured || null, alternatives: locals.filter(r => r.method_id === paper.method_id && r.benchmark_id === paper.benchmark_id) };
  }
  function fullRun(cell) {
    if (cell.status !== 'complete' || !Number.isInteger(cell.required_folds) || cell.required_folds < 1 || cell.completed_folds !== cell.required_folds || cell.fold_results?.length !== cell.required_folds) return false;
    const keys = new Set(cell.fold_results.map(r => `${r.seed}:${r.fold}`));
    const seeds = cell.seeds || [];
    const expected = [42, 123, 2021, 2024, 3407];
    const oldModels = cell.source?.superseded_model_sha256 || [];
    return keys.size === cell.required_folds && seeds.length === 5 && expected.every(s => seeds.includes(s) && cell.fold_results.filter(r => r.seed === s).length === cell.required_folds / 5) && !cell.fold_results.some(r => oldModels.includes(r.model_sha256)) && ['accuracy', 'f1_macro', 'f1_weighted'].every(k => finite(cell.metrics?.[k]) && cell.metrics[k] >= 0 && cell.metrics[k] <= 1);
  }
  function forMetric(row, metric) {
    if (row.choice) {
      const paperValue = metricValue(row.choice.paper, metric);
      if (paperValue === null) return null;
      const paper = { ...row.choice.paper, primary_metric: metric, primary_score: paperValue };
      const choice = choosePaperScore(paper, row.choice.alternatives);
      return { ...row, metric, metricLabel: metricName(metric), value: choice.value, metrics: choice.selected.metrics || {}, source: choice.source, choice, sourceRecord: choice.selected, uncertainty: metric === row.metric ? row.uncertainty : choice.selected.metrics?.[`${metric}_std`] };
    }
    const value = metricValue({ metrics: row.metrics, primary_metric: row.metric, primary_score: row.value }, metric);
    return value === null ? null : { ...row, metric, metricLabel: metricName(metric), value, uncertainty: row.metrics?.[`${metric}_std`] };
  }
  function imuObservations(registry, evidence, reproduction) {
    const methods = methodIndex(registry);
    const out = [];
    const locals = (evidence.records || []).filter(r => r.kind === 'local');
    for (const paper of (evidence.records || []).filter(r => r.kind === 'paper' && r.display !== false && finite(r.primary_score))) {
      const choice = choosePaperScore(paper, locals);
      const source = evidence.sources?.[paper.source_id] || {};
      out.push({ id: paper.id, method: methods.get(paper.method_id) || { id: paper.method_id, name: paper.method_id }, dataset: paper.benchmark_id === 'opportunity_a' && ['tiny_loso', 'ad_opp', 'if_tsf_loso', 'deepconv_original_gesture'].includes(paper.protocol_id) ? 'opportunity_gesture_18c' : paper.benchmark_id, protocol: paper.protocol_id, protocolLabel: evidence.protocols?.[paper.protocol_id]?.label || paper.protocol_id, protocolNote: evidence.protocols?.[paper.protocol_id]?.split || '', metric: paper.primary_metric, metricLabel: metricName(paper.primary_metric), value: choice.value, metrics: { ...(choice.selected.metrics || {}), [paper.primary_metric]: choice.value }, source: choice.source, direction: 'higher', unit: 'percent', choice, paperUrl: source.url || methods.get(paper.method_id)?.paper || '', locator: source.locator || source.table || '', sourceRecord: choice.selected, paperRole: paper.paper_role, sourceTitle: source.title, review: paper.paper_role === 'comparison' ? 'Comparison result in the cited paper; not independently verified from the model’s original paper.' : '', datasetLabel: paper.dataset_label || (paper.benchmark_id === 'opportunity_a' && ['tiny_loso', 'ad_opp', 'if_tsf_loso'].includes(paper.protocol_id) ? 'OPPORTUNITY · 17 gestures + Null' : ''), uncertainty: paper.reported_uncertainty });
    }
    // Released-checkpoint inference has its own protocol; it is never paired with LODO.
    for (const record of locals.filter(r => r.status === 'released_checkpoint_evaluated' && r.display !== false && finite(r.primary_score))) {
      out.push({ id: record.id, method: methods.get(record.method_id) || { id: record.method_id, name: record.method_id }, dataset: ({ pamap2: 'pamap2_9c', usc_had: 'usc_had_7c' }[record.benchmark_id] || record.benchmark_id), datasetLabel: ({ pamap2: 'PAMAP2 · 9 classes', usc_had: 'USC-HAD · 7 classes' }[record.benchmark_id]), protocol: record.protocol_id, protocolLabel: evidence.protocols?.[record.protocol_id]?.label || record.protocol_id, protocolNote: evidence.protocols?.[record.protocol_id]?.split || '', metric: record.primary_metric, metricLabel: metricName(record.primary_metric), value: record.primary_score, metrics: record.metrics || {}, source: 'run', direction: 'higher', unit: 'percent', sourceRecord: record, paperUrl: methods.get(record.method_id)?.paper || '' });
    }
    for (const cell of reproduction.cells || []) {
      if (!fullRun(cell)) continue;
      const metric = cell.dataset === 'opportunity_a' ? 'f1_weighted' : 'accuracy';
      const opportunity = cell.dataset === 'opportunity_a';
      const protocolNote = opportunity
        ? 'Custom subject-held-out evaluation: ADL only; 113 channels; 30-sample (1 s) windows, stride 15. Each fold uses 2 training subjects, 1 validation subject and 1 test subject. Four test-subject folds × five seeds. Checkpoints were selected by validation Macro-F1; the displayed primary score is Weighted-F1. No Null output class: target voting excludes Null labels, but input windows retain Null frames. Mean over subjects within each seed, then mean and population std over five seeds; ± does not measure subject variation. This is not a paper-protocol reproduction.'
        : 'Official UCI split; fixed subject split for WISDM; all subject folds for PAMAP2 and HHAR. Complete five-seed measurements; paper-calibration status is retained per method.';
      out.push({ id: `raw-v2:${cell.method}:${cell.dataset}`, method: methods.get(cell.method) || { id: cell.method, name: cell.method }, dataset: cell.dataset, protocol: 'raw-v2', protocolLabel: 'Unified raw-v2 · 5 seeds', protocolNote, metric, metricLabel: metricName(metric), value: cell.metrics[metric], metrics: cell.metrics, uncertainty: cell.metrics[`${metric}_std`], source: 'run', direction: 'higher', unit: 'percent', review: opportunity ? 'Custom protocol; paper-protocol reproduction not established. ' + (cell.validation_status === 'standard_eligible' ? 'Historical model calibration does not validate this OPPORTUNITY protocol.' : 'Implementation calibration pending.') : cell.validation_status === 'standard_eligible' ? 'Standard eligible' : 'Calibration pending', auditUrl: opportunity ? 'data/opportunity_audit_20261030.json' : '', sourceRecord: cell, paperUrl: methods.get(cell.method)?.paper || '' });
    }
    // Keep additional publication records accessible. An audited overlay wins over
    // older registry imports for the same method and dataset.
    const covered = new Set(out.filter(r => r.choice).map(r => `${r.method.id}:${r.choice.paper.benchmark_id}`));
    const seen = new Set();
    for (const method of registry) for (const r of method.results || []) {
      const src = String(r.source || '');
      if (covered.has(`${method.id}:${r.benchmark_id}`) || seen.has(`${method.id}:${r.benchmark_id}:${src}`)) continue;
      if (!/paper.*(table|figure|section)|(table|figure).*paper/.test(src) || /(inspired|rerun|reimplementation|training|superseded|unverified)/.test(src) || r.official_code_used === true || r.runtime_seconds != null) continue;
      const metric = r.primary_metric || Object.keys(r.metrics || {})[0];
      const value = metricValue({ ...r, primary_metric: metric }, metric);
      if (!finite(value)) continue;
      seen.add(`${method.id}:${r.benchmark_id}:${src}`);
      out.push({ id: `registry:${method.id}:${r.benchmark_id}:${src}`, method: methods.get(method.id), dataset: r.benchmark_id, protocol: `paper:${method.id}:${src}`, protocolLabel: `${methods.get(method.id)?.name} · paper setting`, protocolNote: r.protocol_note || '', metric, metricLabel: metricName(metric), value, metrics: r.metrics || {}, source: 'paper', direction: 'higher', unit: 'percent', paperUrl: method.paper_url || '', sourceRecord: r });
    }
    return out;
  }
  function qaObservations(data) {
    const ids = new Set(data.leaderboard_method_ids || []);
    const out = [];
    for (const m of [...(data.methods || []), ...(data.adapted_methods || []), ...(data.supplemental_methods || [])]) {
      if (ids.size && !ids.has(m.id)) continue;
      const taxonomy = data.method_taxonomy?.[m.id] || data.method_taxonomy?.[m.method_name] || {};
      const method = { id: m.id, name: taxonomy.display_name || m.method_name, year: m.year || '', venue: m.venue || m.paper_short || '', paper: m.paper_url || '', code: m.code_url || '', modality: taxonomy.modality || m.backbone || '', track: taxonomy.track || '' };
      const scores = Object.values(m.scores || {});
      const unique = new Map();
      for (const r of [...(m.results || []), ...scores]) if (finite(r.primary_score) && !unique.has(r.benchmark_id)) unique.set(r.benchmark_id, r);
      for (const r of unique.values()) {
        const variants = r.variants || [];
        const v = variants[0] || {};
        const native = String(r.scale || '');
        const isPercent = native.startsWith('%') || /%/.test(native) && /acc|accuracy|em|uar|original_test/i.test(r.primary_metric);
        out.push({ id: `qa:${m.id}:${r.benchmark_id}`, method, dataset: r.benchmark_id, protocol: `qa:${r.benchmark_id}:${v.split || 'reported'}:${v.setting || 'reported'}`, protocolLabel: [v.split, v.setting].filter(Boolean).join(' · ') || 'Published test setting', protocolNote: r.protocol_note || '', metric: r.primary_metric, metricLabel: r.primary_metric_label || metricName(r.primary_metric), value: r.primary_score, displayValue: r.display_value, metrics: r.metrics || {}, source: 'paper', direction: r.direction || 'higher', unit: native.includes('% point') ? 'percentage-points' : isPercent ? 'percent' : 'native', paperUrl: r.paper_url || m.paper_url || '', locator: v.source_table || '', review: r.verified === false ? 'Paper import · source locator retained' : '', sourceRecord: r });
      }
    }
    return out;
  }
  function transferObservations(data) {
    const out = [];
    for (const group of data.groups || []) {
      const paper = data.papers?.[group.paper_id] || {};
      for (const entry of [group, ...(data.comparisons?.[group.id] || [])]) {
        for (const r of entry.results || []) {
          const factor = group.native_unit === 'fraction_0_1' ? 1 : 0.01;
          const macro = /macro/i.test(group.metric_names?.f1 || '');
          const metrics = { accuracy: r.accuracy * factor, [macro ? 'f1_macro' : 'f1_score']: r.f1 * factor };
          out.push({ id: `transfer:${group.id}:${entry.method_id}:${r.target_id}`, method: { id: entry.method_id, name: entry.method_id === group.method_id ? META[group.method_id]?.[0] || entry.method_name : entry.method_name, year: paper.year || '', venue: entry.method_id === group.method_id ? 'IMWUT / UbiComp' : `${group.method_name} comparison`, paper: paper.url || '' }, dataset: r.target_id, datasetLabel: `${r.target_dataset} · ${r.num_classes} classes`, protocol: group.id, protocolLabel: group.label, protocolNote: `${entry.role === 'paper_comparison_baseline' ? entry.source_description || group.setting : group.setting + '. ' + group.source_description} ${entry.role === 'paper_comparison_baseline' ? 'Comparison baseline evaluated in the cited paper; training sources can differ.' : ''}`, metric: 'accuracy', metricLabel: 'Accuracy', metrics, value: metrics.accuracy, source: 'paper', direction: 'higher', unit: 'percent', paperUrl: paper.url || '', locator: `Table ${group.provenance?.table} · PDF p.${group.provenance?.pdf_page_1based}`, sourceRecord: { ...r, model_variant: entry.model_variant, source_description: entry.source_description, original_paper_average: entry.paper_reported_average, source_table: group.provenance } });
        }
      }
    }
    return out;
  }
  // Only class-compatible variants share a recurring dataset column. All other
  // label sets keep their own identifiers in rankings and aggregate denominators.
  const mainDataset = id => ({ uci_har_6c: 'uci_har', hhar_6c: 'hhar', realworld_8c: 'realworld_har', pamap2_12c: 'pamap2' }[id] || id);
  function mainPaperObservations(data) {
    const main = new Map((data.groups || []).map(g => [g.id, g.method_id]));
    return transferObservations(data).filter(r => main.get(r.protocol) === r.method.id).map(r => ({ ...r, dataset: mainDataset(r.dataset), paperGroup: r.protocol }));
  }
  function paperAverages(data, normalizeDatasets = false) {
    const out = [];
    for (const g of data.groups || []) for (const entry of [g, ...(data.comparisons?.[g.id] || [])]) {
      const factor = g.native_unit === 'fraction_0_1' ? 1 : .01;
      const f1 = /macro/i.test(g.metric_names?.f1 || '') ? 'f1_macro' : 'f1_score';
      for (const [native, metric] of [['accuracy', 'accuracy'], ['f1', f1]]) {
        const value = entry.paper_reported_average?.[native];
        if (finite(value)) out.push({ method: entry.method_id, protocol: g.id, metric, value: value * factor, datasets: (entry.results || []).map(r => normalizeDatasets ? mainDataset(r.target_id) : r.target_id) });
      }
    }
    return out;
  }
  function sortRows(rows, direction = 'best') {
    return [...rows].sort((a, b) => {
      const delta = a.direction === 'lower' ? a.value - b.value : b.value - a.value;
      return (direction === 'reverse' ? -delta : delta) || a.method.name.localeCompare(b.method.name);
    });
  }
  function groupRows(rows, direction = 'best') {
    const groups = new Map();
    for (const r of rows) {
      const key = `${r.protocol}|${r.metric}|${r.direction}|${r.unit}`;
      if (!groups.has(key)) groups.set(key, { key, label: r.protocolLabel, metric: r.metricLabel, note: r.protocolNote, rows: [] });
      groups.get(key).rows.push(r);
    }
    return [...groups.values()].map(g => ({ ...g, rows: sortRows(g.rows, direction) })).sort((a, b) => b.rows.length - a.rows.length || a.label.localeCompare(b.label));
  }
  function overviewRows(observations, datasetIds, registry) {
    const methods = methodIndex(registry);
    return CORE.map(id => ({ method: methods.get(id) || { id, name: META[id][0], year: META[id][1], venue: META[id][2] }, scores: Object.fromEntries(datasetIds.map(dataset => {
      const variants = { wisdm: ['wisdm_18c'], opportunity_a: ['opportunity_gesture_18c', 'opportunity_4c', 'opportunity_gesture_17c'], pamap2: ['pamap2_9c'] };
      const options = observations.filter(r => r.method.id === id && (r.dataset === dataset || variants[dataset]?.includes(r.dataset)));
      const chosen = options.find(r => ['onehar-joint', 'mosaic-full-shot'].includes(r.protocol)) || options.find(r => r.choice && r.paperRole === 'original') || options.find(r => r.choice) || options.find(r => r.source === 'paper') || options.find(r => r.protocol === 'onehar_released_checkpoint') || options.find(r => r.protocol === 'raw-v2') || options[0] || null;
      return [dataset, chosen];
    })) })).sort((a, b) => Object.values(b.scores).filter(Boolean).length - Object.values(a.scores).filter(Boolean).length || a.method.name.localeCompare(b.method.name));
  }
  function sharedDatasets(observations) {
    const byMethod = new Map();
    for (const row of observations) {
      if (!byMethod.has(row.method.id)) byMethod.set(row.method.id, new Set());
      byMethod.get(row.method.id).add(row.dataset);
    }
    // Select by availability alone, never by a model's performance.
    const panels = [...byMethod.values()].filter(set => set.size >= 2);
    if (!panels.length) return [];
    return [...panels[0]].filter(id => panels.every(set => set.has(id))).sort();
  }
  function aggregateRows(observations, datasetIds, metric, order = 'best', averages = []) {
    const ids = [...new Set(datasetIds)];
    const pool = observations.map(row => forMetric(row, metric)).filter(row => row && ids.includes(row.dataset));
    const signatures = new Set(pool.map(row => `${row.unit}:${row.direction}`));
    if (signatures.size > 1) throw new Error('An overall mean requires matching metric units and direction.');
    const methods = new Map();
    for (const row of pool) {
      if (!methods.has(row.method.id)) methods.set(row.method.id, { method: row.method, scores: {} });
      const entry = methods.get(row.method.id);
      const current = entry.scores[row.dataset];
      const priority = r => ['onehar-joint', 'mosaic-full-shot'].includes(r.protocol) ? 0 : r.choice && r.paperRole === 'original' ? 1 : r.choice ? 2 : r.source === 'paper' ? 3 : r.protocol === 'onehar_released_checkpoint' ? 4 : r.protocol === 'raw-v2' ? 5 : 6;
      if (!current || priority(row) < priority(current)) entry.scores[row.dataset] = row;
    }
    const result = [...methods.values()].map(entry => {
      const values = ids.map(id => entry.scores[id]).filter(Boolean);
      const complete = ids.length >= 2 && values.length === ids.length;
      const calculatedMean = values.length ? values.reduce((sum, row) => sum + row.value, 0) / values.length : null;
      const published = complete && averages.find(a => a.method === entry.method.id && a.metric === metric && a.datasets.length === ids.length && ids.every(id => a.datasets.includes(id)) && values.every(r => r.protocol === a.protocol));
      return { ...entry, datasetIds: ids, coverage: values.length, total: ids.length,
        complete, calculatedMean, averageKind: published ? 'paper' : 'calculated',
        mean: published ? published.value : calculatedMean,
        sourceRuns: values.filter(row => row.source === 'run').length,
        sourcePapers: values.filter(row => row.source === 'paper').length,
        direction: values[0]?.direction || 'higher', unit: values[0]?.unit || 'percent', metric };
    });
    return result.sort((a, b) => {
      if (a.complete !== b.complete) return a.complete ? -1 : 1;
      if (!a.complete && a.coverage !== b.coverage) return b.coverage - a.coverage;
      const delta = a.direction === 'lower' ? a.mean - b.mean : b.mean - a.mean;
      return (order === 'reverse' ? -delta : delta) || a.method.name.localeCompare(b.method.name);
    });
  }
  root.IoTBenchRanking = Object.freeze({ CORE, META, metricName, finite, metricValue, eligibleLocal, choosePaperScore, fullRun, forMetric, imuObservations, qaObservations, transferObservations, mainPaperObservations, paperAverages, sortRows, groupRows, overviewRows, sharedDatasets, aggregateRows });
})(globalThis);
