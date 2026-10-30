(() => {
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const datasets = ['uci_har', 'wisdm', 'pamap2', 'hhar', 'opportunity_a'];
  const names = {uci_har:'UCI-HAR',wisdm:'WISDM',pamap2:'PAMAP2',hhar:'HHAR',opportunity_a:'OPPORTUNITY · gestures'};
  const states = {running:'Training',queued:'Queued',complete:'Complete · calibration audit pending',failed:'Failed · review needed',source_unavailable:'Author code not located',adapter_audit_pending:'Adapter audit pending',paper_fulltext_needed:'Paper full text needed',training_assets_missing:'Training assets needed'};
  function link(url, label) { return /^https:\/\//.test(url || '') ? `<a href="${escape(url)}" target="_blank" rel="noopener">${label} ↗</a>` : ''; }
  async function load() {
    try {
      const response = await fetch('data/imu_reproduction_20261028.json', {cache:'no-store'});
      if (!response.ok) throw Error(`HTTP ${response.status}`);
      const data = await response.json();
      const done = data.cells.filter(c => c.status === 'complete').length;
      const folds = data.cells.reduce((n,c) => n + c.completed_folds, 0);
      document.getElementById('summary').textContent = `${done}/50 combinations complete · ${folds} individual runs complete · updated ${new Date(data.updated_at).toLocaleString()} · refreshes every minute`;
      const methods = [...new Set(data.cells.map(c => c.method))];
      document.getElementById('runs').innerHTML = methods.map(method => {
        const cells = data.cells.filter(c => c.method === method);
        const source = cells[0].source;
        const rows = datasets.map(dataset => {
          const c = cells.find(c => c.dataset === dataset);
          const primary = dataset === 'opportunity_a' ? 'f1_weighted' : 'accuracy';
          const score = c.status === 'complete' ? `${(c.metrics[primary]*100).toFixed(2)} ± ${(c.metrics[primary+'_std']*100).toFixed(2)}%` : '—';
          const cls = c.status === 'complete' ? 'run-complete' : c.status === 'running' ? 'run-running' : c.status === 'failed' ? 'run-error' : '';
          const status = c.validation_status === 'standard_eligible' ? 'Complete · Standard eligible' : states[c.status] || c.status;
          return `<tr><th scope="row">${names[dataset]}</th><td>${score}<br><span class="muted">${primary === 'accuracy' ? 'Accuracy' : 'Weighted F1'}</span></td><td>${c.completed_folds}/${c.required_folds}</td><td class="${cls}">${escape(status)}${c.errors.length ? '<br>'+escape(c.errors.map(e => e.log).join('; ')) : ''}</td></tr>`;
        }).join('');
        return `<section class="run-method" id="${escape(method)}"><h2>${escape(source.name)}</h2><div class="run-links">${link(source.paper,'Original paper')}${link(source.repo,'Author repository')}</div><p>${escape(source.note)}</p><div class="table-scroll"><table><thead><tr><th>Dataset</th><th>Mean ± std</th><th>Runs</th><th>Status</th></tr></thead><tbody>${rows}</tbody></table></div></section>`;
      }).join('');
      document.getElementById('error').textContent = '';
    } catch (error) {
      document.getElementById('error').textContent = `Could not refresh run status: ${error.message}. Previously displayed results may be stale.`;
    }
  }
  load(); setInterval(load, 60000);
})();
