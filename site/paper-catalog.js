(async function () {
  'use strict';
  const root = document.getElementById('paper-dataset-catalog');
  if (!root) return;
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  try {
    const response = await fetch('data/paper_transfer_evidence.json', { cache: 'no-store' });
    if (!response.ok) throw new Error('Paper data unavailable');
    const scope = IoTBenchRanking.paperScope(await response.json());
    const count = document.getElementById('paper-scope-count');
    if (count) count.textContent = `${scope.familyCount} datasets · ${scope.catalog.length} label sets · ${scope.methods.length} comparison methods`;
    const catalog = [...scope.catalog].sort((a, b) => Number(scope.common.includes(b.id)) - Number(scope.common.includes(a.id)) || a.name.localeCompare(b.name) || a.classes - b.classes);
    function render(query = '') {
      const list = catalog.filter(d => `${d.name} ${d.classes} ${d.papers.join(' ')}`.toLowerCase().includes(query.toLowerCase()));
      root.innerHTML = `<div class="paper-catalog-scroll"><table class="paper-catalog-table"><caption class="paper-catalog-caption">${list.length} reported label sets · source paper coverage</caption><thead><tr><th>Dataset / label set</th><th>MOSAIC · 2026</th><th>oneHAR · 2025</th><th>Results</th></tr></thead><tbody>${list.map(d => `<tr data-paper-dataset="${esc(d.id)}"><th scope="row">${esc(d.name)} <span class="paper-class-count">${esc(d.classes)} classes</span>${scope.common.includes(d.id) ? '<span class="paper-shared-badge">Overall ranking</span>' : ''}</th><td>${d.papers.includes('mosaic') ? '✓' : '—'}</td><td>${d.papers.includes('onehar') ? '✓' : '—'}</td><td><a href="leaderboard.html?board=imu&amp;view=ranking&amp;dataset=${encodeURIComponent(d.id)}">Dataset ranking ↗</a></td></tr>`).join('')}</tbody></table></div>${list.length ? '' : '<p>No matching dataset.</p>'}`;
    }
    render();
    document.getElementById('paper-dataset-search')?.addEventListener('input', event => render(event.target.value));
  } catch (error) {
    root.textContent = 'Dataset catalog could not be loaded. Refresh to try again.';
    console.error(error);
  }
})();
