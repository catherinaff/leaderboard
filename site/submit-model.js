import { addFiles, fileKey, formatSize, DATASETS, PROTOCOLS, validateDraft } from './submission-utils.js';
import { newUploadKey, uploadSubmission } from './upload-client.js';

const $ = id => document.getElementById(id);
const form = $('model-form');
const picker = $('code-files');
const preview = $('submission-preview');
const drop = $('drop-zone');
const dialog = $('clear-dialog');
let files = [];
let dialogTrigger;
let dragDepth = 0;
let serviceReady = false, submitting = false, reviewedDraft = null, reviewedFiles = [], uploadKey = '', receipt = null;
const esc = value => String(value).replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));
const say = text => { $('submission-status').textContent = text; };
const selected = name => [...form.querySelectorAll(`input[name="${name}"]:checked`)].map(input => input.value);
const labelDatasets = ids => ids.map(id => DATASETS.find(([key]) => key === id)?.[1] || id).join(', ');

for (const [type, container] of [['targets','target-options'], ['sources','source-options']]) {
  $(container).innerHTML = DATASETS.map(([id, label]) => `<label class="dataset-choice"><input type="checkbox" name="${type}" value="${id}"><span>${label}</span></label>`).join('');
}
$('protocol').innerHTML += PROTOCOLS.map(([id,label]) => `<option value="${id}">${label}</option>`).join('');
const preselected = new URLSearchParams(location.search).get('bench');
const initialDataset = form.querySelector(`input[name="targets"][value="${DATASETS.some(([id]) => id === preselected) ? preselected : ''}"]`);
if (initialDataset) initialDataset.checked = true;

function renderFiles() {
  const list = $('selected-files');
  list.replaceChildren();
  for (const file of files) {
    const li = document.createElement('li');
    const info = document.createElement('div'); info.className = 'selected-file-info';
    const name = document.createElement('span'); name.textContent = file.name;
    const size = document.createElement('small'); size.textContent = formatSize(file.size);
    info.append(name, size);
    const remove = document.createElement('button');
    remove.type = 'button'; remove.className = 'text-button'; remove.textContent = 'Remove';
    remove.setAttribute('aria-label', `Remove ${file.name}`);
    remove.addEventListener('click', () => {
      files = files.filter(item => fileKey(item) !== fileKey(file));
      renderFiles(); clearErrors(); say(`Removed ${file.name}. ${files.length} files selected.`);
      picker.focus();
    });
    li.append(info, remove); list.append(li);
  }
  $('remove-all').hidden = !files.length;
  $('file-summary').textContent = files.length ? `${files.length} file${files.length === 1 ? '' : 's'} · ${formatSize(files.reduce((sum,file) => sum + file.size, 0))} selected locally` : 'No files selected';
}

function receive(incoming) {
  const result = addFiles(files, [...incoming]);
  const added = result.files.length - files.length;
  files = result.files;
  renderFiles();
  const errors = $('file-errors');
  errors.hidden = !result.errors.length;
  errors.innerHTML = result.errors.length ? '<strong>Some files were not added.</strong><ul>' + result.errors.map(message => `<li>${esc(message)}</li>`).join('') + '</ul>' : '';
  if (added) { clearErrors(); say(`${added} file${added === 1 ? '' : 's'} added locally. Nothing uploaded.`); }
  else if (result.errors.length) say('No new files added. Review the file messages.');
}

picker.addEventListener('change', () => { receive(picker.files || []); picker.value = ''; });
picker.addEventListener('cancel', () => say('File selection canceled. Your current draft is unchanged.'));
$('remove-all').addEventListener('click', () => { files = []; picker.value = ''; renderFiles(); $('file-errors').hidden = true; clearErrors(); say('All selected files removed. Model details are unchanged.'); picker.focus(); });

// File selection and preview stay local until the explicit Upload button.
document.addEventListener('dragover', event => { event.preventDefault(); });
document.addEventListener('drop', event => { event.preventDefault(); dragDepth = 0; drop.classList.remove('drag-active'); });
drop.addEventListener('dragenter', event => { event.preventDefault(); dragDepth++; drop.classList.add('drag-active'); });
drop.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; drop.classList.remove('drag-active'); } });
drop.addEventListener('dragover', event => { event.preventDefault(); if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy'; });
drop.addEventListener('drop', event => { event.preventDefault(); receive(event.dataTransfer?.files || []); });

function updateProtocol() {
  const protocol = $('protocol').value;
  $('source-datasets').hidden = !['pairwise','lodo'].includes(protocol);
  $('synthetic-field').hidden = protocol !== 'synthetic';
  $('budget-field').hidden = protocol !== 'fewshot';
  $('protocol-note').textContent = ({
    within: 'Supervised within-dataset evaluation. Specify subject separation, classes and train/test split below.',
    pairwise: 'Choose one source and one different target. This describes one explicit transfer direction.',
    lodo: 'Choose two or more sources and exactly one held-out target. No held-out target training data is assumed.',
    synthetic: 'Synthetic corpus → unseen real datasets. Zero target training labels; no target fine-tuning.',
    fewshot: 'Target-supervised adaptation. State the labeled samples per class; this is separate from zero-shot transfer.',
  })[protocol] || 'Different datasets, class sets and splits are separate evaluation settings.';
}
$('protocol').addEventListener('change', updateProtocol);

function readDraft() {
  const protocol = $('protocol').value;
  return {
    name: $('model-name').value.trim(), framework: $('framework').value,
    description: $('description').value.trim(), entryPoint: $('entry-point').value.trim(), repository: $('repository').value.trim(),
    protocol, targets: selected('targets'), sources: ['pairwise','lodo'].includes(protocol) ? selected('sources') : [],
    syntheticSource: protocol === 'synthetic' ? $('synthetic-source').value.trim() : '',
    labelsPerClass: protocol === 'fewshot' ? $('label-budget').value : '', variantNotes: $('variant-notes').value.trim(),
  };
}

function clearErrors() {
  $('validation-errors').hidden = true;
  $('validation-errors').replaceChildren();
  form.querySelectorAll('[aria-invalid="true"]').forEach(input => input.removeAttribute('aria-invalid'));
}

function focusField(id) {
  const field = $(id);
  (field.querySelector?.('input') || field).focus();
}

function showErrors(errors) {
  const summary = $('validation-errors');
  summary.hidden = false;
  summary.innerHTML = '<strong>Check your draft before previewing.</strong><ul>' + errors.map(error => `<li><a href="#${error.id}">${esc(error.message)}</a></li>`).join('') + '</ul>';
  summary.querySelectorAll('a').forEach(link => link.addEventListener('click', event => { event.preventDefault(); focusField(link.getAttribute('href').slice(1)); }));
  for (const error of errors) $(error.id).setAttribute('aria-invalid', 'true');
  summary.focus();
}

form.addEventListener('submit', event => {
  event.preventDefault(); clearErrors();
  const draft = readDraft();
  const errors = validateDraft(draft, files);
  if (errors.length) { showErrors(errors); return; }
  reviewedDraft = draft; reviewedFiles = [...files]; uploadKey = newUploadKey(); receipt = null; $('preview-badge').textContent = 'Not submitted';
  $('review-notice').hidden = false; $('upload-error').hidden = true; $('upload-receipt').hidden = true; $('upload-progress').hidden = true;
  $('upload-submit').hidden = false; $('edit-draft').hidden = false; $('clear-preview').hidden = false;
  $('upload-submit').disabled = !serviceReady;
  const detail = (label, value) => value ? `<dt>${label}</dt><dd>${esc(value)}</dd>` : '';
  $('preview-content').innerHTML = '<div class="preview-grid"><section><h4>Model</h4><dl>'
    + detail('Name', draft.name) + detail('Framework', draft.framework) + detail('Description', draft.description)
    + detail('Entry point (not executed)', draft.entryPoint) + detail('Repository / paper', draft.repository)
    + '</dl></section><section><h4>Evaluation plan</h4><dl>'
    + detail('Protocol', PROTOCOLS.find(([id]) => id === draft.protocol)[1])
    + detail('Evaluation datasets', labelDatasets(draft.targets)) + detail('Source datasets', labelDatasets(draft.sources))
    + detail('Synthetic source', draft.syntheticSource)
    + detail('Target label budget', ['lodo','synthetic'].includes(draft.protocol) ? '0 target training labels; no target fine-tuning' : draft.protocol === 'fewshot' ? `${draft.labelsPerClass} labeled samples per class` : '')
    + detail('Dataset variants & split', draft.variantNotes) + '</dl></section></div><section class="preview-files"><h4>Files to upload</h4><ul class="preview-file-list">'
    + files.map(file => `<li>${esc(file.name)} · ${formatSize(file.size)}</li>`).join('') + '</ul></section>';
  form.hidden = true; preview.hidden = false; say('Local preview ready. Nothing has been submitted.');
  $('preview-heading').focus();
});
$('edit-draft').addEventListener('click', () => { if (submitting) return; preview.hidden = true; form.hidden = false; reviewedDraft = null; uploadKey = ''; say('Editing your local draft. Selected files are retained.'); $('model-name').focus(); });

function askClear(event) { if (submitting) return; dialogTrigger = event.currentTarget; dialog.showModal(); }
$('clear-draft').addEventListener('click', askClear);
$('clear-preview').addEventListener('click', askClear);
$('keep-draft').addEventListener('click', () => { dialog.close(); dialogTrigger?.focus(); });
dialog.addEventListener('cancel', () => say('Clear canceled. Your current draft is unchanged.'));
function resetDraft() {
  form.reset(); files = []; picker.value = ''; renderFiles(); updateProtocol(); clearErrors();
  reviewedDraft = null; reviewedFiles = []; uploadKey = ''; receipt = null;
  $('file-errors').hidden = true; $('file-errors').replaceChildren();
  $('preview-content').replaceChildren(); preview.hidden = true; form.hidden = false;
  say('Local draft cleared. Original files on your computer are unchanged.'); picker.focus();
}
$('confirm-clear').addEventListener('click', () => { dialog.close(); resetDraft(); });
$('new-submission').addEventListener('click', resetDraft);

async function checkService() {
  try {
    const response = await fetch('api/submissions/config', { cache: 'no-store' });
    const config = await response.json();
    serviceReady = response.ok && config.enabled === true;
  } catch { serviceReady = false; }
  $('service-status').textContent = serviceReady ? 'Upload service ready · files are stored for review.' : 'Upload service unavailable. You can prepare a draft and retry the connection.';
  $('retry-service').hidden = serviceReady;
  $('upload-submit').disabled = !serviceReady || submitting || Boolean(receipt);
}
$('retry-service').addEventListener('click', checkService);
$('upload-submit').addEventListener('click', async () => {
  if (submitting || !serviceReady || !reviewedDraft || receipt) return;
  submitting = true;
  for (const id of ['upload-submit', 'edit-draft', 'clear-preview']) $(id).disabled = true;
  $('upload-error').hidden = true; $('upload-progress').hidden = false; $('upload-meter').value = 0;
  $('upload-progress-text').textContent = 'Uploading 0%'; say('Uploading your model. Keep this page open until storage is confirmed.');
  try {
    receipt = await uploadSubmission(reviewedDraft, reviewedFiles, uploadKey, percent => {
      $('upload-meter').value = percent;
      $('upload-progress-text').textContent = percent < 100 ? `Uploading ${percent}%` : 'Upload sent. Waiting for storage confirmation…';
    });
    $('receipt-id').textContent = receipt.submission_id;
    $('receipt-summary').textContent = `${receipt.model_name} · ${receipt.file_count} file${receipt.file_count === 1 ? '' : 's'} · ${formatSize(receipt.total_bytes)} · Received for review`;
    $('upload-receipt').hidden = false; $('review-notice').hidden = true; $('upload-progress').hidden = true;
    for (const id of ['upload-submit', 'edit-draft', 'clear-preview']) $(id).hidden = true;
    $('preview-badge').textContent = 'Saved on server';
    say(`Model saved on the server. Submission ${receipt.submission_id}.`);
    $('receipt-heading').focus();
  } catch (error) {
    $('upload-progress').hidden = true; $('upload-error').hidden = false;
    $('upload-error').textContent = error.message;
    say('Upload was not confirmed. Your draft and selected files are retained for retry.');
  } finally {
    submitting = false;
    for (const id of ['upload-submit', 'edit-draft', 'clear-preview']) $(id).disabled = false;
  }
});
$('download-receipt').addEventListener('click', () => {
  if (!receipt) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(receipt, null, 2) + '\n'], { type: 'application/json' }));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = `${receipt.submission_id}.json`; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
window.addEventListener('beforeunload', event => { if (submitting) { event.preventDefault(); event.returnValue = ''; } });

renderFiles(); updateProtocol();
$('preview-button').disabled = false;
checkService();
