export const LIMITS = Object.freeze({ count: 20, each: 512 * 1024 * 1024, total: 1024 * 1024 * 1024 });
export const EXTENSIONS = ['.zip', '.tar.gz', '.tgz', '.pt', '.pth', '.onnx', '.safetensors', '.ckpt', '.bin', '.h5', '.hdf5', '.keras', '.tflite', '.py', '.ipynb', '.js', '.ts', '.json', '.yaml', '.yml', '.toml', '.txt', '.md', '.cfg', '.ini', '.sh'];
export const DATASETS = [
  ['uci_har', 'UCI-HAR'], ['hhar', 'HHAR'], ['pamap2', 'PAMAP2'], ['realworld', 'RealWorld'],
  ['usc_had', 'USC-HAD'], ['wisdm', 'WISDM'], ['opportunity', 'OPPORTUNITY'], ['motionsense', 'MotionSense'],
];
export const PROTOCOLS = [
  ['within', 'Within-dataset · supervised'], ['pairwise', 'Pairwise source → target'],
  ['lodo', 'Multi-source · held-out dataset (LODO)'], ['synthetic', 'Synthetic → real · zero-shot'],
  ['fewshot', 'Few-shot target adaptation'],
];
export const FRAMEWORKS = ['PyTorch', 'TensorFlow / Keras', 'JAX', 'Other'];
export const fileKey = file => `${file.name}\u0000${file.size}\u0000${file.lastModified || 0}`;
export const formatSize = bytes => bytes < 1024 ? `${bytes} B` : bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : bytes < 1024 ** 3 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${(bytes / 1024 ** 3).toFixed(1)} GB`;

// File objects are referenced only. No file contents are read, extracted or run.
export function addFiles(current, incoming) {
  const files = [...current];
  const errors = [];
  let total = files.reduce((sum, file) => sum + file.size, 0);
  for (const file of incoming) {
    const lower = file.name.toLowerCase();
    let reason = '';
    if (!EXTENSIONS.some(ext => lower.endsWith(ext))) reason = 'Unsupported file type. Choose a model checkpoint, code archive or supported source file.';
    else if (file.name.length > 240 || /[/\\\u0000-\u001f\u007f]/.test(file.name)) reason = 'Invalid filename.';
    else if (!Number.isFinite(file.size) || file.size < 0) reason = 'The file size could not be read.';
    else if (file.size > LIMITS.each) reason = 'Exceeds the 512 MB per-file limit.';
    else if (files.some(existing => fileKey(existing) === fileKey(file))) reason = 'Already selected.';
    else if (files.some(existing => existing.name === file.name)) reason = 'A file with this name is already selected.';
    else if (files.length >= LIMITS.count) reason = 'A maximum of 20 files can be selected.';
    else if (total + file.size > LIMITS.total) reason = 'Would exceed the 1 GB total limit.';
    if (reason) errors.push(`${file.name}: ${reason}`);
    else { files.push(file); total += file.size; }
  }
  return { files, errors };
}

export function validateDraft(draft, files) {
  const errors = [];
  const issue = (id, message) => errors.push({ id, message });
  if (!files.length) issue('code-files', 'Select at least one code archive or source file.');
  if (!draft.name.trim() || draft.name.length > 100) issue('model-name', 'Enter a model name (up to 100 characters).');
  if (!FRAMEWORKS.includes(draft.framework)) issue('framework', 'Choose a framework.');
  if (draft.description.length > 400) issue('description', 'Keep the description within 400 characters.');
  if (draft.entryPoint.length > 120) issue('entry-point', 'Keep the entry point within 120 characters.');
  if (draft.repository) {
    try { const url = new URL(draft.repository); if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) throw new Error(); }
    catch { issue('repository', 'Enter a public http:// or https:// URL without credentials, or leave it blank.'); }
  }
  if (!PROTOCOLS.some(([id]) => id === draft.protocol)) issue('protocol', 'Choose an evaluation protocol.');
  const known = new Set(DATASETS.map(([id]) => id));
  if (!draft.targets.length || draft.targets.some(id => !known.has(id))) issue('target-datasets', 'Select at least one evaluation dataset.');
  if (['pairwise', 'lodo'].includes(draft.protocol)) {
    if (draft.targets.length !== 1) issue('target-datasets', 'Choose exactly one held-out target for this transfer direction.');
    const enough = draft.protocol === 'pairwise' ? draft.sources.length === 1 : draft.sources.length >= 2;
    if (!enough || draft.sources.some(id => !known.has(id))) issue('source-datasets', draft.protocol === 'pairwise' ? 'Choose exactly one source dataset.' : 'Choose at least two source datasets.');
    if (draft.sources.some(id => draft.targets.includes(id))) issue('source-datasets', 'Source datasets must exclude the held-out target.');
  }
  if (draft.protocol === 'synthetic' && !draft.syntheticSource.trim()) issue('synthetic-source', 'Describe the synthetic pretraining source.');
  if (draft.protocol === 'fewshot' && (!Number.isInteger(Number(draft.labelsPerClass)) || Number(draft.labelsPerClass) < 1 || Number(draft.labelsPerClass) > 10000)) issue('label-budget', 'Enter a whole-number label budget from 1 to 10,000 samples per class.');
  if (!draft.variantNotes.trim() || draft.variantNotes.length > 600) issue('variant-notes', 'Describe the dataset variant, classes and evaluation split (up to 600 characters).');
  return errors;
}
