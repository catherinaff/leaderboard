export function newUploadKey() {
  // getRandomValues works on this HTTP IP site as well as on HTTPS.
  return [...crypto.getRandomValues(new Uint8Array(24))].map(x => x.toString(16).padStart(2, '0')).join('');
}

export function uploadSubmission(draft, files, key, onProgress = () => {}, createRequest = () => new XMLHttpRequest()) {
  return new Promise((resolve, reject) => {
    const request = createRequest();
    const data = new FormData();
    data.append('metadata', JSON.stringify(draft));
    for (const file of files) data.append('files', file, file.name);
    request.upload.addEventListener('progress', event => {
      if (event.lengthComputable) onProgress(Math.min(100, Math.floor(event.loaded / event.total * 100)));
    });
    request.open('POST', 'api/submissions');
    request.setRequestHeader('Idempotency-Key', key);
    request.timeout = 30 * 60 * 1000;
    request.onload = () => {
      let result;
      try { result = JSON.parse(request.responseText); }
      catch { reject(new Error('The server did not confirm storage. Please retry this draft.')); return; }
      if (request.status >= 200 && request.status < 300 && /^sub_\d{8}T\d{6}Z_[0-9a-f]{24}$/.test(result.submission_id) && result.status === 'received') resolve(result);
      else reject(new Error(typeof result.error === 'string' ? result.error : 'The upload could not be saved. Please retry.'));
    };
    request.onerror = () => reject(new Error('Connection interrupted. Your draft is retained; retry to confirm storage.'));
    request.ontimeout = () => reject(new Error('Storage confirmation timed out. Your draft is retained; retry the same draft.'));
    request.onabort = () => reject(new Error('Upload interrupted. Your draft is retained.'));
    request.send(data);
  });
}
