/* Resolve application links under either / or a website subdirectory. */
(function () {
  const base = new URL('.', document.currentScript.src);
  globalThis.IoTBenchPath = function (value) {
    const text = String(value);
    return new URL(text.startsWith('/') && !text.startsWith('//') ? '.' + text : text, base).href;
  };
})();
