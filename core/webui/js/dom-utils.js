import { state } from './state.js';

export function debounce(fn, delayMs = 120) {
  let timer = null;
  return function debounced(...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), delayMs);
  };
}

export function isUserEditingDom() {
  if (state.isEditingTerms) return true;
  const active = document.activeElement;
  if (!active || !active.matches) return false;
  return active.matches('[data-term-field], #terms-theme-input, #domain-csv-tbody input');
}

export function autoResizeThemeTextarea(el) {
  if (!el) return;
  el.style.height = 'auto';
  const next = Math.min(96, Math.max(32, (el.scrollHeight || 30) + 2));
  el.style.height = `${next}px`;
}

export async function apiFetch(url, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body && !(options.body instanceof FormData) && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }
  const res = await fetch(url, { ...options, headers });
  if (!res.ok) {
    let msg = res.statusText || `HTTP ${res.status}`;
    const rawText = await res.text();
    if (rawText) {
      try {
        const data = JSON.parse(rawText);
        msg = data.detail || data.message || JSON.stringify(data);
      } catch (parseErr) {
        msg = rawText.slice(0, 200) || parseErr.message || msg;
      }
    }
    throw new Error(msg);
  }
  return res.json();
}

export function escapeHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

export function escapeAttr(str) {
  return escapeHtml(str).replace(/"/g, '&quot;');
}

export function formatTimecode(sec) {
  const s = Math.max(0, Math.floor(Number(sec || 0)));
  const hh = String(Math.floor(s / 3600)).padStart(2, '0');
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  return `${hh}:${mm}:${ss}`;
}

export function formatWordTimestamp(sec) {
  const val = Math.max(0, Number(sec || 0));
  const mm = String(Math.floor(val / 60)).padStart(2, '0');
  const ss = (val % 60).toFixed(2).padStart(5, '0');
  return `${mm}:${ss}`;
}

export function sanitizeErrorSummary(rawErr) {
  const text = String(rawErr || '').trim();
  if (!text) return '执行异常 · 查看运行日志';
  const hasCodeNoise = /traceback|exception|error:|is not defined|errno|file "|line \d+|https?:\/\//i.test(text);
  if (hasCodeNoise || text.length > 26) {
    return '执行异常 · 查看运行日志';
  }
  return text;
}

export function setAutosaveStatus(statusState, text) {
  const saveBadge = document.getElementById('autosave-indicator');
  if (!saveBadge) return;
  saveBadge.className = text ? `status-label ${statusState}` : 'status-label';
  saveBadge.textContent = text || '';
}

let supportsDisplayTransition;
function canTransitionDisplay() {
  if (supportsDisplayTransition !== undefined) return supportsDisplayTransition;
  if (!window.CSS?.supports?.('transition-behavior', 'allow-discrete') || !document.body) {
    return false;
  }
  const probe = document.createElement('div');
  probe.style.cssText = 'transition: display 1s allow-discrete !important; display: block;';
  document.body.appendChild(probe);
  getComputedStyle(probe).display;
  probe.style.display = 'none';
  supportsDisplayTransition = getComputedStyle(probe).display === 'block';
  probe.remove();
  return supportsDisplayTransition;
}

function supportsTopLayerExit() {
  return window.CSS?.supports?.('overlay', 'auto') && canTransitionDisplay();
}

export async function closeTopLayer(dialogEl) {
  if (!dialogEl || !dialogEl.open) return;
  if (!supportsTopLayerExit()) {
    dialogEl.setAttribute('data-closing', '');
    const animations = dialogEl.getAnimations({ subtree: true });
    if (animations.length > 0) {
      await Promise.race([
        Promise.allSettled(animations.map(a => a.finished)),
        new Promise(r => setTimeout(r, 400)),
      ]);
    }
    if (!dialogEl.hasAttribute('data-closing')) return;
    dialogEl.removeAttribute('data-closing');
  }
  dialogEl.close();
}

export function wireModalDialog(dialogEl) {
  if (!dialogEl) return;
  dialogEl.addEventListener('cancel', event => {
    event.preventDefault();
    closeTopLayer(dialogEl);
  });
  if (!('closedBy' in HTMLDialogElement.prototype)) {
    dialogEl.addEventListener('click', event => {
      if (event.target !== dialogEl) return;
      const rect = dialogEl.getBoundingClientRect();
      const isDialogContent =
        rect.top <= event.clientY &&
        event.clientY <= rect.top + rect.height &&
        rect.left <= event.clientX &&
        event.clientX <= rect.left + rect.width;
      if (isDialogContent) return;
      closeTopLayer(dialogEl);
    });
  }
}
