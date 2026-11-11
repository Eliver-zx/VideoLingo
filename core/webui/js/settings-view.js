import { state, TASK_DETAIL_CACHE } from './state.js';
import { apiFetch, escapeAttr, setAutosaveStatus } from './dom-utils.js';
import { appendLog, reportActionError } from './console-log.js';
import { renderDomainList } from './glossary-view.js';
import {
  normalizeAssColorToHex,
  updateAssPreview,
  initAssStyleEvents,
} from './ass-preview.js';

export { updateAssPreview };

let saveTimer = null;
const pendingConfigUpdates = {};

const PLACEHOLDER_API_KEYS = new Set([
  'YOUR_API_KEY',
  'your_elevenlabs_api_key',
  'YOUR_FISH_AUDIO_KEY',
  'YOUR_SF_KEY',
]);

export function switchSettingsSection(targetId) {
  document.querySelectorAll('.settings-nav-item').forEach(i => {
    i.classList.toggle('active', i.getAttribute('data-settings-page') === targetId);
  });
  document.querySelectorAll('.settings-page').forEach(page => {
    page.classList.toggle('hidden', page.id !== targetId);
  });
  const canvas = document.querySelector('.settings-canvas');
  if (canvas) canvas.scrollTop = 0;
}

function enhancePasswordInputs() {
  const eyeOpenSvg =
    '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
  const eyeOffSvg =
    '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>';
  document.querySelectorAll('#settings-form input[data-secret="true"]').forEach(inp => {
    if (inp.parentElement?.classList.contains('pwd-wrap')) return;
    const wrap = document.createElement('span');
    wrap.className = 'pwd-wrap';
    inp.parentNode?.insertBefore(wrap, inp);
    wrap.appendChild(inp);
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'pwd-toggle';
    btn.title = '显示或隐藏密钥';
    btn.setAttribute('aria-label', '显示或隐藏密钥');
    btn.innerHTML = eyeOpenSvg;
    btn.addEventListener('click', e => {
      e.preventDefault();
      e.stopPropagation();
      btn.innerHTML = inp.classList.toggle('is-revealed') ? eyeOffSvg : eyeOpenSvg;
    });
    wrap.appendChild(btn);
  });
}

function syncLocalBackendRows() {
  const backend = document.getElementById('cfg-local-backend')?.value || 'qwen';
  document.querySelectorAll('[data-local-only]').forEach(row => {
    row.classList.toggle('hidden', row.getAttribute('data-local-only') !== backend);
  });
}

function syncMaiProviderRows() {
  const provider = document.getElementById('cfg-mai-provider')?.value || 'azure';
  document.querySelectorAll('[data-mai-only]').forEach(row => {
    row.classList.toggle('hidden', row.getAttribute('data-mai-only') !== provider);
  });
}

function syncDemucsRow() {
  const demucsChk = document.getElementById('cfg-demucs-chk');
  const row = document.getElementById('row-demucs-model');
  if (row && demucsChk) row.classList.toggle('hidden', !demucsChk.checked);
}

function syncAsrRuntimePanel() {
  const asrRuntimeSelect = document.getElementById('cfg-asr-runtime');
  if (!asrRuntimeSelect) return;
  const val = asrRuntimeSelect.value;
  const targetPanel = val === 'replicate_parakeet' || val === 'replicate_whisper' ? 'replicate' : val;
  document.querySelectorAll('#asr-engine-panels [data-asr-panel]').forEach(p => {
    p.classList.toggle('hidden', p.getAttribute('data-asr-panel') !== targetPanel);
  });
  const titleEl = document.getElementById('asr-panel-title');
  if (titleEl && asrRuntimeSelect.selectedOptions[0]) {
    titleEl.textContent = `${asrRuntimeSelect.selectedOptions[0].textContent} 参数`;
  }
  syncLocalBackendRows();
  syncMaiProviderRows();
}

function syncFishModeRows() {
  const mode = document.getElementById('cfg-fish-mode')?.value || 'clone';
  document.getElementById('row-fish-voice-id')?.classList.toggle('hidden', mode !== 'preset');
}

function syncSfFishModeRows() {
  const mode = document.getElementById('cfg-sf-fish-mode')?.value || 'preset';
  document.getElementById('row-sf-fish-voice')?.classList.toggle('hidden', mode !== 'preset');
  document.getElementById('row-sf-fish-voice-id')?.classList.toggle('hidden', mode !== 'custom');
}

function syncTtsMethodPanel() {
  const ttsMethodSelect = document.getElementById('cfg-tts-method');
  if (!ttsMethodSelect) return;
  const val = ttsMethodSelect.value;
  document.querySelectorAll('#tts-engine-panels [data-tts-panel]').forEach(p => {
    p.classList.toggle('hidden', p.getAttribute('data-tts-panel') !== val);
  });
  const titleEl = document.getElementById('tts-panel-title');
  if (titleEl && ttsMethodSelect.selectedOptions[0]) {
    titleEl.textContent = `${ttsMethodSelect.selectedOptions[0].textContent} 参数`;
  }
  syncFishModeRows();
  syncSfFishModeRows();
}

function readCfgControlValue(inp) {
  if (inp.type === 'checkbox') return inp.checked;
  if (inp.type === 'number') {
    const n = Number(inp.value);
    return Number.isNaN(n) ? inp.value : n;
  }
  return inp.value;
}

function scheduleConfigFlush() {
  setAutosaveStatus('paused', '保存中…');
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    if (state.isBackendConnected && Object.keys(pendingConfigUpdates).length > 0) {
      const payload = { ...pendingConfigUpdates };
      Object.keys(pendingConfigUpdates).forEach(k => delete pendingConfigUpdates[k]);
      try {
        await apiFetch('/api/config', {
          method: 'PUT',
          body: JSON.stringify({ updates: payload }),
        });
        setAutosaveStatus('completed', '已保存');
      } catch (err) {
        reportActionError('CONFIG', err);
      }
    } else {
      setAutosaveStatus('completed', '已保存');
    }
  }, 400);
}

function bindCfgControls() {
  document.querySelectorAll('[data-cfg]').forEach(inp => {
    const handler = () => {
      const path = inp.getAttribute('data-cfg');
      if (path) {
        const val = readCfgControlValue(inp);
        pendingConfigUpdates[path] = val;
        document.querySelectorAll(`[data-cfg="${path}"]`).forEach(peer => {
          if (peer !== inp) {
            if (peer.type === 'checkbox') peer.checked = Boolean(val);
            else peer.value = String(val ?? '');
          }
        });
        if (path === 'glossary_domain') {
          state.currentConfig.glossary_domain = val;
          TASK_DETAIL_CACHE.clear();
          const taskDomInput = document.getElementById('task-domain');
          if (taskDomInput) taskDomInput.value = String(val ?? '');
          renderDomainList();
        }
        if (path === 'ffmpeg_gpu' && !state.burnGpuTouched) {
          const burnGpuEl = document.getElementById('burn-gpu-chk');
          if (burnGpuEl) burnGpuEl.checked = Boolean(val);
        }
      }
      scheduleConfigFlush();
    };
    const evtName = inp.tagName === 'SELECT' || inp.type === 'checkbox' ? 'change' : 'input';
    inp.addEventListener(evtName, handler);
  });
}

export async function loadBackendConfig() {
  try {
    const res = await apiFetch('/api/config');
    state.isBackendConnected = true;
    state.currentConfig = res.config || res || {};
    document.querySelectorAll('[data-cfg]').forEach(inp => {
      const path = inp.getAttribute('data-cfg');
      if (!path) return;
      const val = path
        .split('.')
        .reduce((acc, part) => (acc && acc[part] !== undefined ? acc[part] : undefined), state.currentConfig);
      if (val === undefined || val === null) return;
      if (inp.type === 'checkbox') {
        inp.checked = Boolean(val);
      } else if (path === 'subtitle.style.translation.font_color' || path === 'subtitle.style.source.font_color') {
        inp.value = normalizeAssColorToHex(val, path.includes('translation') ? '#FFFF00' : '#FFFFFF');
      } else if (
        path === 'subtitle.style.translation.outline_color' ||
        path === 'subtitle.style.source.outline_color'
      ) {
        inp.value = normalizeAssColorToHex(val, '#000000');
      } else if (
        (inp.hasAttribute('data-secret') || inp.closest('.pwd-wrap')) &&
        PLACEHOLDER_API_KEYS.has(String(val).trim())
      ) {
        inp.value = '';
      } else {
        inp.value = String(val);
      }
    });
    if (!state.burnGpuTouched && state.currentConfig.ffmpeg_gpu !== undefined) {
      const burnGpuEl = document.getElementById('burn-gpu-chk');
      if (burnGpuEl) burnGpuEl.checked = Boolean(state.currentConfig.ffmpeg_gpu);
    }
    if (state.currentConfig.target_language) {
      const tgtInp = document.getElementById('task-tgt-lang');
      if (tgtInp && !tgtInp.dataset.userEdited) {
        tgtInp.value = String(state.currentConfig.target_language);
      }
    }
    if (state.currentConfig.glossary_domain !== undefined) {
      const taskDomInp = document.getElementById('task-domain');
      if (taskDomInp && !taskDomInp.dataset.userEdited) {
        taskDomInp.value = String(state.currentConfig.glossary_domain ?? '');
      }
      renderDomainList();
    }
    syncAsrRuntimePanel();
    syncDemucsRow();
    syncTtsMethodPanel();
    updateAssPreview();
  } catch (err) {
    appendLog('warn', 'CONFIG', `加载全局设置失败: ${err.message}`, true, undefined, {
      taskId: '',
      persist: false,
    });
  }
}

export function initSettingsViewEvents() {
  document.querySelectorAll('.settings-nav-item').forEach(item => {
    item.addEventListener('click', () => {
      const target = item.getAttribute('data-settings-page');
      if (target) switchSettingsSection(target);
    });
  });

  enhancePasswordInputs();

  document.getElementById('cfg-asr-runtime')?.addEventListener('change', syncAsrRuntimePanel);
  document.getElementById('cfg-local-backend')?.addEventListener('change', syncLocalBackendRows);
  document.getElementById('cfg-mai-provider')?.addEventListener('change', syncMaiProviderRows);
  document.getElementById('cfg-demucs-chk')?.addEventListener('change', syncDemucsRow);
  document.getElementById('cfg-tts-method')?.addEventListener('change', syncTtsMethodPanel);
  document.getElementById('cfg-fish-mode')?.addEventListener('change', syncFishModeRows);
  document.getElementById('cfg-sf-fish-mode')?.addEventListener('change', syncSfFishModeRows);

  initAssStyleEvents({
    onBatchReset: resets => {
      for (const [id, val] of resets) {
        const el = document.getElementById(id);
        if (el) {
          el.value = val;
          const path = el.getAttribute('data-cfg');
          if (path) pendingConfigUpdates[path] = readCfgControlValue(el);
        }
      }
      scheduleConfigFlush();
    },
  });

  document.getElementById('btn-detect-mai-region')?.addEventListener('click', async e => {
    const btn = e.currentTarget;
    const apiKey = document.querySelector('[data-cfg="whisper.mai_api_key"]')?.value?.trim() || '';
    const regionInput = document.querySelector('[data-cfg="whisper.mai_region"]');
    if (!apiKey) {
      setAutosaveStatus('error', '请先输入 Azure Key');
      return;
    }
    const origLabel = btn.textContent;
    btn.disabled = true;
    btn.textContent = '探测中…';
    try {
      const res = await apiFetch('/api/mai/detect-region', {
        method: 'POST',
        body: JSON.stringify({ api_key: apiKey }),
      });
      if (res.region && regionInput) {
        regionInput.value = res.region;
        regionInput.dispatchEvent(new Event('input', { bubbles: true }));
        setAutosaveStatus('completed', `区域: ${res.region}`);
      }
    } catch (err) {
      reportActionError('MAI', err);
    } finally {
      btn.disabled = false;
      btn.textContent = origLabel;
    }
  });

  document.getElementById('btn-fetch-models')?.addEventListener('click', async () => {
    const baseUrl = document.querySelector('[data-cfg="api.base_url"]')?.value || '';
    const apiKey = document.querySelector('[data-cfg="api.key"]')?.value || '';
    const statusEl = document.getElementById('llm-check-status');
    if (statusEl) {
      statusEl.className = 'status-label running';
      statusEl.textContent = '获取模型中…';
    }
    try {
      const res = await apiFetch('/api/llm/models', {
        method: 'POST',
        body: JSON.stringify({ base_url: baseUrl, api_key: apiKey }),
      });
      const ok = Array.isArray(res.models) && res.models.length > 0 && !res.error;
      if (ok) {
        const dl = document.getElementById('llm-models-list');
        if (dl) dl.innerHTML = res.models.map(m => `<option value="${escapeAttr(m)}"></option>`).join('');
      }
      if (statusEl) {
        statusEl.textContent = ok ? `可用 ${res.models.length} 个模型` : res.error || res.message || '获取失败';
        statusEl.className = ok ? 'status-label completed' : 'status-label error';
      }
    } catch (err) {
      if (statusEl) {
        statusEl.textContent = `获取失败: ${err.message}`;
        statusEl.className = 'status-label error';
      }
    }
  });

  document.getElementById('btn-check-llm')?.addEventListener('click', async () => {
    const baseUrl = document.querySelector('[data-cfg="api.base_url"]')?.value || '';
    const apiKey = document.querySelector('[data-cfg="api.key"]')?.value || '';
    const model = document.querySelector('[data-cfg="api.model"]')?.value || '';
    const reasoningEffort = document.querySelector('[data-cfg="api.reasoning_effort"]')?.value || 'none';
    const statusEl = document.getElementById('llm-check-status');
    if (statusEl) {
      statusEl.className = 'status-label running';
      statusEl.textContent = '检测连通中…';
    }
    try {
      const res = await apiFetch('/api/llm/check', {
        method: 'POST',
        body: JSON.stringify({ base_url: baseUrl, api_key: apiKey, model, reasoning_effort: reasoningEffort }),
      });
      const ok = Boolean(res.valid ?? res.ok);
      if (statusEl) {
        statusEl.textContent = ok ? '接口连通正常' : `检测失败: ${res.error || res.message || '不可用'}`;
        statusEl.className = ok ? 'status-label completed' : 'status-label error';
      }
    } catch (err) {
      if (statusEl) {
        statusEl.textContent = `检测异常: ${err.message}`;
        statusEl.className = 'status-label error';
      }
    }
  });

  document.getElementById('btn-pick-cookies')?.addEventListener('click', async () => {
    if (!state.isBackendConnected) return;
    try {
      const res = await apiFetch('/api/dialogs/pick-files', {
        method: 'POST',
        body: JSON.stringify({ kind: 'cookies', multi: false }),
      });
      if (res.paths?.length) {
        const inp = document.getElementById('cfg-cookies-path');
        if (inp) {
          inp.value = res.paths[0];
          inp.dispatchEvent(new Event('input', { bubbles: true }));
        }
      }
    } catch (err) {
      reportActionError('FILE', err);
    }
  });

  document.getElementById('btn-pick-model-dir')?.addEventListener('click', async () => {
    if (!state.isBackendConnected) return;
    try {
      const res = await apiFetch('/api/dialogs/pick-files', {
        method: 'POST',
        body: JSON.stringify({ kind: 'directory', multi: false }),
      });
      if (res.paths?.length) {
        const inp = document.getElementById('cfg-model-dir');
        if (inp) {
          inp.value = res.paths[0];
          inp.dispatchEvent(new Event('input', { bubbles: true }));
        }
      }
    } catch (err) {
      reportActionError('FILE', err);
    }
  });

  bindCfgControls();
}
