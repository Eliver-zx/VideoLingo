import { state, burnRuntimeState, burnLogs } from './state.js';
import { apiFetch, escapeHtml, escapeAttr, setAutosaveStatus } from './dom-utils.js';
import { appendLog, reportActionError } from './console-log.js';
import { switchView } from './router.js';
import { openVideoPreviewModal } from './video-preview.js';

let burnVideoFile = null;
let burnSubFile = null;
let burnSecSubFile = null;
let burnOutputManuallyEdited = false;
let burnSubPickerTarget = 'primary';

function deriveBurnDefaultOutput(rawVideoPath) {
  const cleaned = String(rawVideoPath || '').trim().replace(/^["']+|["']+$/g, '');
  if (!cleaned) return '';
  const slashIdx = Math.max(cleaned.lastIndexOf('/'), cleaned.lastIndexOf('\\'));
  const dir = slashIdx >= 0 ? cleaned.slice(0, slashIdx + 1) : '';
  const name = slashIdx >= 0 ? cleaned.slice(slashIdx + 1) : cleaned;
  const stem = name.replace(/\.[^.]+$/, '') || 'output';
  return `${dir}${stem}_sub.mp4`;
}

export function syncBurnPreviewHint() {
  const burnVideoInputEl = document.getElementById('burn-video-path');
  const burnSubInputEl = document.getElementById('burn-sub-path');
  const burnSecSubInputEl = document.getElementById('burn-sec-sub-path');
  const primaryVal = (burnSubInputEl?.value || '').trim().toLowerCase();
  const secVal = (burnSecSubInputEl?.value || '').trim();
  const hintEl = document.getElementById('burn-style-mode-hint');
  const burnSrcEl = document.getElementById('burn-preview-src');
  const clearSecBtn = document.getElementById('btn-burn-clear-sec-sub');

  if (clearSecBtn) clearSecBtn.classList.toggle('hidden', !secVal);
  document
    .getElementById('burn-video-drop')
    ?.classList.toggle('has-file', Boolean((burnVideoInputEl?.value || '').trim()));
  document.getElementById('burn-sub-drop')?.classList.toggle('has-file', Boolean(primaryVal));
  document.getElementById('burn-sec-sub-drop')?.classList.toggle('has-file', Boolean(secVal));

  if (hintEl) {
    hintEl.textContent = /\.(ass|ssa)(\s*\(.*\))?$/i.test(primaryVal)
      ? 'ASS / SSA 将保留文件内嵌特效与排版'
      : '导入 .srt 时自动应用全局字幕样式';
  }
  if (burnSrcEl) {
    burnSrcEl.style.display = secVal ? 'block' : 'none';
  }
}

function syncAutoBurnOutput() {
  const burnVideoInputEl = document.getElementById('burn-video-path');
  const burnOutInputEl = document.getElementById('burn-output-path');
  if (burnOutputManuallyEdited || !burnOutInputEl) return;
  if (burnVideoFile) {
    const stem = burnVideoFile.name.replace(/\.[^.]+$/, '') || 'video';
    burnOutInputEl.value = '';
    burnOutInputEl.placeholder = `自动输出至 ./output/${stem}_sub.mp4`;
  } else {
    burnOutInputEl.value = deriveBurnDefaultOutput(burnVideoInputEl?.value || '');
    burnOutInputEl.placeholder = '选择源视频后自动生成同目录 _sub.mp4';
  }
}

function updateBurnLogPreview() {
  const prevEl = document.getElementById('burn-console-preview');
  const last = burnLogs[burnLogs.length - 1];
  if (prevEl) {
    prevEl.textContent = last ? `${last.time} · [${last.label}] ${last.msg}` : '等待开始烧录…';
  }
}

export function renderBurnLogs() {
  const box = document.getElementById('burn-log-console');
  updateBurnLogPreview();
  if (!box) return;
  if (burnLogs.length === 0) {
    box.dataset.emptyPlaceholder = '1';
    box.innerHTML =
      '<div class="log-row"><span class="log-time">--:--:--</span><span class="log-tag">IDLE</span><span class="log-msg">选择源视频与字幕后点击「开始烧录」，实时输出压制进度…</span></div>';
  } else {
    delete box.dataset.emptyPlaceholder;
    box.innerHTML = burnLogs
      .map(
        item => `
        <div class="log-row ${item.warn ? 'warn' : ''}">
          <span class="log-time">${escapeHtml(item.time || '--:--:--')}</span>
          <span class="log-tag ${escapeAttr(item.tag || 'asr')}">${escapeHtml(item.label || 'BURN')}</span>
          <span class="log-msg">${escapeHtml(item.msg || '')}</span>
        </div>
      `
      )
      .join('');
  }
  box.scrollTop = box.scrollHeight;
}

export function appendBurnLogEntry(entry) {
  if (!entry) return;
  burnLogs.push(entry);
  if (burnLogs.length > 300) {
    burnLogs.splice(0, burnLogs.length - 300);
  }
  updateBurnLogPreview();
  const box = document.getElementById('burn-log-console');
  if (!box) return;
  if (box.dataset.emptyPlaceholder === '1') {
    box.innerHTML = '';
    delete box.dataset.emptyPlaceholder;
  }
  const row = document.createElement('div');
  row.className = entry.warn ? 'log-row warn' : 'log-row';
  row.innerHTML = `
    <span class="log-time">${escapeHtml(entry.time || '--:--:--')}</span>
    <span class="log-tag ${escapeAttr(entry.tag || 'asr')}">${escapeHtml(entry.label || 'BURN')}</span>
    <span class="log-msg">${escapeHtml(entry.msg || '')}</span>
  `;
  box.appendChild(row);
  while (box.childElementCount > 300 && box.firstElementChild) {
    box.removeChild(box.firstElementChild);
  }
  box.scrollTop = box.scrollHeight;
}

export function renderBurnControls() {
  const burnVideoInputEl = document.getElementById('burn-video-path');
  const burnSubInputEl = document.getElementById('burn-sub-path');
  const st = burnRuntimeState.state || 'idle';
  const isRunning = st === 'running';
  const hasInputs = Boolean(
    (burnVideoInputEl?.value || '').trim() && (burnSubInputEl?.value || '').trim()
  );
  const badge = document.getElementById('burn-status-badge');
  const strip = document.getElementById('burn-progress-strip');
  const stepEl = document.getElementById('burn-step-text');
  const pctEl = document.getElementById('burn-progress-pct');
  const fillEl = document.getElementById('burn-progress-fill');
  const startBtn = document.getElementById('btn-burn-start');
  const startLbl = document.getElementById('btn-burn-start-label');
  const stopBtn = document.getElementById('btn-burn-stop');
  const openBtn = document.getElementById('btn-burn-open-output');
  const prevBtn = document.getElementById('btn-burn-preview');

  if (badge) {
    if (isRunning) {
      badge.className = 'status-label running';
      badge.textContent = '正在压制';
    } else if (st === 'completed') {
      badge.className = 'status-label completed';
      badge.textContent = '烧录完成';
    } else if (st === 'error') {
      badge.className = 'status-label error';
      badge.textContent = '烧录失败';
    } else if (st === 'stopped') {
      badge.className = 'status-label paused';
      badge.textContent = '已停止';
    } else if (hasInputs) {
      badge.className = 'status-label paused';
      badge.textContent = '就绪待烧录';
    } else {
      badge.className = 'status-label';
      badge.textContent = '待选择文件';
    }
  }

  if (strip) strip.classList.toggle('hidden', st === 'idle');
  const pct = Math.max(0, Math.min(100, Math.round((burnRuntimeState.progress || 0) * 100)));
  if (stepEl) {
    stepEl.textContent = burnRuntimeState.sub_step || (isRunning ? '正在压制字幕…' : '准备就绪');
  }
  if (pctEl) pctEl.textContent = `${pct}%`;
  if (fillEl) {
    fillEl.style.inlineSize = `${pct}%`;
    fillEl.classList.toggle('burn-done', st === 'completed');
    fillEl.classList.toggle('error', st === 'error');
  }

  if (startBtn) startBtn.disabled = isRunning;
  if (startLbl) startLbl.textContent = isRunning ? '正在压制…' : '开始烧录';
  if (stopBtn) {
    stopBtn.disabled = !isRunning;
    stopBtn.classList.toggle('hidden', !isRunning);
  }

  const canOpen = Boolean(
    burnRuntimeState.completed_output_path || (st !== 'idle' && burnRuntimeState.output_path)
  );
  if (openBtn) {
    openBtn.disabled = !canOpen;
    openBtn.classList.toggle('hidden', !canOpen);
  }

  const canPreview = Boolean(burnRuntimeState.completed_output_path);
  if (prevBtn) {
    prevBtn.disabled = !canPreview;
    prevBtn.classList.toggle('hidden', !canPreview);
  }
  syncBurnPreviewHint();
}

function setBurnBrowserFile(slot, file) {
  if (!file) return;
  const burnVideoInputEl = document.getElementById('burn-video-path');
  const burnSubInputEl = document.getElementById('burn-sub-path');
  const burnSecSubInputEl = document.getElementById('burn-sec-sub-path');
  if (slot === 'video') {
    burnVideoFile = file;
    const lbl = `${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
    if (burnVideoInputEl) {
      burnVideoInputEl.value = lbl;
      burnVideoInputEl.dataset.droppedLabel = lbl;
    }
    syncAutoBurnOutput();
  } else if (slot === 'sec') {
    burnSecSubFile = file;
    const lbl = `${file.name} (${Math.max(1, Math.round(file.size / 1024))} KB)`;
    if (burnSecSubInputEl) {
      burnSecSubInputEl.value = lbl;
      burnSecSubInputEl.dataset.droppedLabel = lbl;
    }
  } else {
    burnSubFile = file;
    const lbl = `${file.name} (${Math.max(1, Math.round(file.size / 1024))} KB)`;
    if (burnSubInputEl) {
      burnSubInputEl.value = lbl;
      burnSubInputEl.dataset.droppedLabel = lbl;
    }
  }
  renderBurnControls();
}

export function handleBurnDroppedFiles(fileList, explicitSlot = '') {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  if (explicitSlot === 'video') return setBurnBrowserFile('video', files[0]);
  if (explicitSlot === 'sub') return setBurnBrowserFile('primary', files[0]);
  if (explicitSlot === 'sec') return setBurnBrowserFile('sec', files[0]);
  const subs = files.filter(f => /\.(srt|ass|ssa)$/i.test(f.name));
  const vids = files.filter(f => !/\.(srt|ass|ssa)$/i.test(f.name));
  if (vids[0]) setBurnBrowserFile('video', vids[0]);
  if (subs[0]) setBurnBrowserFile('primary', subs[0]);
  if (subs[1]) setBurnBrowserFile('sec', subs[1]);
}

export async function loadBackendBurnStatus() {
  try {
    const snap = await apiFetch('/api/burn/status');
    state.isBackendConnected = true;
    Object.assign(burnRuntimeState, snap);
    if (Array.isArray(snap.logs)) {
      burnLogs.splice(0, burnLogs.length, ...snap.logs);
    }
    renderBurnControls();
    renderBurnLogs();
  } catch (err) {
    appendLog('warn', 'BURN', `获取烧录状态失败: ${err.message}`, true, undefined, {
      taskId: '',
      persist: false,
    });
  }
}

export function initBurnViewEvents({ onJumpAssStyle } = {}) {
  const burnVideoInputEl = document.getElementById('burn-video-path');
  const burnSubInputEl = document.getElementById('burn-sub-path');
  const burnSecSubInputEl = document.getElementById('burn-sec-sub-path');
  const burnOutInputEl = document.getElementById('burn-output-path');

  const burnBrowserVideoPicker = document.createElement('input');
  burnBrowserVideoPicker.type = 'file';
  burnBrowserVideoPicker.accept = 'video/*,.mp4,.mkv,.mov,.webm,.avi';
  burnBrowserVideoPicker.style.display = 'none';
  document.body.appendChild(burnBrowserVideoPicker);

  const burnBrowserSubPicker = document.createElement('input');
  burnBrowserSubPicker.type = 'file';
  burnBrowserSubPicker.accept = '.srt,.ass,.ssa';
  burnBrowserSubPicker.style.display = 'none';
  document.body.appendChild(burnBrowserSubPicker);

  burnVideoInputEl?.addEventListener('input', () => {
    if (burnVideoFile && burnVideoInputEl.value.trim() !== burnVideoInputEl.dataset.droppedLabel) {
      burnVideoFile = null;
      delete burnVideoInputEl.dataset.droppedLabel;
    }
    syncAutoBurnOutput();
    renderBurnControls();
  });

  burnSubInputEl?.addEventListener('input', () => {
    if (burnSubFile && burnSubInputEl.value.trim() !== burnSubInputEl.dataset.droppedLabel) {
      burnSubFile = null;
      delete burnSubInputEl.dataset.droppedLabel;
    }
    renderBurnControls();
  });

  burnSecSubInputEl?.addEventListener('input', () => {
    if (
      burnSecSubFile &&
      burnSecSubInputEl.value.trim() !== burnSecSubInputEl.dataset.droppedLabel
    ) {
      burnSecSubFile = null;
      delete burnSecSubInputEl.dataset.droppedLabel;
    }
    renderBurnControls();
  });

  burnOutInputEl?.addEventListener('input', () => {
    burnOutputManuallyEdited = Boolean(burnOutInputEl.value.trim());
  });

  document.getElementById('btn-burn-reset-output')?.addEventListener('click', () => {
    burnOutputManuallyEdited = false;
    syncAutoBurnOutput();
  });

  document.getElementById('btn-burn-clear-sec-sub')?.addEventListener('click', () => {
    burnSecSubFile = null;
    if (burnSecSubInputEl) {
      burnSecSubInputEl.value = '';
      delete burnSecSubInputEl.dataset.droppedLabel;
    }
    renderBurnControls();
  });

  document.getElementById('burn-gpu-chk')?.addEventListener('change', () => {
    state.burnGpuTouched = true;
  });

  document.getElementById('btn-burn-jump-style')?.addEventListener('click', () => {
    switchView('settings');
    if (typeof onJumpAssStyle === 'function') onJumpAssStyle('cfg-sec-ass');
  });

  const burnSurface = document.getElementById('burn-drop-surface');
  if (burnSurface) {
    ['dragenter', 'dragover'].forEach(ev => {
      burnSurface.addEventListener(ev, e => {
        if (e.dataTransfer?.types?.includes('Files')) {
          e.preventDefault();
          burnSurface.classList.add('dragover');
        }
      });
    });
    burnSurface.addEventListener('dragleave', e => {
      if (!burnSurface.contains(e.relatedTarget)) burnSurface.classList.remove('dragover');
    });
    burnSurface.addEventListener('drop', e => {
      if (!e.dataTransfer?.files?.length) return;
      e.preventDefault();
      e.stopPropagation();
      burnSurface.classList.remove('dragover');
      if (e.target.closest('#burn-sec-sub-drop')) handleBurnDroppedFiles(e.dataTransfer.files, 'sec');
      else if (e.target.closest('#burn-sub-drop')) handleBurnDroppedFiles(e.dataTransfer.files, 'sub');
      else if (e.target.closest('#burn-video-drop')) handleBurnDroppedFiles(e.dataTransfer.files, 'video');
      else handleBurnDroppedFiles(e.dataTransfer.files);
    });
  }

  burnBrowserVideoPicker.addEventListener('change', () => {
    if (burnBrowserVideoPicker.files?.length) setBurnBrowserFile('video', burnBrowserVideoPicker.files[0]);
    burnBrowserVideoPicker.value = '';
  });
  burnBrowserSubPicker.addEventListener('change', () => {
    if (burnBrowserSubPicker.files?.length) setBurnBrowserFile(burnSubPickerTarget, burnBrowserSubPicker.files[0]);
    burnBrowserSubPicker.value = '';
  });

  document.getElementById('btn-burn-pick-video')?.addEventListener('click', async () => {
    if (state.isBackendConnected) {
      try {
        const res = await apiFetch('/api/dialogs/pick-files', {
          method: 'POST',
          body: JSON.stringify({ kind: 'video', multi: false }),
        });
        if (res.paths?.length) {
          burnVideoFile = null;
          if (burnVideoInputEl) {
            delete burnVideoInputEl.dataset.droppedLabel;
            burnVideoInputEl.value = res.paths[0];
          }
          syncAutoBurnOutput();
          renderBurnControls();
          return;
        }
        if (Array.isArray(res.paths)) return;
      } catch (err) {
        appendBurnLogEntry({
          time: new Date().toTimeString().slice(0, 8),
          tag: 'warn',
          label: 'FILE',
          msg: err.message,
          warn: true,
        });
      }
    }
    burnBrowserVideoPicker.click();
  });

  const pickBurnSubtitle = async slot => {
    if (state.isBackendConnected) {
      try {
        const res = await apiFetch('/api/dialogs/pick-files', {
          method: 'POST',
          body: JSON.stringify({ kind: 'subtitle', multi: false }),
        });
        if (res.paths?.length) {
          if (slot === 'sec') {
            burnSecSubFile = null;
            if (burnSecSubInputEl) {
              delete burnSecSubInputEl.dataset.droppedLabel;
              burnSecSubInputEl.value = res.paths[0];
            }
          } else {
            burnSubFile = null;
            if (burnSubInputEl) {
              delete burnSubInputEl.dataset.droppedLabel;
              burnSubInputEl.value = res.paths[0];
            }
          }
          renderBurnControls();
          return;
        }
        if (Array.isArray(res.paths)) return;
      } catch (err) {
        appendBurnLogEntry({
          time: new Date().toTimeString().slice(0, 8),
          tag: 'warn',
          label: 'FILE',
          msg: err.message,
          warn: true,
        });
      }
    }
    burnSubPickerTarget = slot;
    burnBrowserSubPicker.click();
  };

  document.getElementById('btn-burn-pick-sub')?.addEventListener('click', () => pickBurnSubtitle('primary'));
  document.getElementById('btn-burn-pick-sec-sub')?.addEventListener('click', () => pickBurnSubtitle('sec'));

  document.getElementById('btn-burn-pick-output')?.addEventListener('click', async () => {
    if (!state.isBackendConnected) return;
    try {
      const curOut = (burnOutInputEl?.value || '').trim();
      const defName = curOut ? curOut.split(/[\\/]/).pop() : 'output_sub.mp4';
      const res = await apiFetch('/api/dialogs/pick-files', {
        method: 'POST',
        body: JSON.stringify({ kind: 'save_video', multi: false, default_name: defName || 'output_sub.mp4' }),
      });
      if (res.paths?.length && burnOutInputEl) {
        burnOutputManuallyEdited = true;
        burnOutInputEl.value = res.paths[0];
      }
    } catch (err) {
      appendBurnLogEntry({
        time: new Date().toTimeString().slice(0, 8),
        tag: 'warn',
        label: 'FILE',
        msg: err.message,
        warn: true,
      });
    }
  });

  document.getElementById('btn-burn-start')?.addEventListener('click', async () => {
    const vidStr = (burnVideoInputEl?.value || '').trim();
    const subStr = (burnSubInputEl?.value || '').trim();
    const secStr = (burnSecSubInputEl?.value || '').trim();
    const outStr = (burnOutInputEl?.value || '').trim();
    const useGpu = Boolean(document.getElementById('burn-gpu-chk')?.checked);

    if (!vidStr && !burnVideoFile) {
      burnVideoInputEl?.focus();
      setAutosaveStatus('error', '请选择源视频');
      return;
    }
    if (!subStr && !burnSubFile) {
      burnSubInputEl?.focus();
      setAutosaveStatus('error', '请选择主字幕');
      return;
    }
    if (!state.isBackendConnected) return;

    try {
      burnLogs.splice(0, burnLogs.length);
      let snap;
      if (burnVideoFile || burnSubFile || burnSecSubFile) {
        const form = new FormData();
        if (burnVideoFile) form.append('video', burnVideoFile);
        else form.append('video_path', vidStr);
        if (burnSubFile) form.append('subtitle', burnSubFile);
        else form.append('subtitle_path', subStr);
        if (burnSecSubFile) form.append('secondary_subtitle', burnSecSubFile);
        else if (secStr) form.append('secondary_subtitle_path', secStr);
        if (outStr) form.append('output_path', outStr);
        form.append('ffmpeg_gpu', String(useGpu));
        snap = await apiFetch('/api/burn/upload', { method: 'POST', body: form });
        burnVideoFile = null;
        burnSubFile = null;
        burnSecSubFile = null;
        if (snap.video_path && burnVideoInputEl) {
          delete burnVideoInputEl.dataset.droppedLabel;
          burnVideoInputEl.value = snap.video_path;
        }
        if (snap.subtitle_path && burnSubInputEl) {
          delete burnSubInputEl.dataset.droppedLabel;
          burnSubInputEl.value = snap.subtitle_path;
        }
        if (snap.secondary_subtitle_path && burnSecSubInputEl) {
          delete burnSecSubInputEl.dataset.droppedLabel;
          burnSecSubInputEl.value = snap.secondary_subtitle_path;
        }
        if (!burnOutputManuallyEdited && snap.output_path && burnOutInputEl) {
          burnOutInputEl.value = snap.output_path;
        }
      } else {
        snap = await apiFetch('/api/burn/start', {
          method: 'POST',
          body: JSON.stringify({
            video_path: vidStr,
            subtitle_path: subStr,
            secondary_subtitle_path: secStr || null,
            output_path: outStr || null,
            ffmpeg_gpu: useGpu,
          }),
        });
      }
      Object.assign(burnRuntimeState, snap);
      if (Array.isArray(snap.logs)) {
        burnLogs.splice(0, burnLogs.length, ...snap.logs);
      }
      renderBurnControls();
      renderBurnLogs();
    } catch (err) {
      burnRuntimeState.state = 'error';
      burnRuntimeState.sub_step = '启动失败';
      burnLogs.push({
        time: new Date().toTimeString().slice(0, 8),
        tag: 'warn',
        label: 'ERROR',
        msg: err.message,
        warn: true,
      });
      renderBurnControls();
      renderBurnLogs();
      setAutosaveStatus('error', '烧录启动失败');
    }
  });

  document.getElementById('btn-burn-stop')?.addEventListener('click', async () => {
    if (!state.isBackendConnected) return;
    try {
      const snap = await apiFetch('/api/burn/stop', { method: 'POST' });
      Object.assign(burnRuntimeState, snap);
      renderBurnControls();
    } catch (err) {
      reportActionError('BURN', err);
    }
  });

  document.getElementById('btn-burn-open-output')?.addEventListener('click', async () => {
    if (!state.isBackendConnected) return;
    try {
      await apiFetch('/api/burn/open-output', { method: 'POST' });
    } catch (err) {
      reportActionError('BURN', err);
    }
  });

  document.getElementById('btn-burn-preview')?.addEventListener('click', () => {
    if (!burnRuntimeState.completed_output_path) return;
    const outName = burnRuntimeState.completed_output_path.split(/[\\/]/).pop() || '烧录成片预览';
    openVideoPreviewModal({
      title: outName,
      sources: [{ label: '烧录成片', url: `/api/burn/preview?t=${Date.now()}` }],
    });
  });

  document.getElementById('btn-burn-copy-log')?.addEventListener('click', e => {
    const btn = e.currentTarget;
    const text = burnLogs.map(l => `${l.time} [${l.label}] ${l.msg}`).join('\n');
    navigator.clipboard?.writeText(text).catch(() => {});
    btn.textContent = '已复制';
    setTimeout(() => {
      btn.textContent = '复制';
    }, 1200);
  });
}
