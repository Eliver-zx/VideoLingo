import {
  state,
  REAL_TASKS,
  REAL_SYSTEM_LOGS,
  TASK_LOGS_BY_ID,
  PIPELINE_BATCH_NODES,
} from './state.js';
import { apiFetch, escapeHtml, escapeAttr, setAutosaveStatus } from './dom-utils.js';

const pendingLogWritesByTask = new Map();

export function getTaskLogBucket(taskId) {
  const key = String(taskId || '');
  let bucket = TASK_LOGS_BY_ID.get(key);
  if (!bucket) {
    bucket = [];
    TASK_LOGS_BY_ID.set(key, bucket);
  }
  return bucket;
}

function updateLogWidgets() {
  let warnCount = 0;
  for (let i = 0; i < REAL_SYSTEM_LOGS.length; i += 1) {
    if (REAL_SYSTEM_LOGS[i].warn) warnCount += 1;
  }
  const warnBtn = document.getElementById('btn-filter-warn-log');
  if (warnBtn) {
    warnBtn.textContent = state.warnOnlyLog ? '全部日志' : `仅看告警 (${warnCount})`;
  }
  const logNode = PIPELINE_BATCH_NODES.find(n => n.id === 'runtime-log');
  if (logNode) {
    logNode.meta = `${REAL_SYSTEM_LOGS.length} 条`;
    const logNodeEl = document.querySelector('#batch-tree-list [data-batch-id="runtime-log"] .batch-meta');
    if (logNodeEl) logNodeEl.textContent = logNode.meta;
  }
  const previewEl = document.getElementById('console-latest-preview');
  const lastItem = REAL_SYSTEM_LOGS[REAL_SYSTEM_LOGS.length - 1];
  if (previewEl) {
    previewEl.textContent = lastItem
      ? `${lastItem.time} · [${lastItem.label}] ${lastItem.msg}`
      : '等待任务运行…';
  }
}

function buildLogRowElement(item) {
  const row = document.createElement('div');
  row.className = item.warn ? 'log-row warn' : 'log-row';
  row.innerHTML = `
    <span class="log-time">${escapeHtml(item.time)}</span>
    <span class="log-tag ${escapeAttr(item.tag)}">${escapeHtml(item.label)}</span>
    <span class="log-msg">${escapeHtml(item.msg)}</span>
  `;
  return row;
}

export function renderSystemLogs(options = {}) {
  const forceDom = Boolean(options.forceDom);
  updateLogWidgets();
  const box = document.getElementById('global-log-console');
  if (!box) return;
  if (!forceDom && state.activeBatchId !== 'runtime-log') {
    state.logsDomDirty = true;
    return;
  }
  state.logsDomDirty = false;
  const list = state.warnOnlyLog ? REAL_SYSTEM_LOGS.filter(x => x.warn) : REAL_SYSTEM_LOGS;
  if (list.length === 0) {
    box.dataset.emptyPlaceholder = '1';
    box.innerHTML = `<div class="log-row"><span class="log-time">--:--:--</span><span class="log-tag">IDLE</span><span class="log-msg">${state.warnOnlyLog ? '暂无告警记录' : '等待任务运行输出日志…'}</span></div>`;
  } else {
    delete box.dataset.emptyPlaceholder;
    box.innerHTML = list
      .map(
        item => `
        <div class="log-row ${item.warn ? 'warn' : ''}">
          <span class="log-time">${escapeHtml(item.time)}</span>
          <span class="log-tag ${escapeAttr(item.tag)}">${escapeHtml(item.label)}</span>
          <span class="log-msg">${escapeHtml(item.msg)}</span>
        </div>
      `
      )
      .join('');
  }
  box.scrollTop = box.scrollHeight;
}

export function flushLogsDomIfNeeded() {
  if (state.logsDomDirty) {
    renderSystemLogs({ forceDom: true });
    return;
  }
  const box = document.getElementById('global-log-console');
  if (box) box.scrollTop = box.scrollHeight;
}

export function syncActiveTaskLogsView() {
  const current = state.activeTaskId ? getTaskLogBucket(state.activeTaskId) : [];
  REAL_SYSTEM_LOGS.splice(0, REAL_SYSTEM_LOGS.length, ...current);
  renderSystemLogs();
}

function persistClientTaskLog(taskId, entry) {
  const prev = pendingLogWritesByTask.get(taskId) || Promise.resolve();
  const next = prev
    .catch(() => {})
    .then(() =>
      apiFetch(`/api/tasks/${encodeURIComponent(taskId)}/logs`, {
        method: 'POST',
        body: JSON.stringify(entry),
      })
    )
    .catch(() => {});
  pendingLogWritesByTask.set(taskId, next);
  return next;
}

export function setTaskLogsFromServer(taskId, logs) {
  if (!taskId) return;
  const normalized = Array.isArray(logs)
    ? logs
        .filter(item => item && item.msg)
        .map(item => ({
          time: item.time || '--:--:--',
          tag: item.tag || 'llm',
          label: item.label || 'RUN',
          msg: item.msg,
          warn: Boolean(item.warn),
        }))
        .slice(-300)
    : [];
  TASK_LOGS_BY_ID.set(String(taskId), normalized);
  if (String(taskId) === String(state.activeTaskId || '')) {
    syncActiveTaskLogsView();
  }
}

export function appendLog(tag, label, msg, warn, time, options = {}) {
  const hasExplicitTask = Object.prototype.hasOwnProperty.call(options, 'taskId');
  const targetTaskId = hasExplicitTask ? String(options.taskId || '') : String(state.activeTaskId || '');
  if (!targetTaskId) return;
  const fromServer = Boolean(options.fromServer);
  const persist = options.persist !== undefined ? Boolean(options.persist) : !fromServer;
  const normalizedTag = tag === 'asr' || tag === 'warn' ? tag : 'llm';
  const entry = {
    time: time || new Date().toTimeString().slice(0, 8),
    tag: warn ? 'warn' : normalizedTag,
    label: String(label || 'RUN').toUpperCase().slice(0, 8),
    msg,
    warn: Boolean(warn),
  };
  const bucket = getTaskLogBucket(targetTaskId);
  bucket.push(entry);
  if (bucket.length > 300) {
    bucket.splice(0, bucket.length - 300);
  }
  if (targetTaskId === String(state.activeTaskId || '')) {
    REAL_SYSTEM_LOGS.push(entry);
    if (REAL_SYSTEM_LOGS.length > 300) {
      REAL_SYSTEM_LOGS.splice(0, REAL_SYSTEM_LOGS.length - 300);
    }
    updateLogWidgets();
    const box = document.getElementById('global-log-console');
    if (state.activeBatchId !== 'runtime-log' || !box) {
      state.logsDomDirty = true;
    } else if (state.logsDomDirty) {
      renderSystemLogs({ forceDom: true });
    } else if (!state.warnOnlyLog || entry.warn) {
      if (box.dataset.emptyPlaceholder === '1') {
        box.innerHTML = '';
        delete box.dataset.emptyPlaceholder;
      }
      box.appendChild(buildLogRowElement(entry));
      while (box.childElementCount > 300 && box.firstElementChild) {
        box.removeChild(box.firstElementChild);
      }
      box.scrollTop = box.scrollHeight;
    }
  }
  if (persist && state.isBackendConnected) {
    const task = REAL_TASKS.find(x => x.id === targetTaskId);
    if (!task || task.fromBackend) {
      persistClientTaskLog(targetTaskId, entry);
    }
  }
}

export function hydrateSystemLogs(logs) {
  if (!Array.isArray(logs) || logs.length === 0) return;
  let activeTouched = false;
  logs.forEach(item => {
    if (!item || !item.msg || !item.task_id) return;
    const tid = String(item.task_id);
    const bucket = getTaskLogBucket(tid);
    const t = item.time || '--:--:--';
    const lbl = item.label || 'RUN';
    const dup = bucket.some(x => x.time === t && x.label === lbl && x.msg === item.msg);
    if (dup) return;
    bucket.push({
      time: t,
      tag: item.tag || 'llm',
      label: lbl,
      msg: item.msg,
      warn: Boolean(item.warn),
    });
    if (bucket.length > 300) {
      bucket.splice(0, bucket.length - 300);
    }
    if (tid === String(state.activeTaskId || '')) {
      activeTouched = true;
    }
  });
  if (activeTouched) {
    syncActiveTaskLogsView();
  }
}

export function reportActionError(scope, err, taskId = '') {
  const msg = err?.message || String(err || '操作失败');
  setAutosaveStatus('error', '操作失败');
  appendLog('warn', scope, msg, true, undefined, { taskId });
}

export function initConsoleLogEvents({ onToggleConsole } = {}) {
  document.getElementById('btn-filter-warn-log')?.addEventListener('click', () => {
    state.warnOnlyLog = !state.warnOnlyLog;
    renderSystemLogs({ forceDom: true });
  });

  document.getElementById('btn-copy-log')?.addEventListener('click', e => {
    const btn = e.currentTarget;
    const text = REAL_SYSTEM_LOGS.map(l => `${l.time} [${l.label}] ${l.msg}`).join('\n');
    navigator.clipboard?.writeText(text).catch(() => {});
    btn.textContent = '已复制';
    setTimeout(() => {
      btn.textContent = '复制';
    }, 1200);
  });

  document.getElementById('btn-toggle-console')?.addEventListener('click', () => {
    if (typeof onToggleConsole === 'function') {
      onToggleConsole();
    }
  });
}
