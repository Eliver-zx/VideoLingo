import {
  state,
  REAL_TASKS,
  REAL_TASK_TERMS,
  PIPELINE_BATCH_NODES,
  TASK_DETAIL_CACHE,
} from './state.js';
import {
  apiFetch,
  debounce,
  escapeHtml,
  escapeAttr,
  setAutosaveStatus,
  autoResizeThemeTextarea,
} from './dom-utils.js';
import { appendLog, reportActionError } from './console-log.js';
import { loadBackendGlossaryDomains } from './glossary-view.js';

let termsAutosaveTimer = null;

export async function persistTaskTermsToBackend() {
  clearTimeout(termsAutosaveTimer);
  const task = REAL_TASKS.find(x => x.id === state.activeTaskId);
  const summaryVal = document.getElementById('terms-theme-input')?.value ?? '';
  const validTerms = REAL_TASK_TERMS.filter(
    t => String(t.src || '').trim() || String(t.tgt || '').trim()
  );
  if (state.isBackendConnected && task?.fromBackend) {
    const res = await apiFetch(`/api/tasks/${encodeURIComponent(state.activeTaskId)}/terms`, {
      method: 'PUT',
      body: JSON.stringify({
        summary: summaryVal,
        terms: validTerms.map(t => ({
          src: String(t.src || '').trim(),
          tgt: String(t.tgt || '').trim(),
          aliases: Array.isArray(t.aliases) ? t.aliases : [],
          note: String(t.note || ''),
          fix_source: Boolean(t.fix_source),
          source: t.source || 'task',
          promote_to_domain: Boolean(t.promote),
        })),
      }),
    });
    REAL_TASK_TERMS.forEach(t => {
      t.dirty = false;
    });
    state.isEditingTerms = false;
    setAutosaveStatus('completed', '术语已保存');
    if (res.promoted_to_domain > 0) {
      TASK_DETAIL_CACHE.clear();
      await loadBackendGlossaryDomains();
    } else if (state.activeTaskId) {
      TASK_DETAIL_CACHE.delete(state.activeTaskId);
    }
    return res;
  }
  state.isEditingTerms = false;
  if (state.activeTaskId) TASK_DETAIL_CACHE.delete(state.activeTaskId);
  setAutosaveStatus('completed', '术语已保存');
  return { saved: validTerms.length, promoted_to_domain: 0 };
}

export function scheduleTermsAutosave() {
  state.isEditingTerms = true;
  if (state.activeTaskId) TASK_DETAIL_CACHE.delete(state.activeTaskId);
  setAutosaveStatus('paused', '保存中…');
  clearTimeout(termsAutosaveTimer);
  termsAutosaveTimer = setTimeout(async () => {
    try {
      await persistTaskTermsToBackend();
    } catch (err) {
      reportActionError('TERMS', err, state.activeTaskId);
    }
  }, 550);
}

export function insertManualTaskTermAfter(afterIdx) {
  const entry = {
    src: '',
    tgt: '',
    aliases: [],
    note: '',
    fix_source: false,
    source: 'manual',
    in_domain_kb: false,
    promote: false,
  };
  const insertAt =
    afterIdx >= 0 && afterIdx < REAL_TASK_TERMS.length ? afterIdx + 1 : REAL_TASK_TERMS.length;
  REAL_TASK_TERMS.splice(insertAt, 0, entry);
  if (state.activeTaskId) TASK_DETAIL_CACHE.delete(state.activeTaskId);
  renderTaskTerms({ forceDom: true });
  const targetInput = document.querySelector(
    `#task-terms-tbody input[data-term-field="src"][data-idx="${insertAt}"]`
  );
  targetInput?.focus();
}

export function renderTaskTerms(options = {}) {
  const forceDom = Boolean(options && options.forceDom);
  const tbody = document.getElementById('task-terms-tbody');
  if (!tbody) return;
  const query = (document.getElementById('task-terms-search')?.value || '').trim().toLowerCase();
  const kbCount = REAL_TASK_TERMS.filter(t => t.in_domain_kb).length;
  const toggleKbBtn = document.getElementById('btn-toggle-kb-terms');
  if (toggleKbBtn) {
    toggleKbBtn.setAttribute('aria-pressed', String(state.showExistingKbTerms));
    toggleKbBtn.textContent = '显示全部';
  }

  const visiblePool = REAL_TASK_TERMS.map((t, idx) => ({ t, idx })).filter(
    ({ t }) => state.showExistingKbTerms || !t.in_domain_kb
  );

  const rows = visiblePool.filter(({ t }) => {
    if (!query) return true;
    return (
      t.src.toLowerCase().includes(query) ||
      t.tgt.toLowerCase().includes(query) ||
      t.note.toLowerCase().includes(query) ||
      t.aliases.some(a => a.toLowerCase().includes(query))
    );
  });

  const counterEl = document.getElementById('task-terms-counter');
  if (counterEl) {
    counterEl.textContent = query
      ? `${rows.length} / ${visiblePool.length} 条`
      : `${visiblePool.length} 条`;
  }

  const termsNode = PIPELINE_BATCH_NODES.find(n => n.id === 'terms-03');
  if (termsNode) {
    const activeTask = REAL_TASKS.find(x => x.id === state.activeTaskId);
    const isPausedTerms = activeTask?.rawState === 'paused' && activeTask?.stateLabel === '待审术语';
    termsNode.meta = isPausedTerms ? `${visiblePool.length} 待审` : `${REAL_TASK_TERMS.length} 条`;
    const metaEl = document.querySelector('#batch-tree-list [data-batch-id="terms-03"] .batch-meta');
    if (metaEl) metaEl.textContent = termsNode.meta;
  }

  if (!forceDom && state.activeBatchId !== 'terms-03') {
    state.termsDomDirty = true;
    return;
  }
  state.termsDomDirty = false;

  if (rows.length === 0) {
    const emptyHint =
      REAL_TASK_TERMS.length === 0
        ? '当前任务尚未提取术语，运行至第 03 阶段后将自动列出新专有名词'
        : !state.showExistingKbTerms && kbCount > 0 && visiblePool.length === 0
          ? `本次识别到的 ${kbCount} 条术语均已在领域词库中，点击上方「显示全部」可查看`
          : '未匹配到符合筛选条件的术语';
    tbody.innerHTML = `
      <tr>
        <td colspan="5" style="padding:32px 16px; text-align:center; color:var(--text-muted); font-size:var(--fs-xs);">
          <div>${escapeHtml(emptyHint)}</div>
          <div style="margin-block-start:10px;">
            <button type="button" class="xs" data-empty-add-term="1">
              <svg class="icon" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>
              <span>新增一行</span>
            </button>
          </div>
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = rows
    .map(({ t, idx }) => {
      const promoteTitle = t.promote ? '已收录进领域词库，点击取消' : '将该译名收录进领域词库';
      return `
      <tr>
        <td>
          <input type="text" class="cell-input mono" value="${escapeAttr(t.src)}" placeholder="原文术语" data-term-field="src" data-idx="${idx}">
        </td>
        <td>
          <input type="text" class="cell-input is-tgt" value="${escapeAttr(t.tgt)}" placeholder="输入目标译名" data-term-field="tgt" data-idx="${idx}">
        </td>
        <td>
          <div class="alias-inline-bar">
            ${t.aliases
              .map(
                (al, aIdx) => `
              <span class="alias-pill">
                <span>${escapeHtml(al)}</span>
                <button type="button" class="alias-del" data-del-alias="${idx}:${aIdx}" title="移除别名">×</button>
              </span>
            `
              )
              .join('')}
            <input type="text" class="alias-add-input mono" placeholder="+ 别名" data-add-alias="${idx}">
          </div>
        </td>
        <td>
          <input type="text" class="cell-input" value="${escapeAttr(t.note)}" placeholder="选填备注" data-term-field="note" data-idx="${idx}">
        </td>
        <td>
          <div class="term-actions">
            <button type="button" class="term-icon-btn ${t.promote ? 'is-promote-on' : ''}" data-term-promote="${idx}" aria-pressed="${Boolean(t.promote)}" title="${escapeAttr(promoteTitle)}" aria-label="${escapeAttr(promoteTitle)}">
              <svg class="icon" viewBox="0 0 24 24"><path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z"/><line x1="12" y1="7" x2="12" y2="13"/><line x1="9" y1="10" x2="15" y2="10"/></svg>
            </button>
            <button type="button" class="term-icon-btn is-del" data-del-term="${idx}" title="移除该条术语" aria-label="移除该条术语">
              <svg class="icon" viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
            <button type="button" class="term-icon-btn" data-insert-term="${idx}" title="在下方插入一行" aria-label="在下方插入一行">
              <svg class="icon" viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            </button>
          </div>
        </td>
      </tr>
    `;
    })
    .join('');
}

export function flushTermsDomIfNeeded() {
  if (state.termsDomDirty) {
    renderTaskTerms({ forceDom: true });
  }
}

async function handleToggleTermPromote(rIdx) {
  const row = REAL_TASK_TERMS[rIdx];
  if (!row) return;
  const srcVal = String(row.src || '').trim();
  const tgtVal = String(row.tgt || '').trim();
  if (!srcVal || !tgtVal) {
    setAutosaveStatus('error', '请先填写原文与译名');
    return;
  }
  const next = !row.promote;
  row.promote = next;
  const task = REAL_TASKS.find(x => x.id === state.activeTaskId);
  if (state.isBackendConnected && task?.fromBackend) {
    try {
      if (next) {
        await apiFetch(`/api/tasks/${encodeURIComponent(state.activeTaskId)}/terms/domain`, {
          method: 'POST',
          body: JSON.stringify({
            action: 'add',
            terms: [
              {
                src: srcVal,
                tgt: tgtVal,
                aliases: row.aliases,
                note: row.note,
                fix_source: Boolean(row.fix_source),
              },
            ],
          }),
        });
        row.dirty = false;
        setAutosaveStatus('completed', '已入库');
      } else {
        await apiFetch(`/api/tasks/${encodeURIComponent(state.activeTaskId)}/terms/domain`, {
          method: 'POST',
          body: JSON.stringify({ action: 'remove', srcs: [srcVal] }),
        });
        setAutosaveStatus('completed', '已移出词库');
      }
      TASK_DETAIL_CACHE.clear();
      renderTaskTerms();
      await loadBackendGlossaryDomains();
    } catch (err) {
      row.promote = !next;
      renderTaskTerms();
      reportActionError('TERMS', err, state.activeTaskId);
    }
  } else {
    renderTaskTerms();
  }
}

export function initTermsReviewEvents({ reloadTasks } = {}) {
  const tbody = document.getElementById('task-terms-tbody');
  tbody?.addEventListener('input', e => {
    const inp = e.target.closest('[data-term-field]');
    if (!inp) return;
    const rIdx = Number(inp.getAttribute('data-idx'));
    const field = inp.getAttribute('data-term-field');
    if (!REAL_TASK_TERMS[rIdx] || !field) return;
    REAL_TASK_TERMS[rIdx][field] = inp.value;
    REAL_TASK_TERMS[rIdx].dirty = true;
    scheduleTermsAutosave();
  });

  tbody?.addEventListener('keydown', e => {
    const inp = e.target.closest('[data-add-alias]');
    if (!inp || e.key !== 'Enter' || !inp.value.trim()) return;
    e.preventDefault();
    const rIdx = Number(inp.getAttribute('data-add-alias'));
    if (!REAL_TASK_TERMS[rIdx]) return;
    REAL_TASK_TERMS[rIdx].aliases.push(inp.value.trim());
    REAL_TASK_TERMS[rIdx].dirty = true;
    renderTaskTerms();
    scheduleTermsAutosave();
  });

  tbody?.addEventListener('click', e => {
    if (e.target.closest('[data-empty-add-term]')) {
      insertManualTaskTermAfter(REAL_TASK_TERMS.length - 1);
      return;
    }
    const delAliasBtn = e.target.closest('[data-del-alias]');
    if (delAliasBtn) {
      const [rIdx, aIdx] = delAliasBtn.getAttribute('data-del-alias').split(':').map(Number);
      if (!REAL_TASK_TERMS[rIdx]) return;
      REAL_TASK_TERMS[rIdx].aliases.splice(aIdx, 1);
      REAL_TASK_TERMS[rIdx].dirty = true;
      renderTaskTerms();
      scheduleTermsAutosave();
      return;
    }
    const promoteBtn = e.target.closest('[data-term-promote]');
    if (promoteBtn) {
      handleToggleTermPromote(Number(promoteBtn.getAttribute('data-term-promote')));
      return;
    }
    const delBtn = e.target.closest('[data-del-term]');
    if (delBtn) {
      const rIdx = Number(delBtn.getAttribute('data-del-term'));
      REAL_TASK_TERMS.splice(rIdx, 1);
      renderTaskTerms();
      scheduleTermsAutosave();
      return;
    }
    const insBtn = e.target.closest('[data-insert-term]');
    if (insBtn) {
      insertManualTaskTermAfter(Number(insBtn.getAttribute('data-insert-term')));
    }
  });

  document.getElementById('terms-theme-input')?.addEventListener('input', e => {
    autoResizeThemeTextarea(e.currentTarget);
    scheduleTermsAutosave();
  });

  document.getElementById('task-terms-search')?.addEventListener('input', debounce(renderTaskTerms, 120));

  document.getElementById('btn-toggle-kb-terms')?.addEventListener('click', () => {
    state.showExistingKbTerms = !state.showExistingKbTerms;
    renderTaskTerms();
  });

  document.getElementById('btn-rerun-term-extract')?.addEventListener('click', async () => {
    const task = REAL_TASKS.find(x => x.id === state.activeTaskId);
    if (!task) return;
    if (
      REAL_TASK_TERMS.length > 0 &&
      !window.confirm('重新提取将覆盖当前未入库的术语审校结果，确定继续吗？')
    ) {
      return;
    }
    if (state.isBackendConnected && task.fromBackend) {
      try {
        const desiredStage = task.desiredStage || 'subtitles';
        const runStage = ['transcribe', 'subtitles', 'dubbing', 'all'].includes(desiredStage)
          ? desiredStage
          : 'subtitles';
        await apiFetch(`/api/tasks/${encodeURIComponent(state.activeTaskId)}/invalidate`, {
          method: 'POST',
          body: JSON.stringify({ stage: 'summarize', autorun: true, run_stage: runStage }),
        });
        TASK_DETAIL_CACHE.delete(state.activeTaskId);
        appendLog('llm', 'TERMS', '已清空术语缓存并重新启动术语提取。', false);
        if (typeof reloadTasks === 'function') await reloadTasks();
      } catch (err) {
        reportActionError('TERMS', err, state.activeTaskId);
      }
    }
  });

  document.getElementById('btn-save-terms-only')?.addEventListener('click', async () => {
    try {
      await persistTaskTermsToBackend();
      appendLog('llm', 'TERMS', '已保存当前术语审校结果。', false);
    } catch (err) {
      reportActionError('TERMS', err, state.activeTaskId);
    }
  });

  document.getElementById('btn-save-terms-and-continue')?.addEventListener('click', async () => {
    const task = REAL_TASKS.find(x => x.id === state.activeTaskId);
    if (!task) return;
    try {
      await persistTaskTermsToBackend();
      if (state.isBackendConnected && task.fromBackend) {
        if (task.rawState === 'paused') {
          await apiFetch(`/api/tasks/${encodeURIComponent(state.activeTaskId)}/resume`, {
            method: 'POST',
          });
        } else if (task.rawState !== 'running' && task.rawState !== 'queued') {
          const desiredStage = task.desiredStage || 'subtitles';
          const dubbingEnabled = Boolean(
            task.dubbingEnabled ?? (desiredStage === 'all' || desiredStage === 'dubbing')
          );
          await apiFetch(`/api/tasks/${encodeURIComponent(state.activeTaskId)}/run`, {
            method: 'POST',
            body: JSON.stringify({ stage: desiredStage, dubbing: dubbingEnabled }),
          });
        }
        state.activeBatchId = 'all';
        if (typeof reloadTasks === 'function') await reloadTasks();
      }
      appendLog('llm', 'TERMS', '已确认术语译名并继续执行后续翻译。', false);
    } catch (err) {
      reportActionError('TERMS', err, state.activeTaskId);
    }
  });
}
