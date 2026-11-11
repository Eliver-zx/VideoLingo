import {
  state,
  REAL_TASKS,
  REAL_TASK_TERMS,
  DEFAULT_PIPELINE_BATCH_NODES,
  PIPELINE_BATCH_NODES,
  STREAM_TABLE_ROWS,
  ASR_SEGMENT_ROWS,
  SUBTITLE_CUE_ROWS,
  TASK_LOGS_BY_ID,
  TASK_DETAIL_CACHE,
} from './state.js';
import {
  apiFetch,
  escapeHtml,
  escapeAttr,
  setAutosaveStatus,
  autoResizeThemeTextarea,
  isUserEditingDom,
} from './dom-utils.js';
import {
  syncActiveTaskLogsView,
  setTaskLogsFromServer,
  appendLog,
  reportActionError,
} from './console-log.js';
import { syncDetailPane } from './router.js';
import {
  cancelPendingStreamChunk,
  mapAsrSegmentsToRows,
  rebuildStreamRowsFromUtterances,
  renderStreamTable,
  syncBatchNodesFromLoadedRows,
} from './stream-table.js';
import { persistTaskTermsToBackend, renderTaskTerms } from './terms-review.js';
import {
  pillClassForTask,
  mapBackendTaskToCard,
  renderBatchTree,
  syncStartResumeButton,
  syncPipelineRibbon,
} from './pipeline-ribbon.js';

export { mapBackendTaskToCard, renderBatchTree, syncStartResumeButton, syncPipelineRibbon };

let activeDetailController = null;
let inflightTasksReload = null;
let pendingTasksReloadOptions = null;
let inflightAsrTaskId = '';
let inflightAsrPromise = null;

export function computeTaskSnapshotSig(raw) {
  if (!raw) return '';
  const ex = raw.execution || {};
  return JSON.stringify([
    raw.updated_at || '',
    ex.state || 'idle',
    ex.current_step || '',
    ex.pause_reason || '',
    raw.config_override?.glossary_domain ?? null,
    raw.stages_done || {},
    raw.files || [],
  ]);
}

function cloneTermsList(terms) {
  return (Array.isArray(terms) ? terms : []).map(t => ({
    ...t,
    aliases: Array.isArray(t.aliases) ? [...t.aliases] : [],
  }));
}

function normalizeBackendTerms(rawTerms) {
  return (Array.isArray(rawTerms) ? rawTerms : []).map(item => ({
    src: item.src || '',
    tgt: item.tgt || '',
    aliases: Array.isArray(item.aliases) ? item.aliases : [],
    note: item.note || '',
    fix_source: Boolean(item.fix_source),
    source: item.source || 'task',
    in_domain_kb: Boolean(item.in_domain_kb),
    promote: Boolean(item.promoted ?? item.promote_to_domain ?? item.promote ?? false),
  }));
}

export function upsertTaskDetailCache(taskId, patch) {
  if (!taskId) return null;
  const prev = TASK_DETAIL_CACHE.get(taskId) || {
    sig: '',
    uttsLoaded: false,
    termsLoaded: false,
    logsLoaded: false,
    hasAsr: false,
    totalWords: 0,
    stagesDone: {},
    streamRows: [],
    cueRows: [],
    asrRows: [],
    terms: [],
    termsSummary: '',
  };
  const next = { ...prev, ...patch };
  TASK_DETAIL_CACHE.set(taskId, next);
  return next;
}

function resetDetailScrollPositions() {
  const sc = document.querySelector('#subtitle-detail-pane .table-scroll');
  if (sc) sc.scrollTop = 0;
  const tc = document.querySelector('#terms-review-panel .table-scroll');
  if (tc) tc.scrollTop = 0;
}

export function renderTaskRail() {
  const box = document.getElementById('task-rail-list');
  const countEl = document.getElementById('nav-task-count');
  if (countEl) {
    countEl.textContent = String(REAL_TASKS.length);
    countEl.dataset.zero = String(REAL_TASKS.length === 0);
  }
  if (!box) return;
  if (REAL_TASKS.length === 0) {
    box.innerHTML = `
      <div class="rail-empty">
        <div>暂无任务</div>
        <div style="margin-block-start:4px;">点击上方「新建」导入视频或字幕</div>
      </div>
    `;
    return;
  }
  box.innerHTML = REAL_TASKS.map(t => {
    const active = t.id === state.activeTaskId ? 'active' : '';
    const pillClass = pillClassForTask(t);
    const cardTip = t.errorDetail ? `${t.title}\n异常详情: ${t.errorDetail}` : t.title;
    return `
      <div class="task-item ${active}" data-task-id="${escapeAttr(t.id)}" title="${escapeAttr(cardTip)}">
        <div class="task-item-top">
          <span class="task-item-name">${escapeHtml(t.title)}</span>
          <span class="status-label ${pillClass}">${escapeHtml(t.stateLabel)}</span>
        </div>
        <div class="task-item-meta">
          <span class="task-item-domain">${escapeHtml(t.domain || '通用')}</span>
          <span class="task-item-summary">${escapeHtml(t.stepSummary)}</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill ${pillClass}" style="inline-size:${t.progress}%;"></div>
        </div>
      </div>
    `;
  }).join('');
}

export function patchTaskRailCard(t) {
  const box = document.getElementById('task-rail-list');
  if (!box || !t) return;
  let cardEl = null;
  const items = box.querySelectorAll('.task-item');
  for (let i = 0; i < items.length; i += 1) {
    if (items[i].getAttribute('data-task-id') === t.id) {
      cardEl = items[i];
      break;
    }
  }
  if (!cardEl) {
    renderTaskRail();
    return;
  }
  const pillClass = pillClassForTask(t);
  cardEl.title = t.errorDetail ? `${t.title}\n异常详情: ${t.errorDetail}` : t.title;
  cardEl.classList.toggle('active', t.id === state.activeTaskId);
  const nameEl = cardEl.querySelector('.task-item-name');
  if (nameEl && nameEl.textContent !== t.title) nameEl.textContent = t.title;
  const badgeEl = cardEl.querySelector('.status-label');
  if (badgeEl) {
    badgeEl.className = `status-label ${pillClass}`;
    badgeEl.textContent = t.stateLabel;
  }
  const domEl = cardEl.querySelector('.task-item-domain');
  if (domEl) domEl.textContent = t.domain || '通用';
  const sumEl = cardEl.querySelector('.task-item-summary');
  if (sumEl) sumEl.textContent = t.stepSummary;
  const fillEl = cardEl.querySelector('.progress-fill');
  if (fillEl) {
    fillEl.className = `progress-fill ${pillClass}`;
    fillEl.style.inlineSize = `${t.progress}%`;
  }
}

export function resetWorkbenchToEmpty() {
  cancelPendingStreamChunk();
  state.isEditingTerms = false;
  state.loadedAsrTaskId = '';
  state.hydratedTaskId = '';
  setAutosaveStatus('idle', '');
  document.getElementById('detail-task-title').textContent = '未选择任务';
  const detailBadge = document.getElementById('detail-task-badge');
  if (detailBadge) {
    detailBadge.className = 'status-label';
    detailBadge.textContent = '等待导入';
  }
  const domPill = document.getElementById('detail-domain-pill');
  if (domPill) {
    domPill.textContent = '通用';
    domPill.classList.add('hidden');
  }
  REAL_TASK_TERMS.splice(0, REAL_TASK_TERMS.length);
  STREAM_TABLE_ROWS.splice(0, STREAM_TABLE_ROWS.length);
  ASR_SEGMENT_ROWS.splice(0, ASR_SEGMENT_ROWS.length);
  SUBTITLE_CUE_ROWS.splice(0, SUBTITLE_CUE_ROWS.length);
  PIPELINE_BATCH_NODES.splice(
    0,
    PIPELINE_BATCH_NODES.length,
    ...DEFAULT_PIPELINE_BATCH_NODES.map(n => ({ ...n }))
  );
  const themeInput = document.getElementById('terms-theme-input');
  if (themeInput) {
    themeInput.value = '';
    autoResizeThemeTextarea(themeInput);
  }
  state.activeBatchId = 'all';
  syncActiveTaskLogsView();
  syncPipelineRibbon(null, 0, 0);
  renderBatchTree();
  syncDetailPane();
  renderStreamTable();
  renderTaskTerms();
}

function applyCachedTaskDetail(task, cached, preserveEdits) {
  STREAM_TABLE_ROWS.splice(0, STREAM_TABLE_ROWS.length, ...cached.streamRows);
  SUBTITLE_CUE_ROWS.splice(0, SUBTITLE_CUE_ROWS.length, ...cached.cueRows);
  if (cached.hasAsr) {
    ASR_SEGMENT_ROWS.splice(0, ASR_SEGMENT_ROWS.length, ...cached.asrRows);
    state.loadedAsrTaskId = cached.asrRows.length > 0 ? task.id : '';
  } else {
    ASR_SEGMENT_ROWS.splice(0, ASR_SEGMENT_ROWS.length);
    state.loadedAsrTaskId = '';
  }
  if ((!preserveEdits || !state.isEditingTerms) && cached.termsLoaded) {
    REAL_TASK_TERMS.splice(0, REAL_TASK_TERMS.length, ...cloneTermsList(cached.terms));
    const themeInput = document.getElementById('terms-theme-input');
    if (themeInput && document.activeElement !== themeInput) {
      themeInput.value = cached.termsSummary || '';
      autoResizeThemeTextarea(themeInput);
    }
  }
  state.hydratedTaskId = task.id;
  if (task.state === 'paused' && task.stateLabel === '待审术语') {
    state.activeBatchId = 'terms-03';
  } else if (state.activeBatchId === 'terms-03' && task.rawState === 'running') {
    state.activeBatchId = 'all';
  }
  syncBatchNodesFromLoadedRows(cached.stagesDone || task.raw?.stages_done || {}, cached.totalWords || 0);
  syncPipelineRibbon(task, STREAM_TABLE_ROWS.length, REAL_TASK_TERMS.length);
  renderBatchTree();
  syncDetailPane();
  renderStreamTable();
  if (!preserveEdits || !state.isEditingTerms) renderTaskTerms();
}

function clearWorkbenchRowsForSwitch(task) {
  cancelPendingStreamChunk();
  REAL_TASK_TERMS.splice(0, REAL_TASK_TERMS.length);
  STREAM_TABLE_ROWS.splice(0, STREAM_TABLE_ROWS.length);
  ASR_SEGMENT_ROWS.splice(0, ASR_SEGMENT_ROWS.length);
  SUBTITLE_CUE_ROWS.splice(0, SUBTITLE_CUE_ROWS.length);
  state.loadedAsrTaskId = '';
  state.hydratedTaskId = task.id;
  const themeInput = document.getElementById('terms-theme-input');
  if (themeInput && document.activeElement !== themeInput) {
    themeInput.value = '';
    autoResizeThemeTextarea(themeInput);
  }
  if (task.state === 'paused' && task.stateLabel === '待审术语') {
    state.activeBatchId = 'terms-03';
  } else if (state.activeBatchId === 'terms-03' && task.rawState === 'running') {
    state.activeBatchId = 'all';
  }
  syncBatchNodesFromLoadedRows(task.raw?.stages_done || {}, 0);
  syncPipelineRibbon(task, 0, 0);
  renderBatchTree();
  syncDetailPane();
  renderStreamTable();
  renderTaskTerms();
}

export async function ensureActiveTaskAsrLoaded(taskId) {
  if (!taskId) return;
  const cached = TASK_DETAIL_CACHE.get(taskId);
  if (cached?.hasAsr) {
    ASR_SEGMENT_ROWS.splice(0, ASR_SEGMENT_ROWS.length, ...cached.asrRows);
    state.loadedAsrTaskId = cached.asrRows.length > 0 ? taskId : '';
    renderStreamTable();
    return;
  }
  if (inflightAsrTaskId === taskId && inflightAsrPromise) {
    return inflightAsrPromise;
  }
  const tbody = document.getElementById('structured-stream-tbody');
  if (tbody && state.activeBatchId === 'asr-01') {
    cancelPendingStreamChunk();
    tbody.innerHTML = `
      <tr>
        <td colspan="3" style="padding:40px 16px; text-align:center; color:var(--text-muted); font-size:var(--fs-xs);">
          正在载入词级时间轴…
        </td>
      </tr>
    `;
  }
  inflightAsrTaskId = taskId;
  const p = (async () => {
    try {
      const uttsData = await apiFetch(
        `/api/tasks/${encodeURIComponent(taskId)}/utterances?include_asr=1`
      );
      const rawAsr = Array.isArray(uttsData.asr_segments) ? uttsData.asr_segments : [];
      const mappedAsr = mapAsrSegmentsToRows(rawAsr);
      upsertTaskDetailCache(taskId, {
        asrRows: mappedAsr,
        hasAsr: true,
        totalWords: Number(uttsData.total_words || 0),
      });
      if (taskId === state.activeTaskId) {
        ASR_SEGMENT_ROWS.splice(0, ASR_SEGMENT_ROWS.length, ...mappedAsr);
        state.loadedAsrTaskId = mappedAsr.length > 0 ? taskId : '';
        const card = REAL_TASKS.find(x => x.id === taskId);
        syncBatchNodesFromLoadedRows(
          card?.raw?.stages_done || {},
          Number(uttsData.total_words || 0)
        );
        if (card) syncPipelineRibbon(card, STREAM_TABLE_ROWS.length, REAL_TASK_TERMS.length);
        renderBatchTree();
        if (state.activeBatchId === 'asr-01') renderStreamTable();
      }
    } catch (err) {
      appendLog('warn', 'ASR', `加载语音识别词流失败 (${taskId}): ${err.message}`, true, undefined, {
        taskId,
        persist: false,
      });
    } finally {
      if (inflightAsrTaskId === taskId) {
        inflightAsrTaskId = '';
        inflightAsrPromise = null;
      }
    }
  })();
  inflightAsrPromise = p;
  return p;
}

export async function selectActiveTask(taskId, options = {}) {
  const preserveEdits = Boolean(options.preserveEdits);
  const forceNetwork = Boolean(options.forceNetwork);
  if (options.resetEdits) {
    state.isEditingTerms = false;
    setAutosaveStatus('idle', '');
    resetDetailScrollPositions();
  }
  const task = REAL_TASKS.find(x => x.id === taskId);
  if (!task) {
    resetWorkbenchToEmpty();
    renderTaskRail();
    return;
  }
  renderTaskRail();
  document.getElementById('detail-task-title').textContent = task.title;
  const domPill = document.getElementById('detail-domain-pill');
  if (domPill) {
    domPill.textContent = task.domain || '通用';
    domPill.classList.remove('hidden');
  }
  syncActiveTaskLogsView();
  if (!task.fromBackend) {
    syncPipelineRibbon(task, STREAM_TABLE_ROWS.length, REAL_TASK_TERMS.length);
    renderBatchTree();
    syncDetailPane();
    renderStreamTable();
    renderTaskTerms();
    return;
  }

  const cached = !forceNetwork ? TASK_DETAIL_CACHE.get(task.id) : null;
  if (cached?.uttsLoaded) {
    applyCachedTaskDetail(task, cached, preserveEdits);
    const currentSig = computeTaskSnapshotSig(options.prefetchedTask || task.raw);
    const isLiveState =
      task.rawState === 'running' || task.rawState === 'stopping' || task.rawState === 'queued';
    const needAsrNow =
      state.activeBatchId === 'asr-01' && Boolean(task.raw?.stages_done?.asr) && !cached.hasAsr;
    if (!isLiveState && cached.sig === currentSig && cached.termsLoaded && cached.logsLoaded && !needAsrNow) {
      if (activeDetailController) {
        activeDetailController.abort();
        activeDetailController = null;
      }
      return;
    }
  } else if (state.hydratedTaskId !== task.id) {
    clearWorkbenchRowsForSwitch(task);
  }

  const canReuseTaskRaw =
    task.raw && task.rawState !== 'running' && task.rawState !== 'stopping' ? task.raw : null;
  await loadBackendTaskDetail(task.id, {
    preserveEdits,
    prefetchedTask: options.prefetchedTask || canReuseTaskRaw,
  });
}

export async function loadBackendTasks(options = {}) {
  const skipActiveDetail = Boolean(options.skipActiveDetail);
  const forceDetailReload = Boolean(options.forceDetailReload);
  if (inflightTasksReload) {
    pendingTasksReloadOptions = pendingTasksReloadOptions
      ? {
          skipActiveDetail: pendingTasksReloadOptions.skipActiveDetail && skipActiveDetail,
          forceDetailReload: pendingTasksReloadOptions.forceDetailReload || forceDetailReload,
        }
      : { skipActiveDetail, forceDetailReload };
    return inflightTasksReload;
  }
  const run = (async () => {
    try {
      const data = await apiFetch('/api/tasks');
      state.isBackendConnected = true;
      const prevActive = REAL_TASKS.find(t => t.id === state.activeTaskId);
      const prevState = prevActive?.rawState || prevActive?.state || '';
      const tasks = (data.tasks || []).map(mapBackendTaskToCard);
      const liveIds = new Set(tasks.map(t => t.id));
      for (const key of Array.from(TASK_LOGS_BY_ID.keys())) {
        if (key && !liveIds.has(key)) TASK_LOGS_BY_ID.delete(key);
      }
      for (const key of Array.from(TASK_DETAIL_CACHE.keys())) {
        if (key && !liveIds.has(key)) TASK_DETAIL_CACHE.delete(key);
      }
      REAL_TASKS.splice(0, REAL_TASKS.length, ...tasks);
      if (REAL_TASKS.length > 0) {
        if (!REAL_TASKS.some(t => t.id === state.activeTaskId)) {
          state.activeTaskId = REAL_TASKS[0].id;
        }
        renderTaskRail();
        const nextActive = REAL_TASKS.find(t => t.id === state.activeTaskId);
        const nextState = nextActive?.rawState || nextActive?.state || '';
        const stateChanged = prevState !== nextState;
        if (!skipActiveDetail || stateChanged || forceDetailReload || !prevActive) {
          if (!isUserEditingDom() || stateChanged || forceDetailReload) {
            await selectActiveTask(state.activeTaskId, {
              preserveEdits: true,
              forceNetwork: forceDetailReload || stateChanged,
              prefetchedTask: nextActive?.raw || null,
            });
          } else if (nextActive) {
            syncPipelineRibbon(nextActive, STREAM_TABLE_ROWS.length, REAL_TASK_TERMS.length);
          }
        } else if (nextActive) {
          syncPipelineRibbon(nextActive, STREAM_TABLE_ROWS.length, REAL_TASK_TERMS.length);
        }
      } else {
        state.activeTaskId = '';
        renderTaskRail();
        resetWorkbenchToEmpty();
      }
    } catch (err) {
      appendLog('warn', 'TASKS', `同步任务列表失败: ${err.message}`, true, undefined, {
        taskId: '',
        persist: false,
      });
    } finally {
      inflightTasksReload = null;
      if (pendingTasksReloadOptions) {
        const nextOpts = pendingTasksReloadOptions;
        pendingTasksReloadOptions = null;
        await loadBackendTasks(nextOpts);
      }
    }
  })();
  inflightTasksReload = run;
  return run;
}

export async function loadBackendTaskDetail(taskId, options = {}) {
  const preserveEdits = Boolean(options.preserveEdits);
  const prefetchedTask = options.prefetchedTask || null;
  if (activeDetailController) activeDetailController.abort();
  const controller = new AbortController();
  activeDetailController = controller;
  const sig = { signal: controller.signal };
  const taskKnown = prefetchedTask || REAL_TASKS.find(x => x.id === taskId)?.raw || null;
  const cachedEntry = TASK_DETAIL_CACHE.get(taskId);
  const hasCachedAsr =
    (state.loadedAsrTaskId === taskId && ASR_SEGMENT_ROWS.length > 0) || Boolean(cachedEntry?.hasAsr);
  const needAsr =
    Boolean(options.forceAsr) ||
    state.activeBatchId === 'asr-01' ||
    Boolean(taskKnown && taskKnown.stages_done?.asr && !taskKnown.stages_done?.split);
  const requestIncludeAsr = needAsr && (!hasCachedAsr || Boolean(options.forceAsr));
  const uttsUrl = `/api/tasks/${encodeURIComponent(taskId)}/utterances?include_asr=${requestIncludeAsr ? '1' : '0'}`;
  try {
    const taskPromise = prefetchedTask
      ? Promise.resolve(prefetchedTask)
      : apiFetch(`/api/tasks/${encodeURIComponent(taskId)}`, sig);
    const uttsPromise = apiFetch(uttsUrl, sig);
    const termsPromise = apiFetch(`/api/tasks/${encodeURIComponent(taskId)}/terms`, sig);
    const logsPromise = apiFetch(`/api/tasks/${encodeURIComponent(taskId)}/logs`, sig);
    termsPromise.catch(() => {});
    logsPromise.catch(() => {});

    const [taskData, uttsData] = await Promise.all([taskPromise, uttsPromise]);
    if (controller.signal.aborted) return;

    const card = mapBackendTaskToCard(taskData);
    const idx = REAL_TASKS.findIndex(x => x.id === taskId);
    if (idx >= 0) REAL_TASKS[idx] = card;
    renderTaskRail();

    const utts = Array.isArray(uttsData.utterances) ? uttsData.utterances : [];
    const totalWords = Number(uttsData.total_words || 0);
    const asrSegments = requestIncludeAsr
      ? Array.isArray(uttsData.asr_segments)
        ? uttsData.asr_segments
        : []
      : null;

    if (taskId === state.activeTaskId) {
      if (!requestIncludeAsr && cachedEntry?.hasAsr && state.loadedAsrTaskId !== taskId) {
        ASR_SEGMENT_ROWS.splice(0, ASR_SEGMENT_ROWS.length, ...cachedEntry.asrRows);
        state.loadedAsrTaskId = cachedEntry.asrRows.length > 0 ? taskId : '';
      } else if (!requestIncludeAsr && state.loadedAsrTaskId !== taskId) {
        ASR_SEGMENT_ROWS.splice(0, ASR_SEGMENT_ROWS.length);
        state.loadedAsrTaskId = '';
      }
      rebuildStreamRowsFromUtterances(utts, taskData.stages_done || {}, asrSegments, totalWords);
      if (requestIncludeAsr) {
        state.loadedAsrTaskId = ASR_SEGMENT_ROWS.length > 0 ? taskId : '';
      }
      state.hydratedTaskId = taskId;
      upsertTaskDetailCache(taskId, {
        sig: computeTaskSnapshotSig(taskData),
        uttsLoaded: true,
        totalWords,
        stagesDone: { ...(taskData.stages_done || {}) },
        streamRows: [...STREAM_TABLE_ROWS],
        cueRows: [...SUBTITLE_CUE_ROWS],
        ...(requestIncludeAsr ? { asrRows: [...ASR_SEGMENT_ROWS], hasAsr: true } : {}),
      });

      if (card.state === 'paused' && card.stateLabel === '待审术语') {
        state.activeBatchId = 'terms-03';
      } else if (state.activeBatchId === 'terms-03' && card.rawState === 'running') {
        state.activeBatchId = 'all';
      }
      syncPipelineRibbon(card, STREAM_TABLE_ROWS.length, REAL_TASK_TERMS.length);
      renderBatchTree();
      syncDetailPane();
      renderStreamTable();
    }

    const [termsData, logsData] = await Promise.all([termsPromise, logsPromise]);
    if (activeDetailController === controller) activeDetailController = null;
    if (controller.signal.aborted) return;

    setTaskLogsFromServer(taskId, logsData?.logs);
    const normalizedTerms = normalizeBackendTerms(termsData?.terms);
    const termsSummary = termsData?.summary || '';
    upsertTaskDetailCache(taskId, {
      sig: computeTaskSnapshotSig(taskData),
      termsLoaded: true,
      logsLoaded: true,
      terms: cloneTermsList(normalizedTerms),
      termsSummary,
    });

    if (taskId !== state.activeTaskId) {
      return;
    }

    if (!preserveEdits || !state.isEditingTerms) {
      REAL_TASK_TERMS.splice(0, REAL_TASK_TERMS.length, ...normalizedTerms);
      const themeInput = document.getElementById('terms-theme-input');
      if (themeInput && document.activeElement !== themeInput) {
        themeInput.value = termsSummary;
        autoResizeThemeTextarea(themeInput);
      }
      syncPipelineRibbon(card, STREAM_TABLE_ROWS.length, REAL_TASK_TERMS.length);
      renderTaskTerms();
    }
  } catch (err) {
    if (err?.name === 'AbortError') return;
    appendLog('warn', 'TASK', `加载任务详情失败 (${taskId}): ${err.message}`, true, undefined, {
      taskId,
      persist: false,
    });
  }
}

export function initTaskRailEvents() {
  document.getElementById('task-rail-list')?.addEventListener('click', async e => {
    const card = e.target.closest('[data-task-id]');
    if (!card) return;
    const nextId = card.getAttribute('data-task-id') || '';
    if (!nextId || nextId === state.activeTaskId) return;
    state.activeTaskId = nextId;
    await selectActiveTask(state.activeTaskId, { resetEdits: true });
  });

  document.getElementById('batch-tree-list')?.addEventListener('click', async e => {
    const node = e.target.closest('[data-batch-id]');
    if (!node) return;
    const nextBatch = node.getAttribute('data-batch-id') || 'all';
    const stageChanged = nextBatch !== state.activeBatchId;
    state.activeBatchId = nextBatch;
    if (stageChanged) resetDetailScrollPositions();
    renderBatchTree();
    syncDetailPane();
    if (state.activeBatchId === 'asr-01') {
      const activeTask = REAL_TASKS.find(x => x.id === state.activeTaskId);
      if (
        activeTask?.fromBackend &&
        activeTask.raw?.stages_done?.asr &&
        state.loadedAsrTaskId !== state.activeTaskId
      ) {
        await ensureActiveTaskAsrLoaded(state.activeTaskId);
        return;
      }
    }
    if (state.activeBatchId !== 'terms-03' && state.activeBatchId !== 'runtime-log') {
      renderStreamTable();
    }
  });

  document.getElementById('btn-open-output-root')?.addEventListener('click', async () => {
    if (!state.isBackendConnected) return;
    try {
      await apiFetch('/api/output/open-dir', { method: 'POST' });
    } catch (err) {
      if (state.activeTaskId) {
        try {
          await apiFetch(`/api/tasks/${encodeURIComponent(state.activeTaskId)}/open-dir`, {
            method: 'POST',
          });
          return;
        } catch (innerErr) {
          reportActionError('OPEN', innerErr, state.activeTaskId);
          return;
        }
      }
      reportActionError('OPEN', err);
    }
  });

  document.getElementById('btn-sync-subtitles-from-srt')?.addEventListener('click', async () => {
    const task = REAL_TASKS.find(x => x.id === state.activeTaskId);
    if (!task || !state.isBackendConnected || !task.fromBackend) return;
    try {
      setAutosaveStatus('running', '同步 SRT…');
      const res = await apiFetch(
        `/api/tasks/${encodeURIComponent(state.activeTaskId)}/subtitles/sync`,
        { method: 'POST' }
      );
      TASK_DETAIL_CACHE.delete(state.activeTaskId);
      await loadBackendTaskDetail(state.activeTaskId, { preserveEdits: true });
      setAutosaveStatus('completed', `已同步 ${res.cues || 0} 轴`);
      appendLog('llm', 'SUBS', `已从本地 SRT 文件同步更新数据库 (${res.cues || 0} 轴)。`, false);
    } catch (err) {
      setAutosaveStatus('error', '同步失败');
      reportActionError('SUBS', err, state.activeTaskId);
    }
  });

  document.getElementById('btn-save-utterances-and-continue')?.addEventListener('click', async () => {
    const task = REAL_TASKS.find(x => x.id === state.activeTaskId);
    try {
      if (state.isBackendConnected && task?.fromBackend && task.rawState === 'paused') {
        await apiFetch(`/api/tasks/${encodeURIComponent(state.activeTaskId)}/resume`, {
          method: 'POST',
        });
        TASK_DETAIL_CACHE.delete(state.activeTaskId);
        await loadBackendTasks();
      }
      appendLog('llm', 'SUBS', `已确认字幕并继续后续任务 (${state.activeTaskId})。`, false);
    } catch (err) {
      reportActionError('SUBS', err, state.activeTaskId);
    }
  });

  document.getElementById('btn-start-resume-task')?.addEventListener('click', async () => {
    const task = REAL_TASKS.find(x => x.id === state.activeTaskId);
    if (!task) return;
    const rawState = task.rawState || task.state || 'idle';
    if (rawState === 'running' || rawState === 'stopping' || rawState === 'queued') return;

    if (state.isBackendConnected && task.fromBackend) {
      try {
        if (state.isEditingTerms) await persistTaskTermsToBackend();
        if (rawState === 'paused') {
          await apiFetch(`/api/tasks/${encodeURIComponent(state.activeTaskId)}/resume`, {
            method: 'POST',
          });
          appendLog('llm', 'RUN', `继续执行任务 ${state.activeTaskId}…`, false);
        } else {
          const desiredStage =
            task.desiredStage || task.raw?.execution?.desired_stage || 'subtitles';
          const dubbingEnabled = Boolean(
            task.dubbingEnabled ??
              task.raw?.execution?.dubbing_enabled ??
              (desiredStage === 'all' || desiredStage === 'dubbing')
          );
          await apiFetch(`/api/tasks/${encodeURIComponent(state.activeTaskId)}/run`, {
            method: 'POST',
            body: JSON.stringify({ stage: desiredStage, dubbing: dubbingEnabled }),
          });
          appendLog(
            'asr',
            'RUN',
            `${task.hasAnyProgress ? '继续' : '启动'}任务 ${state.activeTaskId} (${desiredStage})…`,
            false
          );
        }
        TASK_DETAIL_CACHE.delete(state.activeTaskId);
        await loadBackendTasks();
      } catch (err) {
        reportActionError('RUN', err, state.activeTaskId);
      }
    }
  });

  document.getElementById('btn-open-task-dir')?.addEventListener('click', async () => {
    const task = REAL_TASKS.find(x => x.id === state.activeTaskId);
    if (!task || !state.isBackendConnected || !task.fromBackend) return;
    try {
      await apiFetch(`/api/tasks/${encodeURIComponent(state.activeTaskId)}/open-dir`, {
        method: 'POST',
      });
    } catch (err) {
      reportActionError('OPEN', err, state.activeTaskId);
    }
  });

  document.getElementById('btn-burn-sub-video')?.addEventListener('click', async () => {
    const task = REAL_TASKS.find(x => x.id === state.activeTaskId);
    if (!task || !state.isBackendConnected || !task.fromBackend) return;
    try {
      await apiFetch(`/api/tasks/${encodeURIComponent(state.activeTaskId)}/run`, {
        method: 'POST',
        body: JSON.stringify({ stage: 'burn', dubbing: false }),
      });
      TASK_DETAIL_CACHE.delete(state.activeTaskId);
      await loadBackendTasks();
    } catch (err) {
      reportActionError('VIDEO', err, state.activeTaskId);
    }
  });

  document.getElementById('btn-run-dubbing')?.addEventListener('click', async () => {
    const task = REAL_TASKS.find(x => x.id === state.activeTaskId);
    if (!task || !state.isBackendConnected || !task.fromBackend) return;
    try {
      await apiFetch(`/api/tasks/${encodeURIComponent(state.activeTaskId)}/run`, {
        method: 'POST',
        body: JSON.stringify({ stage: 'dubbing', dubbing: true }),
      });
      TASK_DETAIL_CACHE.delete(state.activeTaskId);
      await loadBackendTasks();
    } catch (err) {
      reportActionError('TTS', err, state.activeTaskId);
    }
  });

  document.getElementById('btn-stop-task')?.addEventListener('click', async () => {
    const task = REAL_TASKS.find(x => x.id === state.activeTaskId);
    if (!task || !state.isBackendConnected || !task.fromBackend) return;
    try {
      await apiFetch(`/api/tasks/${encodeURIComponent(state.activeTaskId)}/stop`, {
        method: 'POST',
      });
      TASK_DETAIL_CACHE.delete(state.activeTaskId);
      await loadBackendTasks();
    } catch (err) {
      reportActionError('STOP', err, state.activeTaskId);
    }
  });
}
