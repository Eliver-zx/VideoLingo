import {
  state,
  REAL_TASKS,
  REAL_TASK_TERMS,
  STREAM_TABLE_ROWS,
  ASR_SEGMENT_ROWS,
  SUBTITLE_CUE_ROWS,
  TASK_DETAIL_CACHE,
  burnRuntimeState,
} from './state.js';
import { apiFetch, autoResizeThemeTextarea, isUserEditingDom } from './dom-utils.js';
import { appendLog, hydrateSystemLogs } from './console-log.js';
import { syncDetailPane } from './router.js';
import { rebuildStreamRowsFromUtterances, renderStreamTable } from './stream-table.js';
import { renderTaskTerms } from './terms-review.js';
import {
  computeTaskSnapshotSig,
  mapBackendTaskToCard,
  patchTaskRailCard,
  renderBatchTree,
  syncPipelineRibbon,
  loadBackendTasks,
  upsertTaskDetailCache,
} from './task-rail.js';
import { appendBurnLogEntry, renderBurnControls } from './burn-view.js';

const inflightSliceByKey = new Map();
const pendingSliceNextBatch = new Map();

function applyRunnerEventToState(d) {
  const msg =
    d.message ||
    (d.type === 'task_paused' ? d.pause_message : '') ||
    (d.type === 'task_error' ? d.error : '');
  if (msg && d.task_id) {
    const warn =
      Boolean(d.warn ?? (d.type === 'task_error' && d.error)) ||
      d.log_level === 'warn' ||
      d.log_level === 'error';
    const logTag = d.log_tag || (warn ? 'warn' : 'llm');
    const logLabel = d.log_label || String(d.step || d.state || 'RUN').toUpperCase().slice(0, 8);
    appendLog(logTag, logLabel, msg, warn, d.time, { taskId: d.task_id, fromServer: true });
  }
  if (!d.task_id) return null;
  const tCard = REAL_TASKS.find(x => x.id === d.task_id);
  if (!tCard || !tCard.raw) return null;

  const prevState = tCard.rawState || tCard.state || '';
  const prevStep = tCard.raw.execution?.current_step || '';

  tCard.raw.execution = {
    ...(tCard.raw.execution || {}),
    state: d.state || tCard.rawState,
    current_step: d.step ?? tCard.raw.execution?.current_step,
    sub_step: d.sub_step ?? tCard.raw.execution?.sub_step,
    sub_progress: d.sub_progress ?? tCard.raw.execution?.sub_progress,
    progress: d.progress ?? tCard.raw.execution?.progress,
    pause_reason: d.pause_message ?? tCard.raw.execution?.pause_reason,
    error: d.error ?? tCard.raw.execution?.error,
  };
  if (d.stages_done && typeof d.stages_done === 'object') {
    tCard.raw.stages_done = {
      ...(tCard.raw.stages_done || {}),
      ...d.stages_done,
    };
  }
  const updated = mapBackendTaskToCard(tCard.raw);
  Object.assign(tCard, updated);
  if (d.task_id !== state.activeTaskId) {
    TASK_DETAIL_CACHE.delete(d.task_id);
  }
  patchTaskRailCard(tCard);
  const nextState = tCard.rawState || tCard.state || '';
  const nextStep = tCard.raw.execution?.current_step || '';
  return {
    card: tCard,
    stateChanged: prevState !== nextState,
    stepChanged: prevStep !== nextStep,
  };
}

function resolveAutoBatchTransition(d, card, stateChanged) {
  if (!card) return null;
  if (card.state === 'paused' && card.stateLabel === '待审术语') {
    return 'terms-03';
  }
  if (d.artifact === 'terms' && !card.raw?.stages_done?.translate) {
    return 'terms-03';
  }
  if (
    (stateChanged || d.artifact === 'utterances') &&
    state.activeBatchId === 'terms-03' &&
    card.state === 'running'
  ) {
    return 'all';
  }
  return null;
}

async function hydrateActiveTaskSlice(taskId, artifact, nextBatchId) {
  if (!taskId || taskId !== state.activeTaskId) return;
  const key = `${taskId}:${artifact}`;
  if (inflightSliceByKey.has(key)) {
    pendingSliceNextBatch.set(key, nextBatchId || null);
    return;
  }
  const run = (async () => {
    const editing = isUserEditingDom();
    try {
      if (artifact === 'terms' && !state.isEditingTerms && !editing) {
        const termsData = await apiFetch(`/api/tasks/${encodeURIComponent(taskId)}/terms`);
        if (taskId !== state.activeTaskId) return;
        const rawTerms = Array.isArray(termsData.terms) ? termsData.terms : [];
        REAL_TASK_TERMS.splice(
          0,
          REAL_TASK_TERMS.length,
          ...rawTerms.map(item => ({
            src: item.src || '',
            tgt: item.tgt || '',
            aliases: Array.isArray(item.aliases) ? item.aliases : [],
            note: item.note || '',
            fix_source: Boolean(item.fix_source),
            source: item.source || 'task',
            in_domain_kb: Boolean(item.in_domain_kb),
            promote: Boolean(item.promoted ?? item.promote_to_domain ?? item.promote ?? false),
          }))
        );
        const themeInput = document.getElementById('terms-theme-input');
        if (themeInput && document.activeElement !== themeInput) {
          themeInput.value = termsData.summary || '';
          autoResizeThemeTextarea(themeInput);
        }
        const card = REAL_TASKS.find(x => x.id === taskId);
        upsertTaskDetailCache(taskId, {
          sig: computeTaskSnapshotSig(card?.raw),
          termsLoaded: true,
          terms: REAL_TASK_TERMS.map(t => ({ ...t, aliases: [...t.aliases] })),
          termsSummary: termsData.summary || '',
        });
        renderTaskTerms();
      } else if (
        (artifact === 'utterances' || artifact === 'words' || artifact === 'subtitles') &&
        !editing
      ) {
        const needAsr =
          artifact === 'words' ||
          state.activeBatchId === 'asr-01';
        const hasCachedAsr =
          state.loadedAsrTaskId === taskId && ASR_SEGMENT_ROWS.length > 0;
        const requestIncludeAsr = needAsr && (artifact === 'words' || !hasCachedAsr);
        const uttsData = await apiFetch(
          `/api/tasks/${encodeURIComponent(taskId)}/utterances?include_asr=${requestIncludeAsr ? '1' : '0'}`
        );
        if (taskId !== state.activeTaskId) return;
        const card = REAL_TASKS.find(x => x.id === taskId);
        const utts = Array.isArray(uttsData.utterances) ? uttsData.utterances : [];
        const asrSegments = requestIncludeAsr
          ? Array.isArray(uttsData.asr_segments)
            ? uttsData.asr_segments
            : []
          : null;
        const totalWords = Number(uttsData.total_words || 0);
        rebuildStreamRowsFromUtterances(utts, card?.raw?.stages_done || {}, asrSegments, totalWords);
        if (requestIncludeAsr) {
          state.loadedAsrTaskId = ASR_SEGMENT_ROWS.length > 0 ? taskId : '';
        }
        upsertTaskDetailCache(taskId, {
          sig: computeTaskSnapshotSig(card?.raw),
          uttsLoaded: true,
          totalWords,
          stagesDone: { ...(card?.raw?.stages_done || {}) },
          streamRows: [...STREAM_TABLE_ROWS],
          cueRows: [...SUBTITLE_CUE_ROWS],
          ...(requestIncludeAsr ? { asrRows: [...ASR_SEGMENT_ROWS], hasAsr: true } : {}),
        });
      }
      if (taskId !== state.activeTaskId) return;
      const card = REAL_TASKS.find(x => x.id === taskId);
      if (nextBatchId && !editing) {
        state.activeBatchId = nextBatchId;
      }
      if (card) {
        syncPipelineRibbon(card, STREAM_TABLE_ROWS.length, REAL_TASK_TERMS.length);
      }
      renderBatchTree();
      syncDetailPane();
      if (!editing) {
        renderStreamTable();
      }
    } catch (err) {
      appendLog('warn', 'SYNC', `实时同步任务数据失败 (${artifact}): ${err.message}`, true, undefined, {
        taskId,
        persist: false,
      });
    } finally {
      inflightSliceByKey.delete(key);
      if (pendingSliceNextBatch.has(key)) {
        const queuedNext = pendingSliceNextBatch.get(key);
        pendingSliceNextBatch.delete(key);
        hydrateActiveTaskSlice(taskId, artifact, queuedNext);
      }
    }
  })();
  inflightSliceByKey.set(key, run);
  return run;
}

export function connectEventStream() {
  if (typeof EventSource === 'undefined') return;
  try {
    const es = new EventSource('/api/events');
    es.onmessage = e => {
      try {
        const d = JSON.parse(e.data);
        if (!d || d.type === 'heartbeat') return;
        if (d.type === 'connected') {
          if (Array.isArray(d.recent_logs)) {
            hydrateSystemLogs(d.recent_logs);
          }
          return;
        }
        if (d.type === 'standalone_burn') {
          const { type: _t, log: burnLog, ...rest } = d;
          Object.assign(burnRuntimeState, rest);
          if (burnLog) {
            appendBurnLogEntry(burnLog);
          }
          renderBurnControls();
          return;
        }
        const applied = applyRunnerEventToState(d);
        if (d.task_id && d.task_id === state.activeTaskId && applied) {
          syncPipelineRibbon(applied.card, STREAM_TABLE_ROWS.length, REAL_TASK_TERMS.length);
          const nextBatchId = resolveAutoBatchTransition(d, applied.card, applied.stateChanged);
          if (d.type === 'task_artifact' && d.artifact) {
            hydrateActiveTaskSlice(d.task_id, d.artifact, nextBatchId);
            return;
          }
          if (nextBatchId && !isUserEditingDom()) {
            state.activeBatchId = nextBatchId;
            renderBatchTree();
            syncDetailPane();
            renderStreamTable();
          }
        }
        const isLifecycleTransition =
          d.type === 'task_started' ||
          d.type === 'task_paused' ||
          d.type === 'task_resumed' ||
          d.type === 'task_completed' ||
          d.type === 'task_error' ||
          d.type === 'task_stopped' ||
          d.type === 'queue_updated' ||
          Boolean(applied?.stateChanged);
        if (isLifecycleTransition) {
          loadBackendTasks({ skipActiveDetail: false, forceDetailReload: true });
        }
      } catch (parseErr) {
        appendLog('warn', 'SSE', `解析事件流失败: ${parseErr.message}`, true, undefined, {
          taskId: '',
          persist: false,
        });
      }
    };
  } catch (err) {
    appendLog('warn', 'SSE', `连接实时事件流失败: ${err.message}`, true, undefined, {
      taskId: '',
      persist: false,
    });
  }
}
