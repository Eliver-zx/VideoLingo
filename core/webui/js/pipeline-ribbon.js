import { state, PIPELINE_BATCH_NODES, ASR_SEGMENT_ROWS } from './state.js';
import { escapeHtml, escapeAttr, sanitizeErrorSummary } from './dom-utils.js';
import { syncPrimaryCtaFocus } from './router.js';

export function pillClassForTask(t) {
  return t.state === 'completed'
    ? 'completed'
    : t.state === 'paused'
      ? 'paused'
      : t.state === 'error'
        ? 'error'
        : 'running';
}

export function mapBackendTaskToCard(t) {
  const exec = t.execution || {};
  const sd = t.stages_done || {};
  const rawState = exec.state || 'idle';
  const domain =
    t.config_override?.glossary_domain !== undefined
      ? String(t.config_override.glossary_domain)
      : (state.currentConfig.glossary_domain ?? 'destiny2');
  const hasAnyProgress = Boolean(
    sd.asr ||
      sd.split ||
      sd.terms ||
      sd.translate ||
      sd.subtitles ||
      sd.sub_video ||
      sd.dub_audio ||
      sd.dub_video
  );
  let stateLabel = hasAnyProgress ? '待继续' : '待启动';
  let cardState = 'paused';
  let stepSummary = exec.current_step || (hasAnyProgress ? '已暂停 · 可继续' : '就绪待启动');
  let errorDetail = '';
  let progress = hasAnyProgress ? Math.round((exec.progress || 0) * 100) : 0;

  if (rawState === 'paused') {
    cardState = 'paused';
    const reason = String(exec.pause_reason || '').toLowerCase();
    stateLabel = reason.includes('terminology') ? '待审术语' : '待校字幕';
    stepSummary = reason.includes('terminology') ? '等待确认术语译名' : '等待校对成品字幕';
    progress = Math.max(progress, 45);
  } else if (rawState === 'queued') {
    cardState = 'paused';
    stateLabel = '排队中';
    stepSummary = '等待前一任务完成…';
  } else if (rawState === 'running' || rawState === 'stopping') {
    cardState = 'running';
    stateLabel = rawState === 'stopping' ? '停止中' : '运行中';
    const rawProgressPct = Math.round((exec.progress || 0) * 100);
    progress = Math.max(rawProgressPct, 4);
    const subStep = String(exec.sub_step || '').trim();
    if (subStep) {
      stepSummary = `${subStep} · ${progress}%`;
    } else if (exec.current_step) {
      stepSummary = `${exec.current_step} · ${progress}%`;
    } else {
      stepSummary = '正在处理…';
    }
  } else if (rawState === 'stopped') {
    cardState = 'paused';
    stateLabel = '已停止';
    stepSummary = exec.current_step ? `停于: ${exec.current_step}` : '已停止 · 可继续';
  } else if (rawState === 'error') {
    cardState = 'error';
    stateLabel = '异常';
    errorDetail = String(exec.error || '执行出错');
    stepSummary = sanitizeErrorSummary(errorDetail);
  } else if (sd.dub_video || (sd.sub_video && !exec.dubbing_enabled)) {
    cardState = 'completed';
    stateLabel = '已完成';
    stepSummary = sd.dub_video ? '配音成片已输出' : '字幕成片已输出';
    progress = 100;
  } else if (sd.subtitles || sd.translate) {
    cardState = 'completed';
    stateLabel = '字幕就绪';
    stepSummary = 'SRT 已生成 · 待合成';
    progress = 82;
  } else if (sd.terms) {
    cardState = 'paused';
    stateLabel = '待审术语';
    stepSummary = '术语已提取 · 待确认';
    progress = 45;
  } else if (sd.split || sd.asr) {
    cardState = 'paused';
    stateLabel = '待继续';
    stepSummary = sd.split ? '已完成断句 · 待继续' : '已完成识别 · 待继续';
    progress = sd.split ? 30 : 18;
  }

  return {
    id: t.task_id,
    title: t.title || t.media?.filename || t.task_id,
    domain,
    state: cardState,
    rawState,
    hasAnyProgress,
    desiredStage: exec.desired_stage || 'subtitles',
    dubbingEnabled: Boolean(exec.dubbing_enabled),
    stateLabel,
    stepSummary,
    errorDetail,
    progress,
    raw: t,
    fromBackend: true,
  };
}

export function renderBatchTree() {
  const treeEl = document.getElementById('batch-tree-list');
  if (!treeEl) return;
  const doneCount = PIPELINE_BATCH_NODES.filter(
    n => n.group === 'pipeline' && n.state === 'done'
  ).length;
  const countBadge = document.getElementById('batch-count-badge');
  if (countBadge) {
    countBadge.textContent = `${doneCount}/5 完成`;
  }

  const existingNodes = treeEl.querySelectorAll('.batch-node');
  if (existingNodes.length === PIPELINE_BATCH_NODES.length) {
    let canPatchInPlace = true;
    for (let i = 0; i < PIPELINE_BATCH_NODES.length; i += 1) {
      if (existingNodes[i].getAttribute('data-batch-id') !== PIPELINE_BATCH_NODES[i].id) {
        canPatchInPlace = false;
        break;
      }
    }
    if (canPatchInPlace) {
      for (let i = 0; i < PIPELINE_BATCH_NODES.length; i += 1) {
        const n = PIPELINE_BATCH_NODES[i];
        const el = existingNodes[i];
        const stateClass = n.kind === 'step' && n.state ? `is-${n.state}` : '';
        const activeClass = n.id === state.activeBatchId ? 'active' : '';
        el.className = `batch-node ${stateClass} ${activeClass}`.trim();
        const metaEl = el.querySelector('.batch-meta');
        if (metaEl && metaEl.textContent !== n.meta) metaEl.textContent = n.meta;
      }
      return;
    }
  }

  let lastGroup = '';
  const chunks = [];
  for (const n of PIPELINE_BATCH_NODES) {
    if (n.group !== lastGroup) {
      lastGroup = n.group;
      chunks.push(
        `<div class="batch-group-label">${n.group === 'overview' ? '概览与监控' : '流水线阶段'}</div>`
      );
    }
    const activeClass = n.id === state.activeBatchId ? 'active' : '';
    const stateClass = n.kind === 'step' && n.state ? `is-${n.state}` : '';
    const dotHtml = n.kind === 'step' ? '<span class="batch-dot"></span>' : '';
    chunks.push(`
      <div class="batch-node ${stateClass} ${activeClass}" data-batch-id="${escapeAttr(n.id)}">
        <div class="batch-node-main">
          ${dotHtml}
          <span class="batch-step-num">${escapeHtml(n.step)}</span>
          <span class="batch-title">${escapeHtml(n.title)}</span>
        </div>
        <div class="batch-node-right">
          <span class="batch-meta">${escapeHtml(n.meta)}</span>
        </div>
      </div>
    `);
  }
  treeEl.innerHTML = chunks.join('');
}

export function syncStartResumeButton(card) {
  const btn = document.getElementById('btn-start-resume-task');
  const labelEl = document.getElementById('btn-start-resume-label');
  const stopBtn = document.getElementById('btn-stop-task');
  const burnBtn = document.getElementById('btn-burn-sub-video');
  const dubBtn = document.getElementById('btn-run-dubbing');
  const prevBtn = document.getElementById('btn-preview-video-modal');
  const dirBtn = document.getElementById('btn-open-task-dir');

  const hasTask = Boolean(card);
  if (burnBtn) burnBtn.disabled = !hasTask;
  if (dubBtn) dubBtn.disabled = !hasTask;
  if (prevBtn) prevBtn.disabled = !hasTask;
  if (dirBtn) dirBtn.disabled = !hasTask;

  if (!btn || !labelEl) return;
  if (!card) {
    btn.disabled = true;
    labelEl.textContent = '启动任务';
    if (stopBtn) stopBtn.disabled = true;
    syncPrimaryCtaFocus();
    return;
  }

  const rawState = card.rawState || card.state || 'idle';
  const canStop = rawState === 'running' || rawState === 'paused' || rawState === 'queued';
  if (stopBtn) stopBtn.disabled = !canStop;

  if (rawState === 'running') {
    btn.disabled = true;
    labelEl.textContent = '运行中…';
  } else if (rawState === 'stopping') {
    btn.disabled = true;
    labelEl.textContent = '停止中…';
  } else if (rawState === 'queued') {
    btn.disabled = true;
    labelEl.textContent = '排队中…';
  } else if (rawState === 'paused') {
    btn.disabled = false;
    labelEl.textContent = card.stateLabel === '待审术语' ? '确认并继续' : '继续执行';
  } else if (rawState === 'error') {
    btn.disabled = false;
    labelEl.textContent = '重试任务';
  } else if (card.state === 'completed') {
    btn.disabled = false;
    labelEl.textContent = card.stateLabel === '已完成' ? '重新执行' : '继续执行';
  } else if (card.hasAnyProgress || rawState === 'stopped') {
    btn.disabled = false;
    labelEl.textContent = '继续执行';
  } else {
    btn.disabled = false;
    labelEl.textContent = '启动任务';
  }
  syncPrimaryCtaFocus();
}

export function syncPipelineRibbon(card, uttCount = 0, termCount = 0) {
  syncStartResumeButton(card);
  const detailBadge = document.getElementById('detail-task-badge');
  const domPill = document.getElementById('detail-domain-pill');
  const resumeSubsBtn = document.getElementById('btn-save-utterances-and-continue');

  if (!card) {
    if (detailBadge) {
      detailBadge.className = 'status-label';
      detailBadge.textContent = '等待导入';
    }
    if (domPill) domPill.classList.add('hidden');
    if (resumeSubsBtn) resumeSubsBtn.classList.add('hidden');
    PIPELINE_BATCH_NODES.forEach(n => {
      n.state = 'idle';
    });
    renderBatchTree();
    return;
  }

  if (domPill) {
    domPill.textContent = card.domain || '通用';
    domPill.classList.remove('hidden');
  }

  const pillClass = pillClassForTask(card);
  if (detailBadge) {
    detailBadge.className = `status-label ${pillClass}`;
    detailBadge.textContent = card.stateLabel;
  }

  const isPausedAfterTranslate = card.rawState === 'paused' && card.stateLabel !== '待审术语';
  if (resumeSubsBtn) {
    resumeSubsBtn.classList.toggle('hidden', !isPausedAfterTranslate);
  }

  const sd = card.raw?.stages_done || {};
  const stepLower = String(card.raw?.execution?.current_step || '').toLowerCase();
  const isRunning = card.rawState === 'running';
  const isPausedTerms = card.rawState === 'paused' && card.stateLabel === '待审术语';

  const setNodeState = (id, doneCond, runCond, pauseCond = false) => {
    const node = PIPELINE_BATCH_NODES.find(n => n.id === id);
    if (!node) return;
    if (pauseCond) node.state = 'paused';
    else if (runCond) node.state = 'running';
    else if (doneCond) node.state = 'done';
    else node.state = 'idle';
  };

  setNodeState(
    'asr-01',
    Boolean(sd.asr || uttCount > 0 || ASR_SEGMENT_ROWS.length > 0),
    isRunning &&
      (stepLower.includes('transcrib') ||
        stepLower.includes('asr') ||
        stepLower.includes('audio') ||
        stepLower.includes('whisper'))
  );
  setNodeState(
    'nlp-all',
    Boolean(sd.split || uttCount > 0),
    isRunning && stepLower.includes('split') && !stepLower.includes('sub')
  );
  setNodeState(
    'terms-03',
    Boolean((sd.terms || sd.translate) && !isPausedTerms),
    isRunning && (stepLower.includes('term') || stepLower.includes('summar')),
    isPausedTerms
  );
  setNodeState(
    'step-04-group',
    Boolean(sd.translate || sd.subtitles),
    isRunning && stepLower.includes('translat')
  );
  setNodeState(
    'split-04',
    Boolean(sd.subtitles || sd.sub_video || sd.dub_video),
    isRunning && (stepLower.includes('align') || stepLower.includes('subtitle')),
    isPausedAfterTranslate
  );

  const termsNode = PIPELINE_BATCH_NODES.find(n => n.id === 'terms-03');
  if (termsNode) {
    termsNode.meta = isPausedTerms ? `${termCount} 待审` : `${termCount} 条`;
  }
  renderBatchTree();
}
