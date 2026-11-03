// =========================================================================
    // 0. API Client & State Synchronization
    // =========================================================================
    let isBackendConnected = false;
    let activeGlossaryDomain = '';
    let domainTermsMap = {};
    let domainDictEntries = [];
    let domainListMeta = [];
    let currentConfig = {};

    function isUserEditingDom() {
      const active = document.activeElement;
      if (!active) return isEditingTerms || isEditingUtterances;
      const inTerms = Boolean(active.closest?.('#terms-review-panel'));
      const inSubtitles = Boolean(active.closest?.('#structured-stream-tbody'));
      const inGlossary = Boolean(active.closest?.('#domain-csv-tbody'));
      return isEditingTerms || isEditingUtterances || inTerms || inSubtitles || inGlossary;
    }

    function reportActionError(scope, err) {
      const msg = err?.message || String(err || '操作失败');
      setAutosaveStatus('error', '操作失败');
      appendLog('warn', scope, msg, true);
    }

    async function apiFetch(url, options = {}) {
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

    function escapeHtml(str) {
      return String(str ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }

    function escapeAttr(str) {
      return escapeHtml(str).replace(/"/g, '&quot;');
    }

    function formatTimecode(sec) {
      const s = Math.max(0, Math.floor(Number(sec || 0)));
      const hh = String(Math.floor(s / 3600)).padStart(2, '0');
      const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
      const ss = String(s % 60).padStart(2, '0');
      return `${hh}:${mm}:${ss}`;
    }

    function setAutosaveStatus(state, text) {
      const saveBadge = document.getElementById('autosave-indicator');
      if (!saveBadge) return;
      saveBadge.className = `status-pill ${state}`;
      saveBadge.innerHTML = `<span class="status-dot"></span><span>${escapeHtml(text)}</span>`;
    }

    // =========================================================================
    // 1. 顶层 3 视图导航切换 & Hash 路由
    // =========================================================================
    const navButtons = document.querySelectorAll('[data-nav]');
    const views = {
      workbench: document.getElementById('view-workbench'),
      glossary: document.getElementById('view-glossary'),
      settings: document.getElementById('view-settings')
    };

    function switchView(target) {
      if (!views[target]) target = 'workbench';
      navButtons.forEach(b => b.classList.toggle('active', b.getAttribute('data-nav') === target));
      Object.entries(views).forEach(([k, el]) => {
        if (el) el.classList.toggle('hidden', k !== target);
      });
    }

    navButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const target = btn.getAttribute('data-nav');
        switchView(target);
      });
    });

    // =========================================================================
    // 2. 右侧详情槽原地切换 (Subtitle Stream vs Step 03 Terms Workbench)
    // =========================================================================
    const paneSubtitle = document.getElementById('subtitle-detail-pane');
    const paneTerms = document.getElementById('terms-review-panel');

    function syncDetailPane() {
      const isTerms = activeBatchId === 'terms-03';
      paneSubtitle.classList.toggle('hidden', isTerms);
      paneTerms.classList.toggle('hidden', !isTerms);
    }

    // =========================================================================
    // 3. 左侧任务队列轨渲染与切换
    // =========================================================================
    function updateTopRunnerBadge() {
      const badge = document.getElementById('top-runner-badge');
      if (!badge) return;
      if (REAL_TASKS.length === 0) {
        badge.innerHTML = `<span class="status-dot" style="color:var(--text-muted);"></span><span>暂无任务</span>`;
        return;
      }
      const runningCount = REAL_TASKS.filter(t => t.state === 'running').length;
      const pausedCount = REAL_TASKS.filter(t => t.state === 'paused').length;
      if (runningCount > 0) {
        badge.innerHTML = `<span class="status-dot" style="color:var(--brand-bright);"></span><span>${runningCount} 个任务运行中</span>`;
      } else if (pausedCount > 0) {
        badge.innerHTML = `<span class="status-dot" style="color:var(--amber);"></span><span>${pausedCount} 个任务待审校</span>`;
      } else {
        badge.innerHTML = `<span class="status-dot" style="color:var(--emerald-bright);"></span><span>共 ${REAL_TASKS.length} 个任务就绪</span>`;
      }
    }

    function renderTaskRail() {
      const box = document.getElementById('task-rail-list');
      document.getElementById('nav-task-count').textContent = String(REAL_TASKS.length);
      updateTopRunnerBadge();
      if (REAL_TASKS.length === 0) {
        box.innerHTML = `
          <div style="padding:28px 14px; text-align:center; color:var(--text-muted); font-size:var(--fs-xs); line-height:1.6;">
            <div>暂无任务</div>
            <div style="margin-block-start:4px;">点击右上角「新建」导入视频或字幕</div>
          </div>
        `;
        return;
      }
      box.innerHTML = REAL_TASKS.map(t => {
        const active = t.id === activeTaskId ? 'active' : '';
        const pillClass = t.state === 'completed' ? 'completed' : (t.state === 'paused' ? 'paused' : (t.state === 'error' ? 'error' : 'running'));
        return `
          <div class="task-item ${active}" data-task-id="${escapeAttr(t.id)}">
            <div class="task-item-top">
              <span class="task-item-name" title="${escapeAttr(t.title)}">${escapeHtml(t.title)}</span>
              <span class="status-pill ${pillClass}">
                <span class="status-dot"></span>
                <span>${escapeHtml(t.stateLabel)}</span>
              </span>
            </div>
            <div class="task-item-meta">
              <span class="mono">${escapeHtml(t.domain || '无领域')}</span>
              <span>${escapeHtml(t.stepSummary)}</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill ${pillClass}" style="inline-size:${t.progress}%;"></div>
            </div>
          </div>
        `;
      }).join('');

      box.querySelectorAll('[data-task-id]').forEach(card => {
        card.addEventListener('click', async () => {
          const nextId = card.getAttribute('data-task-id');
          const changed = nextId !== activeTaskId;
          activeTaskId = nextId;
          await selectActiveTask(activeTaskId, { resetEdits: changed });
        });
      });
    }

    function resetWorkbenchToEmpty() {
      isEditingTerms = false;
      isEditingUtterances = false;
      setAutosaveStatus('completed', '已就绪');
      document.getElementById('detail-task-title').textContent = '未选择任务';
      const detailBadge = document.getElementById('detail-task-badge');
      if (detailBadge) {
        detailBadge.className = 'status-pill';
        detailBadge.innerHTML = '<span class="status-dot"></span><span>等待导入</span>';
      }
      document.getElementById('detail-domain-pill').textContent = '无领域';
      REAL_TASK_TERMS.splice(0, REAL_TASK_TERMS.length);
      STREAM_TABLE_ROWS.splice(0, STREAM_TABLE_ROWS.length);
      PIPELINE_BATCH_NODES.splice(
        0,
        PIPELINE_BATCH_NODES.length,
        ...DEFAULT_PIPELINE_BATCH_NODES.map(n => ({ ...n }))
      );
      const themeInput = document.getElementById('terms-theme-input');
      if (themeInput) themeInput.value = '';
      activeBatchId = 'all';
      syncPipelineRibbon(null, 0, 0);
      renderBatchTree();
      syncDetailPane();
      renderStreamTable();
      renderTaskTerms();
    }

    async function selectActiveTask(taskId, options = {}) {
      const resetEdits = Boolean(options.resetEdits);
      if (resetEdits) {
        isEditingTerms = false;
        isEditingUtterances = false;
        setAutosaveStatus('completed', '已就绪');
      }
      const task = REAL_TASKS.find(x => x.id === taskId);
      if (!task) {
        resetWorkbenchToEmpty();
        renderTaskRail();
        return;
      }
      document.getElementById('detail-task-title').textContent = task.title;
      document.getElementById('detail-domain-pill').textContent = task.domain || '无领域';
      const domSel = document.getElementById('detail-domain-select');
      if (domSel) domSel.value = task.domain || '';
      if (task.fromBackend) {
        await loadBackendTaskDetail(task.id);
      } else {
        syncPipelineRibbon(task, STREAM_TABLE_ROWS.length, REAL_TASK_TERMS.length);
        renderBatchTree();
        syncDetailPane();
        renderStreamTable();
        renderTaskTerms();
      }
      renderTaskRail();
    }

    function mapBackendTaskToCard(t) {
      const exec = t.execution || {};
      const sd = t.stages_done || {};
      const state = exec.state || 'idle';
      const domain = t.config_override?.glossary_domain !== undefined
        ? String(t.config_override.glossary_domain)
        : (currentConfig.glossary_domain ?? 'destiny2');
      let stateLabel = '就绪';
      let cardState = 'running';
      let stepSummary = exec.current_step || '等待运行';
      let progress = Math.round((exec.progress || 0) * 100);

      if (state === 'paused') {
        cardState = 'paused';
        const reason = String(exec.pause_reason || '').toLowerCase();
        stateLabel = reason.includes('terminology') ? '待审术语' : '待校字幕';
        stepSummary = exec.pause_reason || '等待人工审校';
        progress = Math.max(progress, 45);
      } else if (state === 'running') {
        cardState = 'running';
        stateLabel = '运行中';
        stepSummary = exec.current_step || '正在处理…';
        progress = Math.max(progress, 15);
      } else if (state === 'error') {
        cardState = 'error';
        stateLabel = '异常';
        stepSummary = exec.error || '执行出错';
      } else if (sd.dub_video || sd.sub_video) {
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
      }

      return {
        id: t.task_id,
        title: t.title || t.media?.filename || t.task_id,
        domain,
        state: cardState,
        rawState: state,
        stateLabel,
        stepSummary,
        progress,
        raw: t,
        fromBackend: true,
      };
    }

    async function loadBackendTasks(options = {}) {
      const skipActiveDetail = Boolean(options.skipActiveDetail);
      try {
        const data = await apiFetch('/api/tasks');
        isBackendConnected = true;
        const prevActive = REAL_TASKS.find(t => t.id === activeTaskId);
        const prevState = prevActive?.rawState || prevActive?.state || '';
        const tasks = (data.tasks || []).map(mapBackendTaskToCard);
        REAL_TASKS.splice(0, REAL_TASKS.length, ...tasks);
        if (REAL_TASKS.length > 0) {
          if (!REAL_TASKS.some(t => t.id === activeTaskId)) {
            activeTaskId = REAL_TASKS[0].id;
          }
          renderTaskRail();
          const nextActive = REAL_TASKS.find(t => t.id === activeTaskId);
          const nextState = nextActive?.rawState || nextActive?.state || '';
          const stateChanged = prevState !== nextState;
          if (!skipActiveDetail || stateChanged || !prevActive) {
            if (!isUserEditingDom() || stateChanged) {
              await selectActiveTask(activeTaskId);
            } else if (nextActive) {
              syncPipelineRibbon(nextActive, STREAM_TABLE_ROWS.length, REAL_TASK_TERMS.length);
            }
          } else if (nextActive) {
            syncPipelineRibbon(nextActive, STREAM_TABLE_ROWS.length, REAL_TASK_TERMS.length);
          }
        } else {
          activeTaskId = '';
          renderTaskRail();
          resetWorkbenchToEmpty();
        }
      } catch (err) {
        appendLog('warn', 'TASKS', `同步任务列表失败: ${err.message}`, true);
      }
    }

    async function loadBackendTaskDetail(taskId, options = {}) {
      const preserveEdits = Boolean(options.preserveEdits);
      try {
        const [taskData, termsData, uttsData] = await Promise.all([
          apiFetch(`/api/tasks/${encodeURIComponent(taskId)}`),
          apiFetch(`/api/tasks/${encodeURIComponent(taskId)}/terms`),
          apiFetch(`/api/tasks/${encodeURIComponent(taskId)}/utterances`),
        ]);
        const card = mapBackendTaskToCard(taskData);
        const idx = REAL_TASKS.findIndex(x => x.id === taskId);
        if (idx >= 0) REAL_TASKS[idx] = card;
        if (taskId !== activeTaskId) {
          renderTaskRail();
          return;
        }

        if (!preserveEdits || !isEditingTerms) {
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
              promote: Boolean(item.promote_to_domain ?? item.promote ?? false),
            }))
          );
          const themeInput = document.getElementById('terms-theme-input');
          if (themeInput && document.activeElement !== themeInput) {
            themeInput.value = termsData.summary || '';
          }
        }

        const utts = Array.isArray(uttsData.utterances) ? uttsData.utterances : [];
        const sd = taskData.stages_done || {};
        if (!preserveEdits || !isEditingUtterances) {
          rebuildStreamRowsFromUtterances(utts, sd);
        }

        syncPipelineRibbon(card, STREAM_TABLE_ROWS.length, REAL_TASK_TERMS.length);
        if (card.state === 'paused' && card.stateLabel === '待审术语') {
          activeBatchId = 'terms-03';
        } else if (activeBatchId === 'terms-03' && REAL_TASK_TERMS.length === 0) {
          activeBatchId = 'all';
        }
        renderBatchTree();
        syncDetailPane();
        if (!preserveEdits || !isEditingUtterances) {
          renderStreamTable();
        }
        if (!preserveEdits || !isEditingTerms) {
          renderTaskTerms();
        }
      } catch (err) {
        appendLog('warn', 'TASK', `加载任务详情失败 (${taskId}): ${err.message}`, true);
      }
    }

    function rebuildStreamRowsFromUtterances(utts, sd = {}) {
      if (utts.length > 0) {
        const batchCount = Math.max(1, Math.ceil(utts.length / 15));
        const rows = utts.map((u, i) => {
          const bIdx = Math.floor(i / 15) + 1;
          const bId = `batch-${String(bIdx).padStart(2, '0')}`;
          const numStr = `#${String(i + 1).padStart(3, '0')}`;
          const tcStr = `${formatTimecode(u.start)} → ${formatTimecode(u.end)}`;
          const hasCues = Array.isArray(u.sub_cues) && u.sub_cues.length > 1;
          const srcText = String(u.src ?? '');
          const tgtText = String(u.tgt ?? '');
          const ttsText = String(u.tts_text ?? tgtText);
          return {
            idx: numStr,
            stage: hasCues ? 'split' : 'translate',
            batchId: bId,
            isSplit: hasCues,
            tc: tcStr,
            meta: hasCues ? `B${String(bIdx).padStart(2, '0')} · 拆为 ${u.sub_cues.length} 轴` : `B${String(bIdx).padStart(2, '0')}`,
            srcText,
            tgtText,
            ttsText,
            srcHtml: escapeHtml(srcText),
            tgtHtml: escapeHtml(tgtText),
            uttIndex: i,
            rawUtt: { ...u, src: srcText, tgt: tgtText, tts_text: ttsText },
          };
        });
        STREAM_TABLE_ROWS.splice(0, STREAM_TABLE_ROWS.length, ...rows);

        const baseNodes = [
          { id: 'all', kind: 'summary', step: 'ALL', title: '全部字幕', meta: `${utts.length} 句`, time: '', desc: `全部字幕 (${utts.length} 句)` },
          { id: 'asr-01', kind: 'step', step: '01', title: '语音识别', meta: sd.asr ? '已完成' : '未开始', time: '', desc: '语音识别' },
          { id: 'nlp-all', kind: 'step', step: '02', title: '标点与断句', meta: `${utts.length} 句`, time: '', desc: `标点与断句 · ${utts.length} 句` },
          { id: 'terms-03', kind: 'step', step: '03', title: '术语审校', meta: `${REAL_TASK_TERMS.length} 条`, time: '', desc: '术语审校' },
          { id: 'step-04-group', kind: 'group', step: '04', title: '字幕翻译', meta: `${batchCount} 批`, time: '', desc: `字幕翻译 · ${batchCount} 批` },
        ];
        for (let b = 1; b <= batchCount; b++) {
          const bCode = String(b).padStart(2, '0');
          const sIdx = String((b - 1) * 15 + 1).padStart(3, '0');
          const eIdx = String(Math.min(b * 15, utts.length)).padStart(3, '0');
          const cnt = Math.min(b * 15, utts.length) - (b - 1) * 15;
          baseNodes.push({
            id: `batch-${bCode}`,
            kind: 'child',
            step: `B${bCode}`,
            title: `#${sIdx}–${eIdx}`,
            meta: `${cnt} 句`,
            time: '',
            desc: `第 ${sIdx}–${eIdx} 句`,
          });
        }
        const splitRows = rows.filter(r => r.isSplit);
        if (splitRows.length > 0) {
          baseNodes.push({
            id: 'split-04',
            kind: 'child',
            step: '切行',
            title: '长句拆行',
            meta: `${splitRows.length} 句`,
            time: '',
            desc: `长句拆行 (${splitRows.map(r => r.idx).slice(0, 3).join(', ')})`,
          });
        }
        PIPELINE_BATCH_NODES.splice(0, PIPELINE_BATCH_NODES.length, ...baseNodes);
        if (!PIPELINE_BATCH_NODES.some(n => n.id === activeBatchId)) {
          activeBatchId = 'all';
        }
      } else {
        STREAM_TABLE_ROWS.splice(0, STREAM_TABLE_ROWS.length);
        PIPELINE_BATCH_NODES.splice(
          0,
          PIPELINE_BATCH_NODES.length,
          { id: 'all', kind: 'summary', step: 'ALL', title: '全部字幕', meta: '0 句', time: '', desc: '全部字幕 (0 句)' },
          { id: 'asr-01', kind: 'step', step: '01', title: '语音识别', meta: sd.asr ? '已完成' : '未开始', time: '', desc: '语音识别' },
          { id: 'nlp-all', kind: 'step', step: '02', title: '标点与断句', meta: '0 句', time: '', desc: '标点与断句' },
          { id: 'terms-03', kind: 'step', step: '03', title: '术语审校', meta: `${REAL_TASK_TERMS.length} 条`, time: '', desc: '术语审校' },
          { id: 'step-04-group', kind: 'group', step: '04', title: '字幕翻译', meta: '0 批', time: '', desc: '字幕翻译' }
        );
        if (!PIPELINE_BATCH_NODES.some(n => n.id === activeBatchId)) {
          activeBatchId = 'all';
        }
      }
    }

    // Top-layer dialog exit animation & light-dismiss helper
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

    async function closeTopLayer(dialogEl) {
      if (!dialogEl || !dialogEl.open) return;
      if (!supportsTopLayerExit()) {
        dialogEl.setAttribute('data-closing', '');
        const animations = dialogEl.getAnimations({ subtree: true });
        if (animations.length > 0) {
          await Promise.race([
            Promise.allSettled(animations.map(a => a.finished)),
            new Promise(r => setTimeout(r, 400))
          ]);
        }
        if (!dialogEl.hasAttribute('data-closing')) return;
        dialogEl.removeAttribute('data-closing');
      }
      dialogEl.close();
    }

    function wireModalDialog(dialogEl) {
      if (!dialogEl) return;
      dialogEl.addEventListener('cancel', event => {
        event.preventDefault();
        closeTopLayer(dialogEl);
      });
      if (!('closedBy' in HTMLDialogElement.prototype)) {
        dialogEl.addEventListener('click', event => {
          if (event.target !== dialogEl) return;
          const rect = dialogEl.getBoundingClientRect();
          const isDialogContent = (
            rect.top <= event.clientY &&
            event.clientY <= rect.top + rect.height &&
            rect.left <= event.clientX &&
            event.clientX <= rect.left + rect.width
          );
          if (isDialogContent) return;
          closeTopLayer(dialogEl);
        });
      }
    }

    // 新建任务弹窗
    const newTaskDialog = document.getElementById('new-task-dialog');
    const dropWell = document.getElementById('upload-dropzone');
    wireModalDialog(newTaskDialog);

    function closeAllNtDropdowns(except = null) {
      newTaskDialog.querySelectorAll('.nt-dropdown.open').forEach(dd => {
        if (dd !== except) {
          dd.classList.remove('open');
          dd.querySelector('.nt-select-trigger')?.setAttribute('aria-expanded', 'false');
        }
      });
    }

    function wireNtDropdown(dropdownEl, onSelect) {
      const trigger = dropdownEl.querySelector('.nt-select-trigger');
      const menu = dropdownEl.querySelector('.nt-menu');
      trigger.addEventListener('click', e => {
        e.stopPropagation();
        const willOpen = !dropdownEl.classList.contains('open');
        closeAllNtDropdowns(dropdownEl);
        dropdownEl.classList.toggle('open', willOpen);
        trigger.setAttribute('aria-expanded', String(willOpen));
      });
      menu.addEventListener('click', e => {
        const item = e.target.closest('.nt-menu-item');
        if (!item) return;
        menu.querySelectorAll('.nt-menu-item').forEach(btn => {
          btn.classList.toggle('active', btn === item);
          btn.setAttribute('aria-selected', String(btn === item));
        });
        onSelect(item);
        dropdownEl.classList.remove('open');
        trigger.setAttribute('aria-expanded', 'false');
      });
    }

    wireNtDropdown(document.getElementById('nt-src-lang-dropdown'), item => {
      document.getElementById('task-src-lang').value = item.dataset.value || 'auto';
      document.getElementById('nt-src-lang-label').textContent = item.dataset.label || item.textContent.trim();
    });

    wireNtDropdown(document.getElementById('nt-domain-dropdown'), item => {
      const val = item.dataset.value ?? '';
      const sub = item.dataset.sub || '';
      document.getElementById('task-domain').value = val;
      const slugEl = document.getElementById('nt-domain-slug');
      slugEl.textContent = val || '不挂载词表';
      slugEl.classList.toggle('mono', Boolean(val));
      document.getElementById('nt-domain-sub').textContent = sub;
    });

    newTaskDialog.addEventListener('click', e => {
      if (!e.target.closest('.nt-dropdown')) closeAllNtDropdowns();
    });

    let droppedFile = null;
    let droppedSubFile = null;
    const localVideoFileInput = document.createElement('input');
    localVideoFileInput.type = 'file';
    localVideoFileInput.accept = '.mp4,.mov,.mkv,.avi,.flv,.wmv,.webm,.wav,.mp3,.flac,.m4a,.srt';
    localVideoFileInput.style.display = 'none';
    document.body.appendChild(localVideoFileInput);

    function formatFileSize(bytes) {
      const b = Number(bytes || 0);
      if (b >= 1024 * 1024 * 1024) return `${(b / (1024 * 1024 * 1024)).toFixed(2)} GB`;
      if (b >= 1024 * 1024) return `${(b / (1024 * 1024)).toFixed(1)} MB`;
      if (b >= 1024) return `${(b / 1024).toFixed(0)} KB`;
      return `${b} B`;
    }

    function selectBrowserFile(file) {
      if (!file) return;
      droppedFile = file;
      const label = `${file.name} (${formatFileSize(file.size)})`;
      const inputEl = document.getElementById('input-source-path');
      inputEl.value = label;
      inputEl.dataset.droppedLabel = label;
      dropWell.classList.add('has-file');
    }

    function selectSubtitleFile(file) {
      if (!file) return;
      droppedSubFile = file;
      const label = `${file.name} (${formatFileSize(file.size)})`;
      const subEl = document.getElementById('input-sub-path');
      if (subEl) {
        subEl.value = label;
        subEl.dataset.droppedLabel = label;
      }
    }

    function handleDroppedFiles(fileList) {
      const files = Array.from(fileList || []);
      if (!files.length) return;
      if (files.length >= 2) {
        const srt = files.find(f => f.name.toLowerCase().endsWith('.srt'));
        const media = files.find(f => !f.name.toLowerCase().endsWith('.srt')) || files[0];
        selectBrowserFile(media);
        if (srt && srt !== media) selectSubtitleFile(srt);
        return;
      }
      const single = files[0];
      const sourceInput = document.getElementById('input-source-path');
      if (single.name.toLowerCase().endsWith('.srt') && (droppedFile || sourceInput.value.trim())) {
        selectSubtitleFile(single);
      } else {
        selectBrowserFile(single);
      }
    }

    document.getElementById('input-source-path')?.addEventListener('input', e => {
      const el = e.currentTarget;
      if (droppedFile && el.value.trim() !== el.dataset.droppedLabel && el.value.trim() !== droppedFile.name) {
        droppedFile = null;
        delete el.dataset.droppedLabel;
        dropWell.classList.remove('has-file');
      }
    });

    document.getElementById('input-sub-path')?.addEventListener('input', e => {
      const el = e.currentTarget;
      if (droppedSubFile && el.value.trim() !== el.dataset.droppedLabel && el.value.trim() !== droppedSubFile.name) {
        droppedSubFile = null;
        delete el.dataset.droppedLabel;
      }
    });

    localVideoFileInput.addEventListener('change', () => {
      if (localVideoFileInput.files?.length) {
        selectBrowserFile(localVideoFileInput.files[0]);
      }
      localVideoFileInput.value = '';
    });

    // Prevent browser from navigating away if a file is dropped anywhere on the page or modal
    window.addEventListener('dragover', e => {
      if (e.dataTransfer?.types?.includes('Files')) {
        e.preventDefault();
      }
    });
    window.addEventListener('drop', e => {
      if (!e.dataTransfer?.files?.length) return;
      e.preventDefault();
      dropWell.classList.remove('dragover');
      if (!newTaskDialog.open) {
        closeAllNtDropdowns();
        newTaskDialog.showModal();
      }
      handleDroppedFiles(e.dataTransfer.files);
    });

    ['dragenter', 'dragover'].forEach(ev => {
      newTaskDialog.addEventListener(ev, e => {
        if (e.dataTransfer?.types?.includes('Files')) {
          e.preventDefault();
          dropWell.classList.add('dragover');
        }
      });
    });
    newTaskDialog.addEventListener('dragleave', e => {
      if (!newTaskDialog.contains(e.relatedTarget)) {
        dropWell.classList.remove('dragover');
      }
    });
    newTaskDialog.addEventListener('drop', e => {
      if (!e.dataTransfer?.files?.length) return;
      e.preventDefault();
      e.stopPropagation();
      dropWell.classList.remove('dragover');
      if (e.target.closest('.nt-affix')) {
        selectSubtitleFile(e.dataTransfer.files[0]);
      } else {
        handleDroppedFiles(e.dataTransfer.files);
      }
    });

    document.getElementById('btn-pick-local')?.addEventListener('click', async () => {
      if (isBackendConnected) {
        try {
          const res = await apiFetch('/api/dialogs/pick-files', { method: 'POST' });
          if (res.paths && res.paths.length > 0) {
            droppedFile = null;
            const inputEl = document.getElementById('input-source-path');
            delete inputEl.dataset.droppedLabel;
            dropWell.classList.remove('has-file');
            inputEl.value = res.paths[0];
            return;
          }
          if (Array.isArray(res.paths)) return;
        } catch (err) {
          appendLog('warn', 'FILE', `调用系统文件选择器失败，已切换为浏览器文件选择: ${err.message}`, true);
        }
      }
      localVideoFileInput.click();
    });

    document.getElementById('btn-toggle-new-task').addEventListener('click', () => {
      closeAllNtDropdowns();
      newTaskDialog.showModal();
    });
    document.getElementById('btn-close-intake').addEventListener('click', () => {
      closeTopLayer(newTaskDialog);
    });
    document.getElementById('btn-cancel-intake').addEventListener('click', () => {
      closeTopLayer(newTaskDialog);
    });

    document.getElementById('btn-open-glossary-center').addEventListener('click', () => {
      closeTopLayer(newTaskDialog);
      document.getElementById('tab-glossary').click();
    });

    document.getElementById('btn-create-task').addEventListener('click', async e => {
      const createBtn = e.currentTarget;
      const origBtnText = createBtn.textContent;
      const sourceInput = document.getElementById('input-source-path');
      const subInput = document.getElementById('input-sub-path');
      const rawPath = sourceInput.value.trim();
      if (!rawPath && !droppedFile) {
        sourceInput.focus();
        appendLog('warn', 'WARN', '请先选择本地媒体文件或输入有效的视频路径 / YouTube 链接', true);
        return;
      }
      const subPath = subInput.value.trim() || null;
      const srcLang = document.getElementById('task-src-lang').value || 'en';
      const tgtLang = document.getElementById('task-tgt-lang').value.trim() || '简体中文';
      const domain = document.getElementById('task-domain').value;
      const scopeRadio = document.querySelector('input[name="task-scope"]:checked');
      const stage = scopeRadio ? scopeRadio.value : 'subtitles';
      const pauseBefore = document.getElementById('task-pause-terms-chk').checked;

      if (isBackendConnected) {
        createBtn.disabled = true;
        try {
          await apiFetch('/api/config', {
            method: 'PUT',
            body: JSON.stringify({ updates: { pause_before_translate: pauseBefore } }),
          });
          let created = null;
          if (droppedFile) {
            createBtn.textContent = '正在上传…';
            const form = new FormData();
            form.append('file', droppedFile);
            if (droppedSubFile) {
              form.append('subtitles', droppedSubFile);
            } else if (subPath) {
              form.append('subtitles_path', subPath);
            }
            form.append('source_language', srcLang);
            form.append('target_language', tgtLang);
            form.append('glossary_domain', domain);
            form.append('stage', stage);
            form.append('autorun', 'true');
            created = await apiFetch('/api/tasks/upload', { method: 'POST', body: form });
            droppedFile = null;
            droppedSubFile = null;
            delete sourceInput.dataset.droppedLabel;
            delete subInput.dataset.droppedLabel;
            dropWell.classList.remove('has-file');
          } else {
            createBtn.textContent = '创建中…';
            created = await apiFetch('/api/tasks', {
              method: 'POST',
              body: JSON.stringify({
                source: rawPath,
                subtitles: subPath,
                source_language: srcLang,
                target_language: tgtLang,
                glossary_domain: domain,
                stage,
                dubbing: stage === 'all' || stage === 'dubbing',
                autorun: true,
              }),
            });
          }
          if (created && created.task_id) {
            activeTaskId = created.task_id;
            sourceInput.value = '';
            subInput.value = '';
            closeTopLayer(newTaskDialog);
            appendLog('asr', 'ASR', `Created workspace output/${created.task_id}/ and started pipeline...`, false);
            await loadBackendTasks();
            return;
          }
        } catch (err) {
          appendLog('warn', 'WARN', `任务创建失败: ${err.message}`, true);
          return;
        } finally {
          createBtn.disabled = false;
          createBtn.textContent = origBtnText;
        }
      }

      const fileName = rawPath.split('/').pop() || rawPath;
      const id = fileName.replace(/\.[^.]+$/, '');
      REAL_TASKS.unshift({
        id,
        title: fileName,
        domain,
        state: 'running',
        stateLabel: '运行中',
        stepSummary: '正在语音识别…',
        progress: 18
      });
      activeTaskId = id;
      document.getElementById('detail-task-title').textContent = fileName;
      document.getElementById('detail-domain-pill').textContent = domain || '无领域';
      closeTopLayer(newTaskDialog);
      renderTaskRail();
      appendLog('asr', 'ASR', `Created workspace output/${id}/ and started pipeline...`, false);
    });

    // =========================================================================
    // 4. Master-Detail 流水线批次树 + 双语字幕表格渲染与过滤
    // =========================================================================
    function renderBatchTree() {
      const treeEl = document.getElementById('batch-tree-list');

      treeEl.innerHTML = PIPELINE_BATCH_NODES.map(n => {
        const kindClass = n.kind ? `is-${n.kind}` : '';
        return `
          <div class="batch-node ${kindClass} ${n.id === activeBatchId ? 'active' : ''}" data-batch-id="${escapeAttr(n.id)}" title="${escapeAttr(n.desc)}">
            <div class="batch-node-main">
              <span class="batch-step-num">${escapeHtml(n.step)}</span>
              <span class="batch-title">${escapeHtml(n.title)}</span>
            </div>
            <div class="batch-node-right">
              <span class="batch-meta">${escapeHtml(n.meta)}</span>
              <span class="batch-time">${escapeHtml(n.time)}</span>
            </div>
          </div>
        `;
      }).join('');

      const current = PIPELINE_BATCH_NODES.find(n => n.id === activeBatchId) || PIPELINE_BATCH_NODES[0];
      const headingEl = document.getElementById('active-batch-heading');
      const tagEl = document.getElementById('active-batch-tag');
      if (headingEl && current) headingEl.textContent = current.desc;
      if (tagEl && current) tagEl.textContent = current.time;

      treeEl.querySelectorAll('[data-batch-id]').forEach(node => {
        node.addEventListener('click', () => {
          activeBatchId = node.getAttribute('data-batch-id');
          renderBatchTree();
          syncDetailPane();
          if (activeBatchId !== 'terms-03') {
            renderStreamTable();
          }
        });
      });
    }

    function calcWeightedLen(text) {
      let len = 0;
      for (const ch of String(text || '')) {
        len += /[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/.test(ch) ? 2 : 1;
      }
      return len;
    }

    function splitTextNearMiddle(text) {
      const s = String(text || '').trim();
      if (!s) return ['', ''];
      const words = s.split(/\s+/);
      if (words.length >= 2) {
        const mid = Math.ceil(words.length / 2);
        return [words.slice(0, mid).join(' '), words.slice(mid).join(' ')];
      }
      const chars = Array.from(s);
      if (chars.length >= 2) {
        const mid = Math.ceil(chars.length / 2);
        return [chars.slice(0, mid).join(''), chars.slice(mid).join('')];
      }
      return [s, ''];
    }

    function splitStreamRowAt(uttIdx) {
      const card = REAL_TASKS.find(t => t.id === activeTaskId);
      const sd = card?.raw?.stages_done || {};
      const utts = STREAM_TABLE_ROWS.map(r => ({ ...(r.rawUtt || {}), src: r.srcText ?? r.rawUtt?.src ?? '', tgt: r.tgtText ?? r.rawUtt?.tgt ?? '', tts_text: r.ttsText ?? r.rawUtt?.tts_text ?? r.tgtText ?? '' }));
      const target = utts[uttIdx];
      if (!target) return;
      const [srcA, srcB] = splitTextNearMiddle(target.src);
      const [tgtA, tgtB] = splitTextNearMiddle(target.tgt);
      const [ttsA, ttsB] = splitTextNearMiddle(target.tts_text || target.tgt);
      const start = Number(target.start || 0);
      const end = Math.max(start, Number(target.end || start));
      const midTime = Number(((start + end) / 2).toFixed(3));
      const span = Array.isArray(target.word_span) && target.word_span.length >= 2 ? target.word_span : [0, 0];
      const wStart = Number(span[0] || 0);
      const wEnd = Math.max(wStart, Number(span[1] || wStart));
      const spanLen = wEnd - wStart;
      let spanA = [wStart, wEnd];
      let spanB = [wStart, wEnd];
      if (spanLen >= 2) {
        const wordsA = String(srcA || '').trim().split(/\s+/).filter(Boolean).length || 1;
        const wordsB = String(srcB || '').trim().split(/\s+/).filter(Boolean).length || 1;
        const ratio = wordsA / Math.max(1, wordsA + wordsB);
        const wMid = Math.min(wEnd - 1, Math.max(wStart + 1, wStart + Math.round(spanLen * ratio)));
        spanA = [wStart, wMid];
        spanB = [wMid, wEnd];
      }

      const firstUtt = {
        ...target,
        id: `${target.id || `u_${uttIdx}`}_a`,
        start,
        end: midTime,
        word_span: spanA,
        src: srcA,
        tgt: tgtA,
        tts_text: ttsA || tgtA,
        sub_cues: [{ src: srcA, tgt: tgtA, trans: tgtA, start, end: midTime, word_span: spanA }],
      };
      const secondUtt = {
        ...target,
        id: `${target.id || `u_${uttIdx}`}_b`,
        start: midTime,
        end,
        word_span: spanB,
        src: srcB,
        tgt: tgtB,
        tts_text: ttsB || tgtB,
        sub_cues: [{ src: srcB, tgt: tgtB, trans: tgtB, start: midTime, end, word_span: spanB }],
      };
      utts.splice(uttIdx, 1, firstUtt, secondUtt);
      isEditingUtterances = true;
      setAutosaveStatus('paused', '字幕未保存');
      rebuildStreamRowsFromUtterances(utts, sd);
      renderBatchTree();
      renderStreamTable();
    }

    function mergeStreamRowWithNext(uttIdx) {
      const card = REAL_TASKS.find(t => t.id === activeTaskId);
      const sd = card?.raw?.stages_done || {};
      const utts = STREAM_TABLE_ROWS.map(r => ({ ...(r.rawUtt || {}), src: r.srcText ?? r.rawUtt?.src ?? '', tgt: r.tgtText ?? r.rawUtt?.tgt ?? '', tts_text: r.ttsText ?? r.rawUtt?.tts_text ?? r.tgtText ?? '' }));
      if (uttIdx < 0 || uttIdx >= utts.length - 1) return;
      const cur = utts[uttIdx];
      const nxt = utts[uttIdx + 1];
      const joinSrc = [String(cur.src || '').trim(), String(nxt.src || '').trim()].filter(Boolean).join(' ');
      const joinTgt = [String(cur.tgt || '').trim(), String(nxt.tgt || '').trim()].filter(Boolean).join(' ');
      const joinTts = [String(cur.tts_text || cur.tgt || '').trim(), String(nxt.tts_text || nxt.tgt || '').trim()].filter(Boolean).join(' ');
      const start = Number(cur.start || 0);
      const end = Math.max(start, Number(nxt.end || cur.end || start));
      const spanA = Array.isArray(cur.word_span) && cur.word_span.length >= 2 ? cur.word_span[0] : 0;
      const spanB = Array.isArray(nxt.word_span) && nxt.word_span.length >= 2 ? nxt.word_span[1] : spanA;
      const merged = {
        ...cur,
        start,
        end,
        word_span: [spanA, spanB],
        src: joinSrc,
        tgt: joinTgt,
        tts_text: joinTts || joinTgt,
        sub_cues: [{ src: joinSrc, tgt: joinTgt, trans: joinTgt, start, end, word_span: [spanA, spanB] }],
      };
      utts.splice(uttIdx, 2, merged);
      isEditingUtterances = true;
      setAutosaveStatus('paused', '字幕未保存');
      rebuildStreamRowsFromUtterances(utts, sd);
      renderBatchTree();
      renderStreamTable();
    }

    async function persistTaskUtterancesToBackend() {
      const task = REAL_TASKS.find(x => x.id === activeTaskId);
      if (!isBackendConnected || !task?.fromBackend) {
        isEditingUtterances = false;
        setAutosaveStatus('completed', '已保存');
        return;
      }
      const utterancesPayload = STREAM_TABLE_ROWS.map((r, i) => {
        const base = r.rawUtt || {};
        const src = String(r.srcText ?? base.src ?? '').trim();
        const tgt = String(r.tgtText ?? base.tgt ?? '').trim();
        const tts_text = String(r.ttsText ?? base.tts_text ?? tgt).trim() || tgt;
        return {
          ...base,
          index: i,
          src,
          tgt,
          tts_text,
        };
      });
      await apiFetch(`/api/tasks/${encodeURIComponent(activeTaskId)}/utterances`, {
        method: 'PUT',
        body: JSON.stringify({ utterances: utterancesPayload }),
      });
      isEditingUtterances = false;
      setAutosaveStatus('completed', '字幕已保存');
    }

    function renderStreamTable() {
      const tbody = document.getElementById('structured-stream-tbody');
      const query = (document.getElementById('subtitle-stream-search').value || '').trim().toLowerCase();

      const filtered = STREAM_TABLE_ROWS.filter(r => {
        if (activeBatchId === 'step-04-group') {
          if (r.stage !== 'translate' && r.stage !== 'split') return false;
        } else if (activeBatchId === 'split-04') {
          if (r.batchId !== 'split-04' && !r.isSplit) return false;
        } else if (activeBatchId !== 'all' && r.batchId !== activeBatchId) {
          return false;
        }
        if (!query) return true;
        const srcPlain = String(r.srcText ?? r.srcHtml ?? '').toLowerCase();
        const tgtPlain = String(r.tgtText ?? r.tgtHtml ?? '').toLowerCase();
        const ttsPlain = String(r.ttsText ?? '').toLowerCase();
        return (
          r.idx.toLowerCase().includes(query) ||
          srcPlain.includes(query) ||
          tgtPlain.includes(query) ||
          ttsPlain.includes(query) ||
          r.meta.toLowerCase().includes(query)
        );
      });

      document.getElementById('stream-row-counter').textContent = query
        ? `${filtered.length} / ${STREAM_TABLE_ROWS.length} 句`
        : `共 ${filtered.length} 句`;

      if (filtered.length === 0) {
        const emptyHint = REAL_TASKS.length === 0
          ? '当前暂无任务，请点击左上角「新建」导入视频或字幕文件'
          : (STREAM_TABLE_ROWS.length === 0 ? '当前阶段尚未生成字幕数据' : '未匹配到符合搜索条件的字幕行');
        tbody.innerHTML = `
          <tr>
            <td colspan="4" style="padding:36px 16px; text-align:center; color:var(--text-muted); font-size:var(--fs-xs);">
              ${escapeHtml(emptyHint)}
            </td>
          </tr>
        `;
        return;
      }

      const maxLenCfg = Number(currentConfig?.subtitle?.max_length || 75);
      const tgtMultCfg = Number(currentConfig?.subtitle?.target_multiplier || 1.2);

      tbody.innerHTML = filtered.map(r => {
        const uIdx = Number(r.uttIndex ?? 0);
        const srcVal = r.srcText !== undefined ? r.srcText : (r.rawUtt?.src ?? '');
        const tgtVal = r.tgtText !== undefined ? r.tgtText : (r.rawUtt?.tgt ?? '');
        const ttsVal = r.ttsText !== undefined ? r.ttsText : (r.rawUtt?.tts_text ?? tgtVal);
        const weightedTgt = Math.round(calcWeightedLen(tgtVal) * tgtMultCfg);
        const isOverLen = weightedTgt > maxLenCfg;
        const canMergeNext = uIdx < STREAM_TABLE_ROWS.length - 1;
        return `
        <tr data-utt-row="${uIdx}">
          <td>
            <span class="sub-idx">${r.idx}</span>
            <div class="row" style="gap:3px; margin-block-start:6px; flex-wrap:wrap;">
              <button type="button" class="xs" style="padding:0 5px; block-size:20px; font-size:10px;" data-split-utt="${uIdx}" title="将该句从中间拆分为两句">拆行</button>
              ${canMergeNext ? `<button type="button" class="xs" style="padding:0 5px; block-size:20px; font-size:10px;" data-merge-utt="${uIdx}" title="与下一句合并">合并</button>` : ''}
            </div>
          </td>
          <td>
            <div class="sub-tc">
              <span>${r.tc}</span>
              <span class="field-hint">${r.meta}</span>
              <span class="field-hint mono" data-utt-len-badge="${uIdx}" style="color:${isOverLen ? 'var(--rose)' : 'var(--text-muted)'};">${weightedTgt}/${maxLenCfg}</span>
            </div>
          </td>
          <td class="sub-src">
            <input type="text" class="cell-input" value="${escapeAttr(srcVal)}" placeholder="原文字幕" data-utt-field="src" data-idx="${uIdx}">
          </td>
          <td class="sub-tgt">
            <input type="text" class="cell-input" style="font-weight:600;" value="${escapeAttr(tgtVal)}" placeholder="译文字幕" data-utt-field="tgt" data-idx="${uIdx}">
            <div class="row" style="gap:6px; margin-block-start:4px;">
              <span class="field-hint mono" style="flex-shrink:0;">TTS</span>
              <input type="text" class="cell-input" style="block-size:24px; font-size:var(--fs-xs); color:var(--text-secondary);" value="${escapeAttr(ttsVal)}" placeholder="配音口播词（默认同译文）" data-utt-field="tts_text" data-idx="${uIdx}">
            </div>
          </td>
        </tr>
      `;
      }).join('');

      tbody.querySelectorAll('[data-utt-field]').forEach(inp => {
        inp.addEventListener('input', () => {
          const idx = Number(inp.getAttribute('data-idx'));
          const field = inp.getAttribute('data-utt-field');
          const row = STREAM_TABLE_ROWS[idx];
          if (!row) return;
          isEditingUtterances = true;
          setAutosaveStatus('paused', '字幕未保存');
          if (field === 'src') {
            row.srcText = inp.value;
            row.srcHtml = escapeHtml(inp.value);
            if (row.rawUtt) row.rawUtt.src = inp.value;
          } else if (field === 'tgt') {
            const prevTgt = row.tgtText;
            row.tgtText = inp.value;
            row.tgtHtml = escapeHtml(inp.value);
            if (row.rawUtt) row.rawUtt.tgt = inp.value;
            if (!row.ttsText || row.ttsText === prevTgt) {
              row.ttsText = inp.value;
              if (row.rawUtt) row.rawUtt.tts_text = inp.value;
              const ttsInp = tbody.querySelector(`input[data-utt-field="tts_text"][data-idx="${idx}"]`);
              if (ttsInp && document.activeElement !== ttsInp) {
                ttsInp.value = inp.value;
              }
            }
            const badge = tbody.querySelector(`[data-utt-len-badge="${idx}"]`);
            if (badge) {
              const wLen = Math.round(calcWeightedLen(inp.value) * tgtMultCfg);
              badge.textContent = `${wLen}/${maxLenCfg}`;
              badge.style.color = wLen > maxLenCfg ? 'var(--rose)' : 'var(--text-muted)';
            }
          } else if (field === 'tts_text') {
            row.ttsText = inp.value;
            if (row.rawUtt) row.rawUtt.tts_text = inp.value;
          }
        });
      });

      tbody.querySelectorAll('[data-split-utt]').forEach(btn => {
        btn.addEventListener('click', () => {
          splitStreamRowAt(Number(btn.getAttribute('data-split-utt')));
        });
      });

      tbody.querySelectorAll('[data-merge-utt]').forEach(btn => {
        btn.addEventListener('click', () => {
          mergeStreamRowWithNext(Number(btn.getAttribute('data-merge-utt')));
        });
      });
    }

    document.getElementById('subtitle-stream-search').addEventListener('input', renderStreamTable);

    document.getElementById('btn-save-utterances')?.addEventListener('click', async () => {
      try {
        await persistTaskUtterancesToBackend();
        appendLog('llm', 'SUBS', `Saved ${STREAM_TABLE_ROWS.length} utterances to project.sqlite.`, false);
      } catch (err) {
        reportActionError('SUBS', err);
      }
    });

    document.getElementById('btn-save-utterances-and-continue')?.addEventListener('click', async () => {
      const task = REAL_TASKS.find(x => x.id === activeTaskId);
      try {
        await persistTaskUtterancesToBackend();
        if (isBackendConnected && task?.fromBackend) {
          if (task.rawState === 'paused') {
            await apiFetch(`/api/tasks/${encodeURIComponent(activeTaskId)}/resume`, { method: 'POST' });
          } else {
            await apiFetch(`/api/tasks/${encodeURIComponent(activeTaskId)}/invalidate`, {
              method: 'POST',
              body: JSON.stringify({ stage: 'sub_split', autorun: true, run_stage: 'subtitles' }),
            });
          }
          await loadBackendTasks();
        }
        appendLog('llm', 'SUBS', `Saved ${STREAM_TABLE_ROWS.length} utterances and resumed subtitle pipeline.`, false);
      } catch (err) {
        reportActionError('SUBS', err);
      }
    });

    // =========================================================================
    // 5. 五阶段流水线进度状态同步
    // =========================================================================
    const detailBadge = document.getElementById('detail-task-badge');
    const pill1 = document.getElementById('pill-stage-1');
    const pill2 = document.getElementById('pill-stage-2');
    const pill3 = document.getElementById('pill-stage-3');
    const pill4 = document.getElementById('pill-stage-4');
    const pill5 = document.getElementById('pill-stage-5');
    const line1 = document.getElementById('pipe-line-1');
    const line2 = document.getElementById('pipe-line-2');
    const line3 = document.getElementById('pipe-line-3');
    const line4 = document.getElementById('pipe-line-4');

    function syncPipelineRibbon(card, uttCount = 0, termCount = 0) {
      const b1 = document.getElementById('pill-badge-1');
      const b2 = document.getElementById('pill-badge-2');
      const b3 = document.getElementById('pill-badge-3');
      const b4 = document.getElementById('pill-badge-4');
      const b5 = document.getElementById('pill-badge-5');

      if (!card) {
        [pill1, pill2, pill3, pill4, pill5].forEach(el => { if (el) el.className = 'pipe-node'; });
        [line1, line2, line3, line4].forEach(el => { if (el) el.className = 'pipe-line'; });
        if (b1) b1.textContent = '未开始';
        if (b2) b2.textContent = '0句';
        if (b3) b3.textContent = '0条';
        if (b4) b4.textContent = '0批';
        if (b5) b5.textContent = '待合成';
        return;
      }

      const sd = card.raw?.stages_done || {};
      const pillClass = card.state === 'completed' ? 'completed' : (card.state === 'paused' ? 'paused' : (card.state === 'error' ? 'error' : 'running'));
      if (detailBadge) {
        detailBadge.className = `status-pill ${pillClass}`;
        detailBadge.innerHTML = `<span class="status-dot"></span><span>${escapeHtml(card.stateLabel || '就绪')}</span>`;
      }

      if (pill1) pill1.className = sd.asr ? 'pipe-node done' : (card.state === 'running' ? 'pipe-node active' : 'pipe-node');
      if (b1) b1.textContent = sd.asr ? '已完成' : (card.state === 'running' ? '识别中' : '未开始');
      if (line1) line1.className = sd.asr ? 'pipe-line done' : 'pipe-line';

      if (pill2) pill2.className = sd.split ? 'pipe-node done' : 'pipe-node';
      if (b2) b2.textContent = `${uttCount}句`;
      if (line2) line2.className = sd.split ? 'pipe-line done' : 'pipe-line';

      if (card.state === 'paused' && card.stateLabel === '待审术语') {
        if (pill3) pill3.className = 'pipe-node paused';
        if (b3) b3.textContent = '待确认';
        if (line3) line3.className = 'pipe-line';
        if (pill4) pill4.className = 'pipe-node';
        if (b4) b4.textContent = '等待中';
      } else {
        if (pill3) pill3.className = sd.terms ? 'pipe-node done' : 'pipe-node';
        if (b3) b3.textContent = `${termCount}条`;
        if (line3) line3.className = sd.terms ? 'pipe-line done' : 'pipe-line';

        const batchCount = uttCount > 0 ? Math.max(1, Math.ceil(uttCount / 15)) : 0;
        const transDone = Boolean(sd.translate || sd.subtitles);
        if (pill4) pill4.className = transDone ? 'pipe-node done' : 'pipe-node';
        if (b4) b4.textContent = transDone && batchCount > 0 ? `${batchCount}/${batchCount}批` : `${batchCount}批`;
        if (line4) line4.className = transDone ? 'pipe-line done' : 'pipe-line';
      }

      const mergeDone = Boolean(sd.sub_video || sd.dub_video);
      if (pill5) pill5.className = mergeDone ? 'pipe-node done' : 'pipe-node';
      if (b5) b5.textContent = sd.dub_video ? '已配音' : (sd.sub_video ? '已压制' : '待合成');
    }

    pill1?.addEventListener('click', () => {
      activeBatchId = 'asr-01';
      renderBatchTree();
      syncDetailPane();
      renderStreamTable();
    });
    pill2?.addEventListener('click', () => {
      activeBatchId = 'nlp-all';
      renderBatchTree();
      syncDetailPane();
      renderStreamTable();
    });
    pill3?.addEventListener('click', () => {
      activeBatchId = 'terms-03';
      renderBatchTree();
      syncDetailPane();
    });
    pill4?.addEventListener('click', () => {
      activeBatchId = 'step-04-group';
      renderBatchTree();
      syncDetailPane();
      renderStreamTable();
    });

    // =========================================================================
    // 6. 术语表渲染 (单行等高行 + 滑动开关 + 实时过滤 + Alias 胶囊增删)
    // =========================================================================
    function renderTaskTerms() {
      const tbody = document.getElementById('task-terms-tbody');
      const query = (document.getElementById('task-terms-search').value || '').trim().toLowerCase();

      const filtered = REAL_TASK_TERMS.map((item, idx) => ({ item, idx })).filter(({ item }) => {
        if (!query) return true;
        return (
          item.src.toLowerCase().includes(query) ||
          item.tgt.toLowerCase().includes(query) ||
          item.note.toLowerCase().includes(query) ||
          item.aliases.some(a => a.toLowerCase().includes(query))
        );
      });

      document.getElementById('task-terms-counter').textContent = query
        ? `${filtered.length} / ${REAL_TASK_TERMS.length} 条`
        : `${REAL_TASK_TERMS.length} 条`;
      const termNode = PIPELINE_BATCH_NODES.find(n => n.id === 'terms-03');
      if (termNode) {
        termNode.meta = `${REAL_TASK_TERMS.length} 条`;
        renderBatchTree();
      }

      if (filtered.length === 0) {
        const emptyHint = REAL_TASK_TERMS.length === 0
          ? '当前任务暂无提取的术语，可点击右上角「+ 新增」手动添加或「重新提取」'
          : '未匹配到符合搜索条件的术语';
        tbody.innerHTML = `
          <tr>
            <td colspan="5" style="padding:36px 16px; text-align:center; color:var(--text-muted); font-size:var(--fs-xs);">
              ${escapeHtml(emptyHint)}
            </td>
          </tr>
        `;
        return;
      }

      tbody.innerHTML = filtered.map(({ item: t, idx }) => `
        <tr data-term-row="${idx}">
          <td><input type="text" class="cell-input mono" value="${escapeAttr(t.src)}" placeholder="原文术语" data-term-field="src" data-idx="${idx}"></td>
          <td><input type="text" class="cell-input" style="font-weight:600; color:var(--brand-bright);" value="${escapeAttr(t.tgt)}" placeholder="目标译名" data-term-field="tgt" data-idx="${idx}"></td>
          <td>
            <div class="alias-inline-bar">
              ${t.aliases.map((al, aIdx) => `
                <span class="alias-pill">
                  <span>${escapeHtml(al)}</span>
                  <button type="button" class="alias-del" data-del-alias="${idx}:${aIdx}" title="移除别名">×</button>
                </span>
              `).join('')}
              <input type="text" class="alias-add-input mono" placeholder="+ 添加别名" data-add-alias="${idx}">
            </div>
          </td>
          <td><input type="text" class="cell-input" value="${escapeAttr(t.note)}" placeholder="选填备注" data-term-field="note" data-idx="${idx}"></td>
          <td>
            <div class="switch-pair">
              <label class="switch-label" title="将该词的语音误识别变体修正回原文字幕">
                <input type="checkbox" class="switch" role="switch" ${t.fix_source ? 'checked' : ''} data-term-fix="${idx}">
                <span>修正原文</span>
              </label>
              <label class="switch-label" title="将该词条保存到领域词表">
                <input type="checkbox" class="switch emerald-sw" role="switch" ${t.promote ? 'checked' : ''} data-term-promote="${idx}">
                <span>存入词表</span>
              </label>
            </div>
          </td>
        </tr>
      `).join('');

      tbody.querySelectorAll('[data-del-alias]').forEach(btn => {
        btn.addEventListener('click', () => {
          const [rIdx, aIdx] = btn.getAttribute('data-del-alias').split(':').map(Number);
          REAL_TASK_TERMS[rIdx].aliases.splice(aIdx, 1);
          isEditingTerms = true;
          setAutosaveStatus('paused', '术语未保存');
          renderTaskTerms();
        });
      });

      tbody.querySelectorAll('[data-add-alias]').forEach(inp => {
        inp.addEventListener('keydown', e => {
          if (e.key === 'Enter' && inp.value.trim()) {
            e.preventDefault();
            const rIdx = Number(inp.getAttribute('data-add-alias'));
            REAL_TASK_TERMS[rIdx].aliases.push(inp.value.trim());
            isEditingTerms = true;
            setAutosaveStatus('paused', '术语未保存');
            renderTaskTerms();
          }
        });
      });

      tbody.querySelectorAll('[data-term-field]').forEach(inp => {
        inp.addEventListener('input', () => {
          const rIdx = Number(inp.getAttribute('data-idx'));
          const field = inp.getAttribute('data-term-field');
          REAL_TASK_TERMS[rIdx][field] = inp.value;
          isEditingTerms = true;
          setAutosaveStatus('paused', '术语未保存');
        });
      });

      tbody.querySelectorAll('[data-term-fix]').forEach(chk => {
        chk.addEventListener('change', () => {
          const rIdx = Number(chk.getAttribute('data-term-fix'));
          REAL_TASK_TERMS[rIdx].fix_source = chk.checked;
          isEditingTerms = true;
          setAutosaveStatus('paused', '术语未保存');
        });
      });

      tbody.querySelectorAll('[data-term-promote]').forEach(chk => {
        chk.addEventListener('change', () => {
          const rIdx = Number(chk.getAttribute('data-term-promote'));
          REAL_TASK_TERMS[rIdx].promote = chk.checked;
          isEditingTerms = true;
          setAutosaveStatus('paused', '术语未保存');
        });
      });
    }

    document.getElementById('terms-theme-input')?.addEventListener('input', () => {
      isEditingTerms = true;
      setAutosaveStatus('paused', '术语未保存');
    });

    document.getElementById('task-terms-search').addEventListener('input', renderTaskTerms);

    document.getElementById('btn-batch-fix-on').addEventListener('click', () => {
      const allOn = REAL_TASK_TERMS.every(t => t.fix_source);
      REAL_TASK_TERMS.forEach(t => { t.fix_source = !allOn; });
      isEditingTerms = true;
      setAutosaveStatus('paused', '术语未保存');
      renderTaskTerms();
    });

    document.getElementById('btn-batch-promote-on').addEventListener('click', () => {
      const allOn = REAL_TASK_TERMS.every(t => t.promote);
      REAL_TASK_TERMS.forEach(t => { t.promote = !allOn; });
      isEditingTerms = true;
      setAutosaveStatus('paused', '术语未保存');
      renderTaskTerms();
    });

    document.getElementById('btn-add-task-term').addEventListener('click', () => {
      REAL_TASK_TERMS.unshift({
        src: '',
        tgt: '',
        aliases: [],
        note: '',
        fix_source: false,
        promote: true
      });
      isEditingTerms = true;
      setAutosaveStatus('paused', '术语未保存');
      renderTaskTerms();
      const firstInput = document.querySelector('#task-terms-tbody input[data-term-field="src"]');
      firstInput?.focus();
    });

    async function persistTaskTermsToBackend() {
      const task = REAL_TASKS.find(x => x.id === activeTaskId);
      if (!isBackendConnected || !task?.fromBackend) {
        isEditingTerms = false;
        setAutosaveStatus('completed', '已保存');
        return;
      }
      const summary = document.getElementById('terms-theme-input')?.value?.trim() || '';
      const glossary_domain = document.getElementById('detail-domain-select')?.value ?? task.domain ?? '';
      const payloadTerms = REAL_TASK_TERMS
        .filter(t => String(t.src || '').trim() || String(t.tgt || '').trim())
        .map(t => ({
          src: t.src,
          tgt: t.tgt,
          aliases: t.aliases,
          note: t.note,
          fix_source: Boolean(t.fix_source),
          promote_to_domain: Boolean(t.promote),
        }));
      await apiFetch(`/api/tasks/${encodeURIComponent(activeTaskId)}/terms`, {
        method: 'PUT',
        body: JSON.stringify({ summary, terms: payloadTerms, glossary_domain }),
      });
      isEditingTerms = false;
      setAutosaveStatus('completed', '已保存');
    }

    document.getElementById('detail-domain-select')?.addEventListener('change', async e => {
      const dom = e.target.value;
      const task = REAL_TASKS.find(x => x.id === activeTaskId);
      if (task) task.domain = dom;
      document.getElementById('detail-domain-pill').textContent = dom || '无领域';
      renderTaskRail();
      if (isBackendConnected && task?.fromBackend) {
        try {
          await apiFetch(`/api/tasks/${encodeURIComponent(activeTaskId)}/terms`, {
            method: 'PUT',
            body: JSON.stringify({ glossary_domain: dom }),
          });
        } catch (err) {
          reportActionError('TERMS', err);
        }
      }
    });

    document.getElementById('btn-rerun-term-extract')?.addEventListener('click', async () => {
      const task = REAL_TASKS.find(x => x.id === activeTaskId);
      if (isBackendConnected && task?.fromBackend) {
        try {
          await persistTaskTermsToBackend();
          await apiFetch(`/api/tasks/${encodeURIComponent(activeTaskId)}/invalidate`, {
            method: 'POST',
            body: JSON.stringify({ stage: 'summarize', autorun: true, run_stage: 'subtitles' }),
          });
          appendLog('llm', 'TERMS', `Invalidated stage 'summarize' and re-running term extraction for ${activeTaskId}...`, false);
          await loadBackendTasks();
          return;
        } catch (err) {
          reportActionError('TERMS', err);
          return;
        }
      }
      appendLog('llm', 'TERMS', `Re-extracting terminology with bound domain glossary...`, false);
    });

    document.getElementById('btn-save-terms-and-continue').addEventListener('click', async () => {
      const task = REAL_TASKS.find(x => x.id === activeTaskId);
      if (isBackendConnected && task?.fromBackend) {
        try {
          await persistTaskTermsToBackend();
          if (task.rawState === 'paused') {
            await apiFetch(`/api/tasks/${encodeURIComponent(activeTaskId)}/resume`, { method: 'POST' });
          } else {
            await apiFetch(`/api/tasks/${encodeURIComponent(activeTaskId)}/invalidate`, {
              method: 'POST',
              body: JSON.stringify({ stage: 'translate', autorun: true, run_stage: 'subtitles' }),
            });
          }
          await loadBackendTasks();
        } catch (err) {
          reportActionError('TERMS', err);
          return;
        }
      }
      activeBatchId = 'step-04-group';
      renderBatchTree();
      syncDetailPane();
      renderStreamTable();
      appendLog('llm', 'TERMS', `Saved terms (${REAL_TASK_TERMS.length} terms) and resumed Step 04 translation...`, false);
    });

    document.getElementById('btn-save-terms-only').addEventListener('click', async () => {
      try {
        await persistTaskTermsToBackend();
        appendLog('llm', 'TERMS', 'Saved task terms to project.sqlite without advancing pipeline checkpoint.', false);
      } catch (err) {
        reportActionError('TERMS', err);
      }
    });

    // =========================================================================
    // 7. 底部可折叠终端日志渲染与过滤
    // =========================================================================
    let warnOnlyLog = false;
    function renderSystemLogs() {
      const box = document.getElementById('global-log-console');
      const warnCount = REAL_SYSTEM_LOGS.filter(x => x.warn).length;
      const warnBtn = document.getElementById('btn-filter-warn-log');
      if (warnBtn) {
        warnBtn.textContent = warnOnlyLog ? '全部日志' : `仅看告警 (${warnCount})`;
      }
      const list = warnOnlyLog ? REAL_SYSTEM_LOGS.filter(x => x.warn) : REAL_SYSTEM_LOGS;
      if (list.length === 0) {
        box.innerHTML = `<div class="log-row"><span class="log-time">--:--:--</span><span class="log-tag">IDLE</span><span class="log-msg">${warnOnlyLog ? '暂无告警日志' : '等待任务运行输出日志…'}</span></div>`;
      } else {
        box.innerHTML = list.map(item => `
          <div class="log-row ${item.warn ? 'warn' : ''}">
            <span class="log-time">${escapeHtml(item.time)}</span>
            <span class="log-tag ${escapeAttr(item.tag)}">${escapeHtml(item.label)}</span>
            <span class="log-msg">${escapeHtml(item.msg)}</span>
          </div>
        `).join('');
      }
      box.scrollTop = box.scrollHeight;
      const previewEl = document.getElementById('console-latest-preview');
      const lastItem = REAL_SYSTEM_LOGS[REAL_SYSTEM_LOGS.length - 1];
      if (previewEl) {
        previewEl.textContent = lastItem
          ? `${lastItem.time} · [${lastItem.label}] ${lastItem.msg}`
          : '等待任务运行…';
      }
    }

    function appendLog(tag, label, msg, warn) {
      const now = new Date().toTimeString().slice(0, 8);
      REAL_SYSTEM_LOGS.push({ time: now, tag, label, msg, warn });
      renderSystemLogs();
    }

    document.getElementById('btn-filter-warn-log').addEventListener('click', () => {
      warnOnlyLog = !warnOnlyLog;
      if (consoleDock.classList.contains('collapsed')) {
        toggleConsoleDock();
      }
      renderSystemLogs();
    });

    document.getElementById('btn-copy-log').addEventListener('click', e => {
      const btn = e.currentTarget;
      const text = REAL_SYSTEM_LOGS.map(l => `${l.time} [${l.label}] ${l.msg}`).join('\n');
      navigator.clipboard?.writeText(text).catch(() => {});
      btn.textContent = '已复制 ✓';
      setTimeout(() => { btn.textContent = '复制'; }, 1200);
    });

    const consoleDock = document.getElementById('console-dock');
    const btnToggleConsole = document.getElementById('btn-toggle-console');
    function toggleConsoleDock() {
      const collapsed = consoleDock.classList.toggle('collapsed');
      btnToggleConsole.textContent = collapsed ? '展开 ▴' : '收起 ▾';
    }
    btnToggleConsole.addEventListener('click', toggleConsoleDock);
    document.getElementById('console-bar-toggle-area')?.addEventListener('click', toggleConsoleDock);
    // =========================================================================
    // 8. Subtitle Edit 交接动作与轻量成片预览弹窗
    // =========================================================================
    document.getElementById('btn-open-task-dir').addEventListener('click', async () => {
      const task = REAL_TASKS.find(x => x.id === activeTaskId);
      if (!task) return;
      if (isBackendConnected && task.fromBackend) {
        try {
          await apiFetch(`/api/tasks/${encodeURIComponent(activeTaskId)}/open-dir`, { method: 'POST' });
        } catch (err) {
          reportActionError('OPEN', err);
          return;
        }
      }
      appendLog('llm', 'OPEN', `Opened local directory output/${activeTaskId}/ in Finder / Explorer for Subtitle Edit review.`, false);
    });

    document.getElementById('btn-burn-sub-video').addEventListener('click', async () => {
      const task = REAL_TASKS.find(x => x.id === activeTaskId);
      if (!task) return;
      document.getElementById('pill-stage-5').className = 'pipe-node active';
      document.getElementById('pill-badge-5').textContent = '烧录中…';
      appendLog('llm', 'MERGE', `Burning bilingual ASS subtitles into output/${activeTaskId}/output_sub.mp4...`, false);
      if (isBackendConnected && task.fromBackend) {
        try {
          if (isEditingUtterances) {
            await persistTaskUtterancesToBackend();
          }
          await apiFetch(`/api/tasks/${encodeURIComponent(activeTaskId)}/run`, {
            method: 'POST',
            body: JSON.stringify({ stage: 'subtitles', dubbing: false }),
          });
          await loadBackendTasks();
        } catch (err) {
          reportActionError('MERGE', err);
        }
      }
    });

    document.getElementById('btn-run-dubbing').addEventListener('click', async () => {
      const task = REAL_TASKS.find(x => x.id === activeTaskId);
      if (!task) return;
      document.getElementById('pill-stage-5').className = 'pipe-node active';
      document.getElementById('pill-badge-5').textContent = '配音中…';
      appendLog('llm', 'TTS', `Running TTS alignment from edited trans.srt -> output/${activeTaskId}/output_dub.mp4...`, false);
      if (isBackendConnected && task.fromBackend) {
        try {
          if (isEditingUtterances) {
            await persistTaskUtterancesToBackend();
          }
          await apiFetch(`/api/tasks/${encodeURIComponent(activeTaskId)}/run`, {
            method: 'POST',
            body: JSON.stringify({ stage: 'dubbing', dubbing: true }),
          });
          await loadBackendTasks();
        } catch (err) {
          reportActionError('TTS', err);
        }
      }
    });

    document.getElementById('btn-stop-task')?.addEventListener('click', async () => {
      const task = REAL_TASKS.find(x => x.id === activeTaskId);
      if (!task) return;
      if (isBackendConnected && task.fromBackend) {
        try {
          await apiFetch(`/api/tasks/${encodeURIComponent(activeTaskId)}/stop`, { method: 'POST' });
          await loadBackendTasks();
        } catch (err) {
          reportActionError('STOP', err);
          return;
        }
      }
      appendLog('warn', 'STOP', `Requested stop for task ${activeTaskId}.`, true);
    });

    const previewDialog = document.getElementById('video-preview-dialog');
    wireModalDialog(previewDialog);
    document.getElementById('btn-preview-video-modal').addEventListener('click', () => {
      const bodyEl = document.getElementById('video-preview-body');
      const task = REAL_TASKS.find(x => x.id === activeTaskId);
      const sd = task?.raw?.stages_done || {};
      const files = task?.raw?.files || [];
      let videoFile = null;
      if (sd.dub_video || files.includes('output_dub.mp4')) {
        videoFile = 'output_dub.mp4';
      } else if (sd.sub_video || files.includes('output_sub.mp4')) {
        videoFile = 'output_sub.mp4';
      } else if (task?.raw?.media?.filename && files.includes(task.raw.media.filename)) {
        videoFile = task.raw.media.filename;
      }
      if (bodyEl) {
        if (task && videoFile) {
          const srcUrl = `/api/tasks/${encodeURIComponent(task.id)}/files/${encodeURIComponent(videoFile)}`;
          bodyEl.innerHTML = `
            <div style="border-radius:var(--radius-md); overflow:hidden; background:#000;">
              <video controls preload="metadata" style="inline-size:100%; display:block; aspect-ratio:16/9;" src="${escapeAttr(srcUrl)}"></video>
            </div>
          `;
        } else {
          bodyEl.innerHTML = `
            <div class="ass-monitor" style="aspect-ratio:16/9;">
              <div class="ass-safe-frame"></div>
              <div class="ass-hud-top">
                <span class="ass-hud-tag" id="preview-video-tag">${task ? escapeHtml(task.title) : '未选择任务'}</span>
                <span class="ass-hud-tag">未生成成片</span>
              </div>
              <div class="ass-caption-stack" style="margin-block-end:22px;">
                <div id="preview-modal-status" style="color:#cbd5e1; font-size:14px;">
                  ${task ? '当前任务尚未生成成片视频（output_sub.mp4 / output_dub.mp4）' : '请先在左侧选择或新建任务'}
                </div>
              </div>
            </div>
          `;
        }
      }
      previewDialog.showModal();
    });
    document.getElementById('btn-close-preview-modal').addEventListener('click', () => {
      const videoEl = previewDialog.querySelector('video');
      if (videoEl) videoEl.pause();
      closeTopLayer(previewDialog);
    });

    // =========================================================================
    // 9. 领域词表中心 (terms.csv + dict.json 检索与转入)
    // =========================================================================
    function getActiveDomainTerms() {
      if (!activeGlossaryDomain) return [];
      if (!domainTermsMap[activeGlossaryDomain]) {
        domainTermsMap[activeGlossaryDomain] = [];
      }
      return domainTermsMap[activeGlossaryDomain];
    }

    function renderDomainList() {
      const box = document.getElementById('domain-list-box');
      if (!box) return;
      if (domainListMeta.length === 0) {
        box.innerHTML = `
          <div style="padding:20px 12px; text-align:center; color:var(--text-muted); font-size:var(--fs-xs);">
            暂无领域词表，可在下方新建
          </div>
        `;
      } else {
        box.innerHTML = domainListMeta.map((d, idx) => {
          const active = d.domain === activeGlossaryDomain ? 'active' : '';
          const dictLabel = d.dict_count >= 10000
            ? `${(d.dict_count / 10000).toFixed(1)}万词典`
            : `${Number(d.dict_count || 0).toLocaleString()}词典`;
          const hint = d.has_dict
            ? `${d.terms_count || 0} 条术语 · ${dictLabel}`
            : `${d.terms_count || 0} 条术语`;
          const pillClass = idx === 0 ? 'status-pill completed' : 'status-pill';
          const pillText = idx === 0 ? '默认' : '可用';
          return `
            <div class="domain-item ${active}" data-domain="${escapeAttr(d.domain)}">
              <div style="min-inline-size:0;">
                <strong class="mono" style="font-size:var(--fs-sm); display:block;">${escapeHtml(d.domain)}</strong>
                <div class="field-hint" style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(hint)}</div>
              </div>
              <span class="${pillClass}">${pillText}</span>
            </div>
          `;
        }).join('');

        box.querySelectorAll('.domain-item').forEach(card => {
          card.addEventListener('click', async () => {
            const dom = card.getAttribute('data-domain');
            await selectGlossaryDomain(dom);
          });
        });
      }

      const activeMeta = domainListMeta.find(d => d.domain === activeGlossaryDomain);
      const dictPill = document.getElementById('dict-count-pill');
      if (dictPill) {
        if (activeMeta?.has_dict) {
          dictPill.className = 'status-pill completed';
          dictPill.innerHTML = `<span class="status-dot"></span><span>${Number(activeMeta.dict_count || 0).toLocaleString()} 词</span>`;
        } else {
          dictPill.className = 'status-pill';
          dictPill.innerHTML = `<span class="status-dot"></span><span>未挂载词典</span>`;
        }
      }

      // Keep #detail-domain-select, Settings [data-cfg="glossary_domain"], and New Task #nt-domain-menu-list synced
      const detailSelect = document.getElementById('detail-domain-select');
      if (detailSelect) {
        const curVal = detailSelect.value;
        detailSelect.innerHTML = [
          '<option value="">不挂载</option>',
          ...domainListMeta.map(d => `<option value="${escapeAttr(d.domain)}">${escapeHtml(d.domain)}</option>`),
        ].join('');
        if ([...detailSelect.options].some(o => o.value === curVal)) {
          detailSelect.value = curVal;
        }
      }

      const cfgDomainSelect = document.querySelector('select[data-cfg="glossary_domain"]');
      if (cfgDomainSelect) {
        const curCfgVal = currentConfig.glossary_domain !== undefined ? String(currentConfig.glossary_domain) : cfgDomainSelect.value;
        cfgDomainSelect.innerHTML = [
          '<option value="">不挂载</option>',
          ...domainListMeta.map(d => `<option value="${escapeAttr(d.domain)}">${escapeHtml(d.domain)}</option>`),
        ].join('');
        if ([...cfgDomainSelect.options].some(o => o.value === curCfgVal)) {
          cfgDomainSelect.value = curCfgVal;
        }
      }

      const ntMenuList = document.getElementById('nt-domain-menu-list');
      if (ntMenuList) {
        const curTaskDom = document.getElementById('task-domain')?.value ?? '';
        const matchedMeta = domainListMeta.find(d => d.domain === curTaskDom);
        const effectiveDom = matchedMeta ? matchedMeta.domain : '';
        ntMenuList.innerHTML = [
          ...domainListMeta.map(d => {
            const sub = `${d.terms_count || 0} 词`;
            const isAct = d.domain === effectiveDom ? 'active' : '';
            return `
              <button type="button" class="nt-menu-item ${isAct}" role="option" aria-selected="${Boolean(isAct)}" data-value="${escapeAttr(d.domain)}" data-sub="${escapeAttr(sub)}">
                <span class="nt-item-main mono">${escapeHtml(d.domain)}</span>
                <span class="nt-item-sub">${escapeHtml(sub)}</span>
              </button>
            `;
          }),
          ...(domainListMeta.length > 0 ? ['<div class="nt-menu-sep" role="separator"></div>'] : []),
          `<button type="button" class="nt-menu-item ${effectiveDom === '' ? 'active' : ''}" role="option" aria-selected="${effectiveDom === ''}" data-value="" data-sub="通用">
            <span class="nt-item-main">不挂载词表</span>
            <span class="nt-item-sub">通用</span>
          </button>`
        ].join('');

        const slugEl = document.getElementById('nt-domain-slug');
        const subEl = document.getElementById('nt-domain-sub');
        if (slugEl && subEl) {
          if (matchedMeta) {
            document.getElementById('task-domain').value = matchedMeta.domain;
            slugEl.textContent = matchedMeta.domain;
            slugEl.classList.add('mono');
            subEl.textContent = `${matchedMeta.terms_count || 0} 词`;
          } else {
            document.getElementById('task-domain').value = '';
            slugEl.textContent = '不挂载词表';
            slugEl.classList.remove('mono');
            subEl.textContent = '通用';
          }
        }
      }
    }

    async function selectGlossaryDomain(dom) {
      activeGlossaryDomain = dom || '';
      const titleEl = document.getElementById('glossary-active-title');
      if (titleEl) {
        titleEl.textContent = activeGlossaryDomain ? `${activeGlossaryDomain} · 自定义术语` : '未选择领域 · 自定义术语';
      }
      if (isBackendConnected && activeGlossaryDomain) {
        try {
          const data = await apiFetch(`/api/glossary/domains/${encodeURIComponent(activeGlossaryDomain)}`);
          const terms = (data.terms || []).map(t => ({
            src: t.src || '',
            tgt: t.tgt || '',
            aliases: Array.isArray(t.aliases) ? t.aliases : [],
            note: t.note || '',
            fix_source: Boolean(t.fix_source),
            promote: true,
          }));
          domainTermsMap[activeGlossaryDomain] = terms;
          const m = domainListMeta.find(x => x.domain === activeGlossaryDomain);
          if (m) {
            m.terms_count = terms.length;
            m.has_dict = Boolean(data.has_dict);
            m.dict_count = data.dict_count ?? 0;
          }
        } catch (err) {
          reportActionError('GLOSSARY', err);
        }
      }
      renderDomainList();
      renderDomainCsvTable();
      await loadDomainDictEntries(activeGlossaryDomain, document.getElementById('dict-search-input')?.value || '');
    }

    async function loadBackendGlossaryDomains() {
      try {
        const data = await apiFetch('/api/glossary/domains');
        isBackendConnected = true;
        domainListMeta = Array.isArray(data.domains)
          ? data.domains.map(d => ({
              domain: d.domain || d.name,
              terms_count: d.terms_count ?? 0,
              has_dict: Boolean(d.has_dict),
              dict_count: d.dict_count ?? 0,
            }))
          : [];
        if (domainListMeta.length > 0) {
          const preferred = data.active_domain !== undefined
            ? (domainListMeta.some(x => x.domain === data.active_domain) ? data.active_domain : '')
            : (currentConfig.glossary_domain !== undefined && domainListMeta.some(x => x.domain === currentConfig.glossary_domain)
                ? String(currentConfig.glossary_domain)
                : domainListMeta[0].domain);
          const taskDomInput = document.getElementById('task-domain');
          if (taskDomInput && !taskDomInput.value && preferred) {
            taskDomInput.value = preferred;
          }
          if (!domainListMeta.some(x => x.domain === activeGlossaryDomain)) {
            activeGlossaryDomain = preferred || domainListMeta[0].domain;
          }
          await selectGlossaryDomain(activeGlossaryDomain);
        } else {
          await selectGlossaryDomain('');
        }
      } catch (err) {
        appendLog('warn', 'GLOSSARY', `加载领域词表列表失败: ${err.message}`, true);
      }
    }

    async function loadDomainDictEntries(dom, keyword = '') {
      if (isBackendConnected && dom) {
        try {
          const data = await apiFetch(`/api/glossary/domains/${encodeURIComponent(dom)}/dict?q=${encodeURIComponent(keyword)}`);
          if (Array.isArray(data.entries)) {
            domainDictEntries = data.entries;
            renderDictSearch(keyword, true);
            return;
          }
        } catch (err) {
          appendLog('warn', 'DICT', `检索内置词典失败 (${dom}): ${err.message}`, true);
        }
      }
      domainDictEntries = [];
      renderDictSearch(keyword, true);
    }

    function renderDomainCsvTable() {
      const tbody = document.getElementById('domain-csv-tbody');
      const q = (document.getElementById('glossary-csv-search').value || '').trim().toLowerCase();
      const list = getActiveDomainTerms();
      const rows = list.map((t, idx) => ({ t, idx })).filter(({ t }) => {
        if (!q) return true;
        return t.src.toLowerCase().includes(q) || t.tgt.toLowerCase().includes(q) || t.aliases.join('|').toLowerCase().includes(q);
      });
      if (rows.length === 0) {
        const emptyHint = !activeGlossaryDomain
          ? '请先在左侧选择或新建一个领域词表'
          : (list.length === 0 ? '当前领域暂无自定义术语，可点击右上角「导入」或「+ 新增」' : '未匹配到符合搜索条件的自定义术语');
        tbody.innerHTML = `
          <tr>
            <td colspan="5" style="padding:28px 16px; text-align:center; color:var(--text-muted); font-size:var(--fs-xs);">
              ${escapeHtml(emptyHint)}
            </td>
          </tr>
        `;
        return;
      }
      tbody.innerHTML = rows.map(({ t, idx }) => `
        <tr>
          <td><input type="text" class="cell-input mono" value="${escapeAttr(t.src)}" placeholder="原文术语" data-csv-field="src" data-idx="${idx}"></td>
          <td><input type="text" class="cell-input" style="font-weight:600; color:var(--brand-bright);" value="${escapeAttr(t.tgt)}" placeholder="目标译名" data-csv-field="tgt" data-idx="${idx}"></td>
          <td>
            <div class="alias-inline-bar">
              ${t.aliases.map((al, aIdx) => `
                <span class="alias-pill">
                  <span>${escapeHtml(al)}</span>
                  <button type="button" class="alias-del" data-del-csv-alias="${idx}:${aIdx}" title="移除别名">×</button>
                </span>
              `).join('')}
              <input type="text" class="alias-add-input mono" placeholder="+ 添加别名" data-add-csv-alias="${idx}">
            </div>
          </td>
          <td><input type="text" class="cell-input" value="${escapeAttr(t.note)}" placeholder="选填备注" data-csv-field="note" data-idx="${idx}"></td>
          <td><button type="button" class="danger xs" data-del-csv="${idx}">删除</button></td>
        </tr>
      `).join('');

      tbody.querySelectorAll('[data-del-csv-alias]').forEach(btn => {
        btn.addEventListener('click', () => {
          const [rIdx, aIdx] = btn.getAttribute('data-del-csv-alias').split(':').map(Number);
          list[rIdx].aliases.splice(aIdx, 1);
          renderDomainCsvTable();
        });
      });

      tbody.querySelectorAll('[data-add-csv-alias]').forEach(inp => {
        inp.addEventListener('keydown', e => {
          if (e.key === 'Enter' && inp.value.trim()) {
            e.preventDefault();
            const rIdx = Number(inp.getAttribute('data-add-csv-alias'));
            list[rIdx].aliases.push(inp.value.trim());
            renderDomainCsvTable();
          }
        });
      });

      tbody.querySelectorAll('[data-csv-field]').forEach(inp => {
        inp.addEventListener('input', () => {
          const rIdx = Number(inp.getAttribute('data-idx'));
          const field = inp.getAttribute('data-csv-field');
          list[rIdx][field] = inp.value;
        });
      });

      tbody.querySelectorAll('[data-del-csv]').forEach(btn => {
        btn.addEventListener('click', () => {
          const idx = Number(btn.getAttribute('data-del-csv'));
          list.splice(idx, 1);
          renderDomainCsvTable();
        });
      });
    }

    document.getElementById('glossary-csv-search').addEventListener('input', renderDomainCsvTable);

    document.getElementById('btn-add-domain-term').addEventListener('click', () => {
      if (!activeGlossaryDomain) return;
      const list = getActiveDomainTerms();
      list.unshift({ src: '', tgt: '', aliases: [], note: '', fix_source: false, promote: true });
      renderDomainCsvTable();
      const firstInput = document.querySelector('#domain-csv-tbody input[data-csv-field="src"]');
      firstInput?.focus();
    });

    document.getElementById('btn-save-domain-csv')?.addEventListener('click', async () => {
      if (!activeGlossaryDomain) return;
      const list = getActiveDomainTerms().filter(t => String(t.src || '').trim() || String(t.tgt || '').trim());
      domainTermsMap[activeGlossaryDomain] = list;
      if (isBackendConnected) {
        try {
          await apiFetch(`/api/glossary/domains/${encodeURIComponent(activeGlossaryDomain)}`, {
            method: 'PUT',
            body: JSON.stringify({
              terms: list.map(t => ({ src: t.src, tgt: t.tgt, aliases: t.aliases, note: t.note })),
            }),
          });
          setAutosaveStatus('completed', '词表已保存');
          await loadBackendGlossaryDomains();
          return;
        } catch (err) {
          reportActionError('GLOSSARY', err);
          return;
        }
      }
      setAutosaveStatus('completed', '词表已保存');
    });

    // Hidden file input for glossary import
    const glossaryFileInput = document.createElement('input');
    glossaryFileInput.type = 'file';
    glossaryFileInput.accept = '.csv,.tsv,.xlsx,.xls,.json';
    glossaryFileInput.style.display = 'none';
    document.body.appendChild(glossaryFileInput);

    document.getElementById('btn-import-glossary')?.addEventListener('click', () => {
      glossaryFileInput.accept = '.csv,.tsv,.xlsx,.xls,.json';
      glossaryFileInput.click();
    });

    document.getElementById('btn-import-dict')?.addEventListener('click', () => {
      glossaryFileInput.accept = '.json,.csv,.xlsx,.xls';
      glossaryFileInput.click();
    });

    glossaryFileInput.addEventListener('change', async () => {
      const file = glossaryFileInput.files?.[0];
      if (!file || !activeGlossaryDomain) return;
      if (isBackendConnected) {
        try {
          const form = new FormData();
          form.append('file', file);
          form.append('mode', 'merge');
          form.append('target', file.name.toLowerCase().endsWith('.json') ? 'dict' : 'terms');
          await apiFetch(`/api/glossary/domains/${encodeURIComponent(activeGlossaryDomain)}/import`, {
            method: 'POST',
            body: form,
          });
          await loadBackendGlossaryDomains();
        } catch (err) {
          reportActionError('IMPORT', err);
        }
      }
      glossaryFileInput.value = '';
    });

    document.getElementById('btn-create-domain')?.addEventListener('click', async () => {
      const inp = document.getElementById('new-domain-input');
      const slug = (inp?.value || '').trim().toLowerCase().replace(/[^a-z0-9_-]/g, '_');
      if (!slug) return;
      if (isBackendConnected) {
        try {
          await apiFetch('/api/glossary/domains', {
            method: 'POST',
            body: JSON.stringify({ domain: slug }),
          });
          inp.value = '';
          activeGlossaryDomain = slug;
          await loadBackendGlossaryDomains();
          return;
        } catch (err) {
          reportActionError('GLOSSARY', err);
          return;
        }
      }
      if (!domainListMeta.some(d => d.domain === slug)) {
        domainListMeta.push({ domain: slug, terms_count: 0, has_dict: false, dict_count: 0 });
        domainTermsMap[slug] = [];
      }
      inp.value = '';
      await selectGlossaryDomain(slug);
    });

    document.getElementById('btn-delete-domain')?.addEventListener('click', async () => {
      if (!activeGlossaryDomain) return;
      if (isBackendConnected) {
        try {
          await apiFetch(`/api/glossary/domains/${encodeURIComponent(activeGlossaryDomain)}`, { method: 'DELETE' });
          await loadBackendGlossaryDomains();
          return;
        } catch (err) {
          reportActionError('GLOSSARY', err);
          return;
        }
      }
      domainListMeta = domainListMeta.filter(d => d.domain !== activeGlossaryDomain);
      await selectGlossaryDomain(domainListMeta[0]?.domain || '');
    });

    function renderDictSearch(keyword = '', prefiltered = false) {
      const tbody = document.getElementById('dict-search-tbody');
      const q = keyword.trim().toLowerCase();
      const sourceList = prefiltered ? domainDictEntries : (domainDictEntries.length > 0 ? domainDictEntries : REAL_DICT_ENTRIES);
      const matched = prefiltered
        ? sourceList
        : sourceList.filter(x => !q || x.k.toLowerCase().includes(q) || x.v.toLowerCase().includes(q));
      if (matched.length === 0) {
        const activeMeta = domainListMeta.find(d => d.domain === activeGlossaryDomain);
        const emptyHint = !activeMeta?.has_dict
          ? '当前领域尚未导入 dict.json 词典，可点击右上角「导入词典」上传 JSON 或大表'
          : '未匹配到符合搜索条件的词条';
        tbody.innerHTML = `
          <tr>
            <td colspan="3" style="padding:28px 16px; text-align:center; color:var(--text-muted); font-size:var(--fs-xs);">
              ${escapeHtml(emptyHint)}
            </td>
          </tr>
        `;
        return;
      }
      tbody.innerHTML = matched.map((item, i) => `
        <tr>
          <td class="mono" style="font-weight:600;">${escapeHtml(item.k)}</td>
          <td style="color:var(--emerald-bright); font-weight:600;">${escapeHtml(item.v)}</td>
          <td><button type="button" class="xs" data-promote-dict="${i}">+ 加入自定义术语</button></td>
        </tr>
      `).join('');

      tbody.querySelectorAll('[data-promote-dict]').forEach(btn => {
        btn.addEventListener('click', () => {
          const item = matched[Number(btn.getAttribute('data-promote-dict'))];
          const list = getActiveDomainTerms();
          if (item && !list.some(x => x.src === item.k)) {
            list.unshift({ src: item.k, tgt: item.v, aliases: [], note: '来自内置词典', fix_source: false, promote: true });
            renderDomainCsvTable();
          }
        });
      });
    }

    document.getElementById('dict-search-input').addEventListener('input', e => {
      loadDomainDictEntries(activeGlossaryDomain, e.target.value);
    });

    document.querySelectorAll('.settings-nav .settings-nav-item').forEach(link => {
      link.addEventListener('click', () => {
        document.querySelectorAll('.settings-nav .settings-nav-item').forEach(l => l.classList.remove('active'));
        link.classList.add('active');
      });
    });

    // =========================================================================
    // 10. 全局设置：ASR 8 环境切换、TTS 5 渠道切换、ASS 16:9 实时预览与 400ms 防抖保存
    // =========================================================================
    const asrRuntimeSelect = document.getElementById('cfg-asr-runtime');
    function syncAsrRuntimePanel() {
      const val = asrRuntimeSelect.value;
      const targetPanel = (val === 'replicate_parakeet' || val === 'replicate_whisper') ? 'replicate' : val;
      document.querySelectorAll('#asr-engine-panels [data-asr-panel]').forEach(p => {
        p.classList.toggle('hidden', p.getAttribute('data-asr-panel') !== targetPanel);
      });
    }
    asrRuntimeSelect.addEventListener('change', syncAsrRuntimePanel);

    const ttsMethodSelect = document.getElementById('cfg-tts-method');
    function syncTtsMethodPanel() {
      const val = ttsMethodSelect.value;
      document.querySelectorAll('#tts-engine-panels [data-tts-panel]').forEach(p => {
        p.classList.toggle('hidden', p.getAttribute('data-tts-panel') !== val);
      });
    }
    ttsMethodSelect.addEventListener('change', syncTtsMethodPanel);

    function normalizeAssColorToHex(raw, fallback = '#FFFF00') {
      const s = String(raw || '').trim();
      const hexMatch = s.match(/^#([0-9a-fA-F]{6})$/);
      if (hexMatch) return `#${hexMatch[1].toUpperCase()}`;
      const assMatch = s.match(/^&[Hh](?:[0-9a-fA-F]{2})?([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$/);
      if (assMatch) {
        const [, bb, gg, rr] = assMatch;
        return `#${rr}${gg}${bb}`.toUpperCase();
      }
      return fallback;
    }

    function updateAssPreview() {
      const mode = document.getElementById('ass-burn-mode').value;
      const trEl = document.getElementById('ass-preview-tr');
      const srcEl = document.getElementById('ass-preview-src');
      const boxEl = document.getElementById('ass-preview-box');

      const trSize = Number(document.getElementById('ass-tr-size').value) || 17;
      const trColor = normalizeAssColorToHex(document.getElementById('ass-tr-color').value, '#FFFF00');
      const trMargin = Number(document.getElementById('ass-tr-margin').value) || 27;
      const srcSize = Number(document.getElementById('ass-src-size').value) || 15;
      const srcColor = normalizeAssColorToHex(document.getElementById('ass-src-color').value, '#FFFFFF');

      const trPicker = document.getElementById('ass-tr-color-picker');
      if (trPicker) trPicker.value = trColor.toLowerCase();
      const srcPicker = document.getElementById('ass-src-color-picker');
      if (srcPicker) srcPicker.value = srcColor.toLowerCase();

      trEl.style.fontSize = `${Math.round(trSize * 1.2)}px`;
      trEl.style.color = trColor;
      srcEl.style.fontSize = `${Math.round(srcSize * 1.08)}px`;
      srcEl.style.color = srcColor;
      boxEl.style.marginBlockEnd = `${Math.max(4, Math.round(trMargin * 0.65))}px`;

      trEl.style.display = (mode === 'source') ? 'none' : 'block';
      srcEl.style.display = (mode === 'translation') ? 'none' : 'block';
    }

    ['ass-burn-mode', 'ass-tr-size', 'ass-tr-color', 'ass-tr-margin', 'ass-src-size', 'ass-src-color'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('input', updateAssPreview);
    });

    document.getElementById('ass-tr-color-picker').addEventListener('input', e => {
      document.getElementById('ass-tr-color').value = e.target.value.toUpperCase();
      document.getElementById('ass-tr-color').dispatchEvent(new Event('input'));
    });
    document.getElementById('ass-src-color-picker').addEventListener('input', e => {
      document.getElementById('ass-src-color').value = e.target.value.toUpperCase();
      document.getElementById('ass-src-color').dispatchEvent(new Event('input'));
    });

    document.getElementById('btn-reset-ass-style')?.addEventListener('click', () => {
      document.getElementById('ass-tr-size').value = '18';
      document.getElementById('ass-tr-color').value = '&H00D7FF';
      document.getElementById('ass-tr-color-picker').value = '#ffd700';
      document.getElementById('ass-tr-outline-w').value = '2.0';
      document.getElementById('ass-tr-margin').value = '28';
      document.getElementById('ass-src-size').value = '14';
      document.getElementById('ass-src-color').value = '#FFFFFF';
      document.getElementById('ass-src-color-picker').value = '#ffffff';
      updateAssPreview();
      document.getElementById('ass-tr-size').dispatchEvent(new Event('input'));
    });

    document.getElementById('btn-detect-mai-region')?.addEventListener('click', async e => {
      const btn = e.currentTarget;
      const apiKey = document.querySelector('[data-cfg="whisper.mai_api_key"]')?.value?.trim() || '';
      const regionInput = document.querySelector('[data-cfg="whisper.mai_region"]');
      if (!apiKey) return;
      btn.textContent = '探测中…';
      try {
        const res = await apiFetch('/api/mai/detect-region', {
          method: 'POST',
          body: JSON.stringify({ api_key: apiKey }),
        });
        if (res.region && regionInput) {
          regionInput.value = res.region;
          regionInput.dispatchEvent(new Event('input'));
        }
      } catch (err) {
        reportActionError('MAI', err);
      }
      btn.textContent = '探测';
    });

    document.getElementById('btn-reset-default-config')?.addEventListener('click', async () => {
      const defaults = {
        max_workers: 4,
        target_language: '简体中文',
        'api.reasoning_effort': 'none',
        reflect_translate: false,
        pause_before_translate: false,
        pause_after_translate: false,
        vocal_separation_model: 'scnet',
        'whisper.language': 'en',
        'whisper.runtime': 'local',
        'whisper.backend': 'qwen',
        'whisper.qwen_model': '1.7b',
        'whisper.qwen_engine': 'auto',
        burn_subtitles: true,
        ffmpeg_gpu: true,
        'subtitle.max_length': 75,
        'subtitle.target_multiplier': 1.3,
        'subtitle.sub_video_subtitles': 'translation',
        'subtitle.dub_video_subtitles': 'translation',
        'speed_factor.min': 1.0,
        'speed_factor.accept': 1.15,
        'speed_factor.max': 1.28,
        min_subtitle_duration: 1.8,
        tolerance: 1.2,
        ytb_resolution: '1080',
        'audio.raw_sample_rate': 44100,
        'audio.raw_bitrate': '192k',
      };
      Object.entries(defaults).forEach(([k, v]) => {
        const el = document.querySelector(`[data-cfg="${k}"]`);
        if (!el) return;
        if (el.type === 'checkbox') el.checked = Boolean(v);
        else el.value = String(v);
      });
      document.getElementById('btn-reset-ass-style')?.click();
      syncAsrRuntimePanel();
      syncTtsMethodPanel();
      if (isBackendConnected) {
        try {
          await apiFetch('/api/config', {
            method: 'PUT',
            body: JSON.stringify({ updates: defaults }),
          });
          setAutosaveStatus('completed', '已恢复默认');
        } catch (err) {
          reportActionError('CONFIG', err);
        }
      } else {
        setAutosaveStatus('completed', '已恢复默认');
      }
    });

    document.getElementById('btn-fetch-models')?.addEventListener('click', async () => {
      const baseUrl = document.querySelector('[data-cfg="api.base_url"]')?.value || '';
      const apiKey = document.querySelector('[data-cfg="api.key"]')?.value || '';
      const statusEl = document.getElementById('llm-check-status');
      if (statusEl) statusEl.textContent = '正在获取模型列表…';
      try {
        const res = await apiFetch('/api/llm/models', {
          method: 'POST',
          body: JSON.stringify({ base_url: baseUrl, api_key: apiKey }),
        });
        const ok = Array.isArray(res.models) && res.models.length > 0 && !res.error;
        if (ok) {
          const dl = document.getElementById('llm-models-list');
          if (dl) {
            dl.innerHTML = res.models.map(m => `<option value="${escapeAttr(m)}"></option>`).join('');
          }
        }
        if (statusEl) {
          statusEl.textContent = ok ? `已获取 ${res.models.length} 个模型` : (res.error || res.message || '获取失败');
        }
      } catch (err) {
        if (statusEl) statusEl.textContent = `获取模型失败: ${err.message}`;
      }
    });

    document.getElementById('btn-check-llm')?.addEventListener('click', async () => {
      const baseUrl = document.querySelector('[data-cfg="api.base_url"]')?.value || '';
      const apiKey = document.querySelector('[data-cfg="api.key"]')?.value || '';
      const model = document.querySelector('[data-cfg="api.model"]')?.value || '';
      const reasoningEffort = document.querySelector('[data-cfg="api.reasoning_effort"]')?.value || 'none';
      const statusEl = document.getElementById('llm-check-status');
      if (statusEl) statusEl.textContent = '正在检测连通性…';
      try {
        const res = await apiFetch('/api/llm/check', {
          method: 'POST',
          body: JSON.stringify({ base_url: baseUrl, api_key: apiKey, model, reasoning_effort: reasoningEffort }),
        });
        const ok = Boolean(res.valid ?? res.ok);
        if (statusEl) {
          statusEl.textContent = ok ? '连通正常' : `检测失败: ${res.error || res.message || '不可用'}`;
          statusEl.style.color = ok ? 'var(--emerald-bright)' : 'var(--rose)';
        }
      } catch (err) {
        if (statusEl) {
          statusEl.textContent = `检测异常: ${err.message}`;
          statusEl.style.color = 'var(--rose)';
        }
      }
    });

    let saveTimer = null;
    const pendingConfigUpdates = {};

    function readCfgControlValue(inp) {
      if (inp.type === 'checkbox') return inp.checked;
      if (inp.type === 'number') {
        const n = Number(inp.value);
        return Number.isNaN(n) ? inp.value : n;
      }
      return inp.value;
    }

    function bindCfgControls() {
      document.querySelectorAll('[data-cfg]').forEach(inp => {
        const handler = () => {
          const path = inp.getAttribute('data-cfg');
          if (path) {
            const val = readCfgControlValue(inp);
            pendingConfigUpdates[path] = val;
            if (path === 'glossary_domain') {
              currentConfig.glossary_domain = val;
              const taskDomInput = document.getElementById('task-domain');
              if (taskDomInput) taskDomInput.value = String(val ?? '');
              renderDomainList();
            }
          }
          setAutosaveStatus('paused', '保存中…');
          clearTimeout(saveTimer);
          saveTimer = setTimeout(async () => {
            if (isBackendConnected && Object.keys(pendingConfigUpdates).length > 0) {
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
        };
        inp.addEventListener('input', handler);
        inp.addEventListener('change', handler);
      });
    }
    bindCfgControls();

    const PLACEHOLDER_API_KEYS = new Set([
      'YOUR_API_KEY',
      'your_elevenlabs_api_key',
      'YOUR_FISH_AUDIO_KEY',
      'YOUR_SF_KEY',
    ]);

    async function loadBackendConfig() {
      try {
        const res = await apiFetch('/api/config');
        isBackendConnected = true;
        currentConfig = res.config || res || {};
        document.querySelectorAll('[data-cfg]').forEach(inp => {
          const path = inp.getAttribute('data-cfg');
          if (!path) return;
          const val = path.split('.').reduce((acc, part) => (acc && acc[part] !== undefined ? acc[part] : undefined), currentConfig);
          if (val === undefined || val === null) return;
          if (inp.type === 'checkbox') {
            inp.checked = Boolean(val);
          } else if (path === 'subtitle.style.translation.font_color' || path === 'subtitle.style.source.font_color') {
            inp.value = normalizeAssColorToHex(val, path.includes('translation') ? '#FFFF00' : '#FFFFFF');
          } else if (path === 'subtitle.style.translation.outline_color' || path === 'subtitle.style.source.outline_color') {
            inp.value = normalizeAssColorToHex(val, '#000000');
          } else if (inp.type === 'password' && PLACEHOLDER_API_KEYS.has(String(val).trim())) {
            inp.value = '';
          } else {
            inp.value = String(val);
          }
        });
        syncAsrRuntimePanel();
        syncTtsMethodPanel();
        updateAssPreview();
      } catch (err) {
        appendLog('warn', 'CONFIG', `加载全局设置失败: ${err.message}`, true);
      }
    }

    // =========================================================================
    // 11. SSE Live Events & Initial Hydration
    // =========================================================================
    function connectEventStream() {
      if (typeof EventSource === 'undefined') return;
      try {
        const es = new EventSource('/api/events');
        es.onmessage = e => {
          try {
            const d = JSON.parse(e.data);
            if (!d || d.type === 'heartbeat' || d.type === 'connected') return;
            if (d.message || d.pause_message || d.error) {
              const msg = d.message || d.pause_message || d.error;
              const warn = Boolean(d.error) || /warn|error|fail|异常|失败/i.test(msg);
              appendLog('llm', (d.step || d.state || 'RUN').toUpperCase().slice(0, 6), msg, warn);
            }
            const isProgressTick = d.type === 'task_progress' || (d.state === 'running' && !d.pause_message && !d.error);
            loadBackendTasks({ skipActiveDetail: isProgressTick });
          } catch (parseErr) {
            appendLog('warn', 'SSE', `解析事件流数据失败: ${parseErr.message}`, true);
          }
        };
      } catch (err) {
        appendLog('warn', 'SSE', `连接实时事件流失败: ${err.message}`, true);
      }
    }

    // 初始化全量视图 (同步立即渲染空状态，随后异步同步后端数据)
    renderTaskRail();
    renderBatchTree();
    renderStreamTable();
    renderTaskTerms();
    renderSystemLogs();
    renderDomainList();
    renderDomainCsvTable();
    renderDictSearch('', true);
    updateAssPreview();

    (async () => {
      await loadBackendConfig();
      await loadBackendGlossaryDomains();
      await loadBackendTasks();
      connectEventStream();
    })();
