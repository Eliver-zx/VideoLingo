// =========================================================================
    // 0. API Client & State Synchronization
    // =========================================================================
    let isBackendConnected = false;
    let activeGlossaryDomain = 'destiny2';
    let domainTermsMap = { destiny2: [...REAL_TASK_TERMS] };
    let domainDictEntries = [...REAL_DICT_ENTRIES];
    let domainListMeta = [
      { domain: 'destiny2', terms_count: 8, has_dict: true, dict_count: 42810 },
      { domain: 'ai_ml', terms_count: 8, has_dict: false, dict_count: 0 },
    ];
    let currentConfig = {};

    async function apiFetch(url, options = {}) {
      const headers = { ...(options.headers || {}) };
      if (options.body && !(options.body instanceof FormData) && !headers['Content-Type']) {
        headers['Content-Type'] = 'application/json';
      }
      const res = await fetch(url, { ...options, headers });
      if (!res.ok) {
        let msg = res.statusText;
        try {
          const data = await res.json();
          msg = data.detail || data.message || JSON.stringify(data);
        } catch (_) {}
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
          activeTaskId = card.getAttribute('data-task-id');
          await selectActiveTask(activeTaskId);
        });
      });
    }

    async function selectActiveTask(taskId) {
      const task = REAL_TASKS.find(x => x.id === taskId);
      if (task) {
        document.getElementById('detail-task-title').textContent = task.title;
        document.getElementById('detail-domain-pill').textContent = task.domain || '无领域';
        const domSel = document.getElementById('detail-domain-select');
        if (domSel) domSel.value = task.domain || '';
        if (task.fromBackend) {
          await loadBackendTaskDetail(task.id);
        } else if (task.state === 'paused') {
          applyTaskStateSimulation('paused_terms');
        } else {
          applyTaskStateSimulation('srt_ready');
        }
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

    async function loadBackendTasks() {
      try {
        const data = await apiFetch('/api/tasks');
        isBackendConnected = true;
        const tasks = (data.tasks || []).map(mapBackendTaskToCard);
        if (tasks.length > 0) {
          REAL_TASKS.splice(0, REAL_TASKS.length, ...tasks);
          if (!REAL_TASKS.some(t => t.id === activeTaskId)) {
            activeTaskId = REAL_TASKS[0].id;
          }
          renderTaskRail();
          await selectActiveTask(activeTaskId);
        }
      } catch (_) {}
    }

    async function loadBackendTaskDetail(taskId) {
      try {
        const [taskData, termsData, uttsData] = await Promise.all([
          apiFetch(`/api/tasks/${encodeURIComponent(taskId)}`),
          apiFetch(`/api/tasks/${encodeURIComponent(taskId)}/terms`),
          apiFetch(`/api/tasks/${encodeURIComponent(taskId)}/utterances`),
        ]);
        const card = mapBackendTaskToCard(taskData);
        const idx = REAL_TASKS.findIndex(x => x.id === taskId);
        if (idx >= 0) REAL_TASKS[idx] = card;

        // Populate terms
        if (Array.isArray(termsData.terms) && termsData.terms.length > 0) {
          REAL_TASK_TERMS.splice(
            0,
            REAL_TASK_TERMS.length,
            ...termsData.terms.map(item => ({
              src: item.src || '',
              tgt: item.tgt || '',
              aliases: Array.isArray(item.aliases) ? item.aliases : [],
              note: item.note || '',
              fix_source: Boolean(item.fix_source),
              promote: Boolean(item.promote_to_domain ?? item.promote ?? false),
            }))
          );
        }
        const themeInput = document.getElementById('terms-theme-input');
        if (themeInput && termsData.summary) {
          themeInput.value = termsData.summary;
        }

        // Populate utterances into STREAM_TABLE_ROWS if present
        const utts = uttsData.utterances || [];
        if (utts.length > 0) {
          const batchCount = Math.max(1, Math.ceil(utts.length / 15));
          const rows = utts.map((u, i) => {
            const bIdx = Math.floor(i / 15) + 1;
            const bId = `batch-${String(bIdx).padStart(2, '0')}`;
            const numStr = `#${String(i + 1).padStart(3, '0')}`;
            const tcStr = `${formatTimecode(u.start)} → ${formatTimecode(u.end)}`;
            const hasCues = Array.isArray(u.sub_cues) && u.sub_cues.length > 1;
            return {
              idx: numStr,
              stage: hasCues ? 'split' : 'translate',
              batchId: bId,
              isSplit: hasCues,
              tc: tcStr,
              meta: hasCues ? `B${String(bIdx).padStart(2, '0')} · 拆为 ${u.sub_cues.length} 轴` : `B${String(bIdx).padStart(2, '0')}`,
              srcHtml: escapeHtml(u.src || ''),
              tgtHtml: escapeHtml(u.tgt || ''),
              uttIndex: i,
              rawUtt: u,
            };
          });
          STREAM_TABLE_ROWS.splice(0, STREAM_TABLE_ROWS.length, ...rows);

          // Rebuild batch tree nodes dynamically to match actual batches
          const baseNodes = [
            { id: 'all', kind: 'summary', step: 'ALL', title: '全部字幕', meta: `${utts.length} 句`, time: '', desc: `全部字幕 (${utts.length} 句)` },
            { id: 'asr-01', kind: 'step', step: '01', title: '语音识别', meta: '已完成', time: '', desc: '语音识别' },
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
          document.getElementById('pill-badge-4').textContent = `${batchCount}/${batchCount}批`;
        }

        renderTaskTerms();
        if (card.state === 'paused' && card.stateLabel === '待审术语') {
          applyTaskStateSimulation('paused_terms');
        } else {
          applyTaskStateSimulation('srt_ready');
          document.getElementById('pill-badge-4').textContent = `${Math.max(1, Math.ceil((utts.length || 189) / 15))}/${Math.max(1, Math.ceil((utts.length || 189) / 15))}批`;
        }
      } catch (_) {}
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
    ['dragenter', 'dragover'].forEach(ev => {
      dropWell.addEventListener(ev, e => {
        e.preventDefault();
        dropWell.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach(ev => {
      dropWell.addEventListener(ev, e => {
        e.preventDefault();
        dropWell.classList.remove('dragover');
        if (ev === 'drop' && e.dataTransfer?.files?.length) {
          droppedFile = e.dataTransfer.files[0];
          document.getElementById('input-source-path').value = droppedFile.name;
        }
      });
    });

    document.getElementById('btn-pick-local')?.addEventListener('click', async () => {
      try {
        const res = await apiFetch('/api/dialogs/pick-files', { method: 'POST' });
        if (res.paths && res.paths.length > 0) {
          droppedFile = null;
          document.getElementById('input-source-path').value = res.paths[0];
        }
      } catch (_) {}
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

    document.getElementById('btn-create-task').addEventListener('click', async () => {
      const rawPath = document.getElementById('input-source-path').value.trim() || 'New_Media_Task.mp4';
      const subPath = document.getElementById('input-sub-path').value.trim() || null;
      const srcLang = document.getElementById('task-src-lang').value || 'en';
      const tgtLang = document.getElementById('task-tgt-lang').value.trim() || '简体中文';
      const domain = document.getElementById('task-domain').value;
      const scopeRadio = document.querySelector('input[name="task-scope"]:checked');
      const stage = scopeRadio ? scopeRadio.value : 'subtitles';
      const pauseBefore = document.getElementById('task-pause-terms-chk').checked;

      if (isBackendConnected) {
        try {
          await apiFetch('/api/config', {
            method: 'PUT',
            body: JSON.stringify({ updates: { pause_before_translate: pauseBefore } }),
          });
          let created = null;
          if (droppedFile && droppedFile.name === rawPath) {
            const form = new FormData();
            form.append('file', droppedFile);
            form.append('source_language', srcLang);
            form.append('target_language', tgtLang);
            form.append('glossary_domain', domain);
            form.append('stage', stage);
            form.append('autorun', 'true');
            created = await apiFetch('/api/tasks/upload', { method: 'POST', body: form });
            droppedFile = null;
          } else {
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
            closeTopLayer(newTaskDialog);
            appendLog('asr', 'ASR', `Created workspace output/${created.task_id}/ and started pipeline...`, false);
            await loadBackendTasks();
            return;
          }
        } catch (err) {
          appendLog('warn', 'WARN', `任务创建提示: ${err.message}`, true);
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
        return (
          r.idx.toLowerCase().includes(query) ||
          r.srcHtml.toLowerCase().includes(query) ||
          r.tgtHtml.toLowerCase().includes(query) ||
          r.meta.toLowerCase().includes(query)
        );
      });

      document.getElementById('stream-row-counter').textContent = query
        ? `${filtered.length} / ${STREAM_TABLE_ROWS.length} 句`
        : `共 ${filtered.length} 句`;

      tbody.innerHTML = filtered.map(r => `
        <tr>
          <td>
            <span class="sub-idx">${r.idx}</span>
          </td>
          <td>
            <div class="sub-tc">
              <span>${r.tc}</span>
              <span class="field-hint">${r.meta}</span>
            </div>
          </td>
          <td class="sub-src">${r.srcHtml}</td>
          <td class="sub-tgt">${r.tgtHtml}</td>
        </tr>
      `).join('');
    }

    document.getElementById('subtitle-stream-search').addEventListener('input', renderStreamTable);

    // =========================================================================
    // 5. 状态模拟切换 (字幕就绪 vs 待审术语)
    // =========================================================================
    const btnSimSrt = document.getElementById('btn-sim-srt-ready');
    const btnSimTerms = document.getElementById('btn-sim-checkpoint-terms');
    const detailBadge = document.getElementById('detail-task-badge');
    const pill3 = document.getElementById('pill-stage-3');
    const pill4 = document.getElementById('pill-stage-4');
    const line3 = document.getElementById('pipe-line-3');

    function applyTaskStateSimulation(mode) {
      if (mode === 'paused_terms') {
        btnSimTerms.classList.add('active');
        btnSimSrt.classList.remove('active');
        detailBadge.className = 'status-pill paused';
        detailBadge.innerHTML = '<span class="status-dot"></span><span>待审术语</span>';
        pill3.className = 'pipe-node paused';
        document.getElementById('pill-badge-3').textContent = '待确认';
        if (line3) line3.className = 'pipe-line';
        pill4.className = 'pipe-node';
        document.getElementById('pill-badge-4').textContent = '等待中';
        activeBatchId = 'terms-03';
        renderBatchTree();
        syncDetailPane();
      } else {
        btnSimSrt.classList.add('active');
        btnSimTerms.classList.remove('active');
        detailBadge.className = 'status-pill completed';
        detailBadge.innerHTML = '<span class="status-dot"></span><span>字幕就绪</span>';
        pill3.className = 'pipe-node done';
        document.getElementById('pill-badge-3').textContent = `${REAL_TASK_TERMS.length}条`;
        if (line3) line3.className = 'pipe-line done';
        pill4.className = 'pipe-node active';
        document.getElementById('pill-badge-4').textContent = '13/13批';
        if (activeBatchId === 'terms-03') activeBatchId = 'all';
        renderBatchTree();
        syncDetailPane();
        renderStreamTable();
      }
    }

    btnSimSrt.addEventListener('click', () => applyTaskStateSimulation('srt_ready'));
    btnSimTerms.addEventListener('click', () => applyTaskStateSimulation('paused_terms'));
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

      tbody.innerHTML = filtered.map(({ item: t, idx }) => `
        <tr data-term-row="${idx}">
          <td><input type="text" class="cell-input mono" value="${escapeAttr(t.src)}" data-term-field="src" data-idx="${idx}"></td>
          <td><input type="text" class="cell-input" style="font-weight:600; color:var(--brand-bright);" value="${escapeAttr(t.tgt)}" data-term-field="tgt" data-idx="${idx}"></td>
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
          <td><input type="text" class="cell-input" value="${escapeAttr(t.note)}" data-term-field="note" data-idx="${idx}"></td>
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
          renderTaskTerms();
          renderDomainCsvTable();
        });
      });

      tbody.querySelectorAll('[data-add-alias]').forEach(inp => {
        inp.addEventListener('keydown', e => {
          if (e.key === 'Enter' && inp.value.trim()) {
            e.preventDefault();
            const rIdx = Number(inp.getAttribute('data-add-alias'));
            REAL_TASK_TERMS[rIdx].aliases.push(inp.value.trim());
            renderTaskTerms();
            renderDomainCsvTable();
          }
        });
      });

      tbody.querySelectorAll('[data-term-field]').forEach(inp => {
        inp.addEventListener('input', () => {
          const rIdx = Number(inp.getAttribute('data-idx'));
          const field = inp.getAttribute('data-term-field');
          REAL_TASK_TERMS[rIdx][field] = inp.value;
        });
      });

      tbody.querySelectorAll('[data-term-fix]').forEach(chk => {
        chk.addEventListener('change', () => {
          const rIdx = Number(chk.getAttribute('data-term-fix'));
          REAL_TASK_TERMS[rIdx].fix_source = chk.checked;
        });
      });

      tbody.querySelectorAll('[data-term-promote]').forEach(chk => {
        chk.addEventListener('change', () => {
          const rIdx = Number(chk.getAttribute('data-term-promote'));
          REAL_TASK_TERMS[rIdx].promote = chk.checked;
        });
      });
    }

    document.getElementById('task-terms-search').addEventListener('input', renderTaskTerms);

    document.getElementById('btn-batch-fix-on').addEventListener('click', () => {
      const allOn = REAL_TASK_TERMS.every(t => t.fix_source);
      REAL_TASK_TERMS.forEach(t => { t.fix_source = !allOn; });
      renderTaskTerms();
    });

    document.getElementById('btn-batch-promote-on').addEventListener('click', () => {
      const allOn = REAL_TASK_TERMS.every(t => t.promote);
      REAL_TASK_TERMS.forEach(t => { t.promote = !allOn; });
      renderTaskTerms();
    });

    document.getElementById('btn-add-task-term').addEventListener('click', () => {
      REAL_TASK_TERMS.unshift({
        src: 'Envious Arsenal',
        tgt: '嫉妒兵工厂',
        aliases: ['Envious'],
        note: '《命运2》武器换弹perk',
        fix_source: true,
        promote: true
      });
      renderTaskTerms();
      renderDomainCsvTable();
    });

    async function persistTaskTermsToBackend() {
      const task = REAL_TASKS.find(x => x.id === activeTaskId);
      if (!isBackendConnected || !task?.fromBackend) return;
      const summary = document.getElementById('terms-theme-input')?.value?.trim() || '';
      const glossary_domain = document.getElementById('detail-domain-select')?.value ?? task.domain ?? '';
      const payloadTerms = REAL_TASK_TERMS.map(t => ({
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
        } catch (_) {}
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
        } catch (_) {}
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
        } catch (_) {}
      }
      activeBatchId = 'step-04-group';
      applyTaskStateSimulation('srt_ready');
      appendLog('llm', 'TERMS', 'Saved output/log/terms.json (promoted 8 terms to glossary/destiny2/terms.csv) and resumed Step 04 translation...', false);
    });

    document.getElementById('btn-save-terms-only').addEventListener('click', async () => {
      await persistTaskTermsToBackend();
      appendLog('llm', 'TERMS', 'Saved output/log/terms.json without advancing pipeline checkpoint.', false);
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
      box.innerHTML = list.map(item => `
        <div class="log-row ${item.warn ? 'warn' : ''}">
          <span class="log-time">${escapeHtml(item.time)}</span>
          <span class="log-tag ${escapeAttr(item.tag)}">${escapeHtml(item.label)}</span>
          <span class="log-msg">${escapeHtml(item.msg)}</span>
        </div>
      `).join('');
      box.scrollTop = box.scrollHeight;
      const previewEl = document.getElementById('console-latest-preview');
      const lastItem = REAL_SYSTEM_LOGS[REAL_SYSTEM_LOGS.length - 1];
      if (previewEl && lastItem) {
        previewEl.textContent = `${lastItem.time} · [${lastItem.label}] ${lastItem.msg}`;
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
      if (isBackendConnected && task?.fromBackend) {
        try {
          await apiFetch(`/api/tasks/${encodeURIComponent(activeTaskId)}/open-dir`, { method: 'POST' });
        } catch (_) {}
      }
      appendLog('llm', 'OPEN', `Opened local directory output/${activeTaskId}/ in Finder / Explorer for Subtitle Edit review.`, false);
    });

    document.getElementById('btn-burn-sub-video').addEventListener('click', async () => {
      document.getElementById('pill-stage-5').className = 'pipe-node active';
      document.getElementById('pill-badge-5').textContent = '烧录中…';
      appendLog('llm', 'MERGE', `Burning bilingual ASS subtitles into output/${activeTaskId}/output_sub.mp4...`, false);
      const task = REAL_TASKS.find(x => x.id === activeTaskId);
      if (isBackendConnected && task?.fromBackend) {
        try {
          await apiFetch(`/api/tasks/${encodeURIComponent(activeTaskId)}/run`, {
            method: 'POST',
            body: JSON.stringify({ stage: 'subtitles', dubbing: false }),
          });
          await loadBackendTasks();
        } catch (_) {}
      }
    });

    document.getElementById('btn-run-dubbing').addEventListener('click', async () => {
      document.getElementById('pill-stage-5').className = 'pipe-node active';
      document.getElementById('pill-badge-5').textContent = '配音中…';
      appendLog('llm', 'TTS', `Running Fish Audio TTS alignment from edited trans.srt -> output/${activeTaskId}/output_dub.mp4...`, false);
      const task = REAL_TASKS.find(x => x.id === activeTaskId);
      if (isBackendConnected && task?.fromBackend) {
        try {
          await apiFetch(`/api/tasks/${encodeURIComponent(activeTaskId)}/run`, {
            method: 'POST',
            body: JSON.stringify({ stage: 'dubbing', dubbing: true }),
          });
          await loadBackendTasks();
        } catch (_) {}
      }
    });

    document.getElementById('btn-stop-task')?.addEventListener('click', async () => {
      const task = REAL_TASKS.find(x => x.id === activeTaskId);
      if (isBackendConnected && task?.fromBackend) {
        try {
          await apiFetch(`/api/tasks/${encodeURIComponent(activeTaskId)}/stop`, { method: 'POST' });
          await loadBackendTasks();
        } catch (_) {}
      }
      appendLog('warn', 'STOP', `Requested stop for task ${activeTaskId}.`, true);
    });

    const previewDialog = document.getElementById('video-preview-dialog');
    wireModalDialog(previewDialog);
    document.getElementById('btn-preview-video-modal').addEventListener('click', () => {
      previewDialog.showModal();
    });
    document.getElementById('btn-close-preview-modal').addEventListener('click', () => {
      closeTopLayer(previewDialog);
    });

    // =========================================================================
    // 9. 领域词表中心 (terms.csv + dict.json 检索与转入)
    // =========================================================================
    function getActiveDomainTerms() {
      if (!domainTermsMap[activeGlossaryDomain]) {
        domainTermsMap[activeGlossaryDomain] = activeGlossaryDomain === 'destiny2' ? [...REAL_TASK_TERMS] : [];
      }
      return activeGlossaryDomain === 'destiny2' ? REAL_TASK_TERMS : domainTermsMap[activeGlossaryDomain];
    }

    function renderDomainList() {
      const box = document.getElementById('domain-list-box');
      if (!box) return;
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

      const activeMeta = domainListMeta.find(d => d.domain === activeGlossaryDomain);
      const dictPill = document.getElementById('dict-count-pill');
      if (dictPill) {
        if (activeMeta?.has_dict) {
          dictPill.className = 'status-pill completed';
          dictPill.innerHTML = `<span class="status-dot"></span><span>${Number(activeMeta.dict_count || 42810).toLocaleString()} 词</span>`;
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
        const curCfgVal = cfgDomainSelect.value;
        cfgDomainSelect.innerHTML = [
          '<option value="">不挂载</option>',
          ...domainListMeta.map(d => `<option value="${escapeAttr(d.domain)}">${escapeHtml(d.domain)}</option>`),
        ].join('');
        if ([...cfgDomainSelect.options].some(o => o.value === curCfgVal)) {
          cfgDomainSelect.value = curCfgVal;
        }
      }

      const ntMenuList = document.getElementById('nt-domain-menu-list');
      if (ntMenuList && domainListMeta.length > 0) {
        const curTaskDom = document.getElementById('task-domain')?.value ?? 'destiny2';
        const subLabels = { destiny2: '命运2', ai_ml: '机器学习', poe2: '流放之路2', unreal5: '虚幻引擎5' };
        ntMenuList.innerHTML = [
          ...domainListMeta.map(d => {
            const sub = subLabels[d.domain] || `${d.terms_count || 0}词`;
            const isAct = d.domain === curTaskDom ? 'active' : '';
            return `
              <button type="button" class="nt-menu-item ${isAct}" role="option" aria-selected="${Boolean(isAct)}" data-value="${escapeAttr(d.domain)}" data-sub="${escapeAttr(sub)}">
                <span class="nt-item-main mono">${escapeHtml(d.domain)}</span>
                <span class="nt-item-sub">${escapeHtml(sub)}</span>
              </button>
            `;
          }),
          '<div class="nt-menu-sep" role="separator"></div>',
          `<button type="button" class="nt-menu-item ${curTaskDom === '' ? 'active' : ''}" role="option" aria-selected="${curTaskDom === ''}" data-value="" data-sub="通用">
            <span class="nt-item-main">不挂载词表</span>
            <span class="nt-item-sub">通用</span>
          </button>`
        ].join('');
      }
    }

    async function selectGlossaryDomain(dom) {
      activeGlossaryDomain = dom;
      document.getElementById('glossary-active-title').textContent = `${dom} · 自定义术语`;
      if (isBackendConnected) {
        try {
          const data = await apiFetch(`/api/glossary/domains/${encodeURIComponent(dom)}`);
          const terms = (data.terms || []).map(t => ({
            src: t.src || '',
            tgt: t.tgt || '',
            aliases: Array.isArray(t.aliases) ? t.aliases : [],
            note: t.note || '',
            fix_source: false,
            promote: true,
          }));
          if (dom === 'destiny2' && terms.length > 0) {
            REAL_TASK_TERMS.splice(0, REAL_TASK_TERMS.length, ...terms);
          }
          domainTermsMap[dom] = terms.length > 0 ? terms : (domainTermsMap[dom] || []);
        } catch (_) {}
      }
      renderDomainList();
      renderDomainCsvTable();
      await loadDomainDictEntries(dom, document.getElementById('dict-search-input')?.value || '');
    }

    async function loadBackendGlossaryDomains() {
      try {
        const data = await apiFetch('/api/glossary/domains');
        isBackendConnected = true;
        if (Array.isArray(data.domains) && data.domains.length > 0) {
          domainListMeta = data.domains.map(d => ({
            domain: d.domain || d.name,
            terms_count: d.terms_count ?? 0,
            has_dict: Boolean(d.has_dict),
            dict_count: d.dict_count ?? (d.has_dict ? 42810 : 0),
          }));
          if (!domainListMeta.some(x => x.domain === activeGlossaryDomain)) {
            activeGlossaryDomain = domainListMeta[0].domain;
          }
          await selectGlossaryDomain(activeGlossaryDomain);
        }
      } catch (_) {}
    }

    async function loadDomainDictEntries(dom, keyword = '') {
      if (isBackendConnected) {
        try {
          const data = await apiFetch(`/api/glossary/domains/${encodeURIComponent(dom)}/dict?q=${encodeURIComponent(keyword)}`);
          if (Array.isArray(data.entries)) {
            domainDictEntries = data.entries;
            renderDictSearch(keyword, true);
            return;
          }
          if (data.entries === null) {
            domainDictEntries = dom === 'destiny2' ? [...REAL_DICT_ENTRIES] : [];
            renderDictSearch(keyword, dom !== 'destiny2');
            return;
          }
        } catch (_) {}
      }
      domainDictEntries = dom === 'destiny2' ? [...REAL_DICT_ENTRIES] : [];
      renderDictSearch(keyword, dom !== 'destiny2');
    }

    function renderDomainCsvTable() {
      const tbody = document.getElementById('domain-csv-tbody');
      const q = (document.getElementById('glossary-csv-search').value || '').trim().toLowerCase();
      const list = getActiveDomainTerms();
      const rows = list.map((t, idx) => ({ t, idx })).filter(({ t }) => {
        if (!q) return true;
        return t.src.toLowerCase().includes(q) || t.tgt.toLowerCase().includes(q) || t.aliases.join('|').toLowerCase().includes(q);
      });
      tbody.innerHTML = rows.map(({ t, idx }) => `
        <tr>
          <td><input type="text" class="cell-input mono" value="${escapeAttr(t.src)}" data-csv-field="src" data-idx="${idx}"></td>
          <td><input type="text" class="cell-input" style="font-weight:600; color:var(--brand-bright);" value="${escapeAttr(t.tgt)}" data-csv-field="tgt" data-idx="${idx}"></td>
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
          <td><input type="text" class="cell-input" value="${escapeAttr(t.note)}" data-csv-field="note" data-idx="${idx}"></td>
          <td><button type="button" class="danger xs" data-del-csv="${idx}">删除</button></td>
        </tr>
      `).join('');

      tbody.querySelectorAll('[data-del-csv-alias]').forEach(btn => {
        btn.addEventListener('click', () => {
          const [rIdx, aIdx] = btn.getAttribute('data-del-csv-alias').split(':').map(Number);
          list[rIdx].aliases.splice(aIdx, 1);
          renderDomainCsvTable();
          renderTaskTerms();
        });
      });

      tbody.querySelectorAll('[data-add-csv-alias]').forEach(inp => {
        inp.addEventListener('keydown', e => {
          if (e.key === 'Enter' && inp.value.trim()) {
            e.preventDefault();
            const rIdx = Number(inp.getAttribute('data-add-csv-alias'));
            list[rIdx].aliases.push(inp.value.trim());
            renderDomainCsvTable();
            renderTaskTerms();
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
          renderTaskTerms();
        });
      });
    }

    document.getElementById('glossary-csv-search').addEventListener('input', renderDomainCsvTable);

    document.getElementById('btn-add-domain-term').addEventListener('click', () => {
      const list = getActiveDomainTerms();
      list.unshift({ src: 'Chill Clip', tgt: '寒意弹夹', aliases: ['Chill'], note: '《命运2》冰影减速perk', fix_source: false, promote: true });
      renderDomainCsvTable();
      renderTaskTerms();
    });

    document.getElementById('btn-save-domain-csv')?.addEventListener('click', async () => {
      const list = getActiveDomainTerms();
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
        } catch (_) {}
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
      if (!file) return;
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
        } catch (_) {}
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
        } catch (_) {}
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
        } catch (_) {}
      }
      domainListMeta = domainListMeta.filter(d => d.domain !== activeGlossaryDomain);
      activeGlossaryDomain = domainListMeta[0]?.domain || 'destiny2';
      renderDomainList();
      renderDomainCsvTable();
    });

    function renderDictSearch(keyword = '', prefiltered = false) {
      const tbody = document.getElementById('dict-search-tbody');
      const q = keyword.trim().toLowerCase();
      const sourceList = prefiltered ? domainDictEntries : (domainDictEntries.length > 0 ? domainDictEntries : REAL_DICT_ENTRIES);
      const matched = prefiltered
        ? sourceList
        : sourceList.filter(x => !q || x.k.toLowerCase().includes(q) || x.v.toLowerCase().includes(q));
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
            renderTaskTerms();
          }
        });
      });
    }

    document.getElementById('dict-search-input').addEventListener('input', e => {
      loadDomainDictEntries(activeGlossaryDomain, e.target.value);
    });

    document.querySelectorAll('#domain-list-box .domain-item').forEach(card => {
      card.addEventListener('click', async () => {
        const dom = card.getAttribute('data-domain');
        await selectGlossaryDomain(dom);
      });
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
      document.getElementById('ass-tr-size').value = '17';
      document.getElementById('ass-tr-color').value = '#FFFF00';
      document.getElementById('ass-tr-color-picker').value = '#ffff00';
      document.getElementById('ass-tr-margin').value = '27';
      document.getElementById('ass-src-size').value = '15';
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
      } catch (_) {}
      btn.textContent = '探测';
    });

    document.getElementById('btn-reset-default-config')?.addEventListener('click', async () => {
      const defaults = {
        max_workers: 4,
        target_language: '简体中文',
        summary_length: 8000,
        reflect_translate: false,
        pause_before_translate: true,
        pause_after_translate: true,
        'whisper.language': 'en',
        'whisper.runtime': 'local',
        'whisper.backend': 'qwen',
        'whisper.qwen_model': '1.7b',
        'whisper.qwen_engine': 'auto',
        burn_subtitles: true,
        ffmpeg_gpu: false,
        'subtitle.max_length': 75,
        'subtitle.target_multiplier': 1.3,
        'subtitle.sub_video_subtitles': 'bilingual',
        'subtitle.dub_video_subtitles': 'translation',
        'speed_factor.min': 1.0,
        'speed_factor.accept': 1.2,
        'speed_factor.max': 1.4,
        min_subtitle_duration: 2.5,
        tolerance: 1.5,
        ytb_resolution: '1080',
        'audio.raw_sample_rate': 32000,
        'audio.raw_bitrate': '128k',
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
        } catch (_) {}
      }
      setAutosaveStatus('completed', '已恢复默认');
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
      const statusEl = document.getElementById('llm-check-status');
      if (statusEl) statusEl.textContent = '正在检测连通性…';
      try {
        const res = await apiFetch('/api/llm/check', {
          method: 'POST',
          body: JSON.stringify({ base_url: baseUrl, api_key: apiKey, model }),
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
            pendingConfigUpdates[path] = readCfgControlValue(inp);
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
              } catch (_) {}
            }
            setAutosaveStatus('completed', '已保存');
          }, 400);
        };
        inp.addEventListener('input', handler);
        inp.addEventListener('change', handler);
      });
    }
    bindCfgControls();

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
          } else {
            inp.value = String(val);
          }
        });
        syncAsrRuntimePanel();
        syncTtsMethodPanel();
        updateAssPreview();
      } catch (_) {}
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
            loadBackendTasks();
          } catch (_) {}
        };
      } catch (_) {}
    }

    // 初始化全量视图 (同步立即渲染，随后异步探测后端数据)
    renderTaskRail();
    renderBatchTree();
    renderStreamTable();
    renderTaskTerms();
    renderSystemLogs();
    renderDomainCsvTable();
    renderDictSearch('Attrition');
    updateAssPreview();

    (async () => {
      await loadBackendConfig();
      await loadBackendGlossaryDomains();
      await loadBackendTasks();
      connectEventStream();
    })();
