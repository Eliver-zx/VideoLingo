import { state } from './state.js';
import { apiFetch, setAutosaveStatus, closeTopLayer, wireModalDialog } from './dom-utils.js';
import { appendLog, reportActionError } from './console-log.js';
import { handleBurnDroppedFiles } from './burn-view.js';
import { loadBackendTasks } from './task-rail.js';

let droppedFile = null;
let droppedSubFile = null;

function closeAllNtDropdowns(newTaskDialog, except = null) {
  newTaskDialog?.querySelectorAll('.nt-dropdown.open').forEach(dd => {
    if (dd !== except) {
      dd.classList.remove('open');
      dd.querySelector('.nt-select-trigger, .nt-combo-toggle')?.setAttribute(
        'aria-expanded',
        'false'
      );
    }
  });
}

function wireNtDropdown(newTaskDialog, dropdownEl, onSelect) {
  if (!dropdownEl) return;
  const trigger = dropdownEl.querySelector('.nt-select-trigger');
  const menu = dropdownEl.querySelector('.nt-menu');
  if (!trigger || !menu) return;
  trigger.addEventListener('click', e => {
    e.stopPropagation();
    const willOpen = !dropdownEl.classList.contains('open');
    closeAllNtDropdowns(newTaskDialog, dropdownEl);
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

function syncSubClearButton() {
  const subInput = document.getElementById('input-sub-path');
  const clearBtn = document.getElementById('btn-clear-sub');
  if (clearBtn && subInput) {
    clearBtn.classList.toggle('hidden', !subInput.value.trim());
  }
}

function syncNewTaskPauseCheckbox() {
  const scope = document.querySelector('input[name="task-scope"]:checked')?.value || 'subtitles';
  const chk = document.getElementById('task-pause-terms-chk');
  const wrap = document.getElementById('nt-pause-terms-wrap');
  const labelEl = document.getElementById('nt-pause-terms-text');
  const irrelevant = scope === 'transcribe' || scope === 'dubbing';
  if (chk) chk.disabled = irrelevant;
  wrap?.classList.toggle('is-disabled', irrelevant);
  if (labelEl) {
    labelEl.textContent = irrelevant ? '当前产出范围不涉及术语翻译' : '翻译前暂停，人工审校术语';
  }
}

export function initIntakeModalEvents() {
  const newTaskDialog = document.getElementById('new-task-dialog');
  const dropWell = document.getElementById('upload-dropzone');
  wireModalDialog(newTaskDialog);

  wireNtDropdown(newTaskDialog, document.getElementById('nt-src-lang-dropdown'), item => {
    document.getElementById('task-src-lang').value = item.dataset.value || 'auto';
    document.getElementById('nt-src-lang-label').textContent =
      item.dataset.label || item.textContent.trim();
  });

  wireNtDropdown(newTaskDialog, document.getElementById('nt-domain-dropdown'), item => {
    const val = item.dataset.value ?? '';
    const sub = item.dataset.sub || '';
    const taskDomInput = document.getElementById('task-domain');
    if (taskDomInput) {
      taskDomInput.value = val;
      taskDomInput.dataset.userEdited = '1';
    }
    const slugEl = document.getElementById('nt-domain-slug');
    slugEl.textContent = val || '不挂载词表';
    slugEl.classList.toggle('mono', Boolean(val));
    document.getElementById('nt-domain-sub').textContent = sub;
  });

  const tgtCombo = document.getElementById('nt-tgt-lang-dropdown');
  const tgtInput = document.getElementById('task-tgt-lang');
  if (tgtCombo && tgtInput) {
    const toggleBtn = tgtCombo.querySelector('.nt-combo-toggle');
    const menu = tgtCombo.querySelector('.nt-menu');
    toggleBtn?.addEventListener('click', e => {
      e.stopPropagation();
      const willOpen = !tgtCombo.classList.contains('open');
      closeAllNtDropdowns(newTaskDialog, tgtCombo);
      tgtCombo.classList.toggle('open', willOpen);
      toggleBtn.setAttribute('aria-expanded', String(willOpen));
    });
    menu?.addEventListener('click', e => {
      const item = e.target.closest('.nt-menu-item');
      if (!item) return;
      tgtInput.value = item.dataset.value || item.textContent.trim();
      tgtInput.dataset.userEdited = '1';
      menu.querySelectorAll('.nt-menu-item').forEach(btn => {
        btn.classList.toggle('active', btn === item);
      });
      tgtCombo.classList.remove('open');
      toggleBtn?.setAttribute('aria-expanded', 'false');
    });
    tgtInput.addEventListener('input', () => {
      tgtInput.dataset.userEdited = '1';
      const cur = tgtInput.value.trim();
      menu?.querySelectorAll('.nt-menu-item').forEach(btn => {
        btn.classList.toggle('active', (btn.dataset.value || '') === cur);
      });
    });
  }

  newTaskDialog?.addEventListener('click', e => {
    if (!e.target.closest('.nt-dropdown')) closeAllNtDropdowns(newTaskDialog);
  });

  const localVideoFileInput = document.createElement('input');
  localVideoFileInput.type = 'file';
  localVideoFileInput.accept = 'video/*,audio/*,.mp4,.mkv,.mov,.webm,.avi,.mp3,.wav,.m4a,.flac,.srt';
  localVideoFileInput.style.display = 'none';
  document.body.appendChild(localVideoFileInput);

  const localSubFileInput = document.createElement('input');
  localSubFileInput.type = 'file';
  localSubFileInput.accept = '.srt';
  localSubFileInput.style.display = 'none';
  document.body.appendChild(localSubFileInput);

  const selectBrowserFile = file => {
    if (!file) return;
    droppedFile = file;
    const inputEl = document.getElementById('input-source-path');
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    const label = `${file.name} (${sizeMb} MB)`;
    inputEl.value = label;
    inputEl.dataset.droppedLabel = label;
    dropWell?.classList.add('has-file');
  };

  const selectSubtitleFile = file => {
    if (!file) return;
    droppedSubFile = file;
    const subEl = document.getElementById('input-sub-path');
    const sizeKb = Math.max(1, Math.round(file.size / 1024));
    const label = `${file.name} (${sizeKb} KB)`;
    subEl.value = label;
    subEl.dataset.droppedLabel = label;
    syncSubClearButton();
  };

  const handleDroppedFiles = fileList => {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    const srt = files.find(f => /\.srt$/i.test(f.name));
    const media = files.find(f => !/\.srt$/i.test(f.name));
    if (media) {
      selectBrowserFile(media);
      if (srt) selectSubtitleFile(srt);
    } else if (srt) {
      const hasPrimarySource = Boolean(
        droppedFile || document.getElementById('input-source-path')?.value.trim()
      );
      if (hasPrimarySource) selectSubtitleFile(srt);
      else selectBrowserFile(srt);
    }
  };

  document.getElementById('input-source-path')?.addEventListener('input', e => {
    const el = e.currentTarget;
    if (
      droppedFile &&
      el.value.trim() !== el.dataset.droppedLabel &&
      el.value.trim() !== droppedFile.name
    ) {
      droppedFile = null;
      delete el.dataset.droppedLabel;
    }
    dropWell?.classList.toggle('has-file', Boolean(droppedFile || el.value.trim()));
  });

  document.getElementById('input-sub-path')?.addEventListener('input', e => {
    const el = e.currentTarget;
    if (
      droppedSubFile &&
      el.value.trim() !== el.dataset.droppedLabel &&
      el.value.trim() !== droppedSubFile.name
    ) {
      droppedSubFile = null;
      delete el.dataset.droppedLabel;
    }
    syncSubClearButton();
  });

  document.getElementById('btn-clear-sub')?.addEventListener('click', () => {
    const subEl = document.getElementById('input-sub-path');
    droppedSubFile = null;
    if (subEl) {
      subEl.value = '';
      delete subEl.dataset.droppedLabel;
    }
    syncSubClearButton();
  });

  localVideoFileInput.addEventListener('change', () => {
    if (localVideoFileInput.files?.length) selectBrowserFile(localVideoFileInput.files[0]);
    localVideoFileInput.value = '';
  });

  localSubFileInput.addEventListener('change', () => {
    if (localSubFileInput.files?.length) selectSubtitleFile(localSubFileInput.files[0]);
    localSubFileInput.value = '';
  });

  window.addEventListener('dragover', e => {
    if (e.dataTransfer?.types?.includes('Files')) e.preventDefault();
  });
  window.addEventListener('drop', e => {
    if (!e.dataTransfer?.files?.length) return;
    e.preventDefault();
    const burnViewEl = document.getElementById('view-burn');
    if (burnViewEl && !burnViewEl.classList.contains('hidden')) {
      handleBurnDroppedFiles(e.dataTransfer.files);
      return;
    }
    dropWell?.classList.remove('dragover');
    if (newTaskDialog && !newTaskDialog.open) {
      closeAllNtDropdowns(newTaskDialog);
      newTaskDialog.showModal();
    }
    handleDroppedFiles(e.dataTransfer.files);
  });

  if (newTaskDialog) {
    ['dragenter', 'dragover'].forEach(ev => {
      newTaskDialog.addEventListener(ev, e => {
        if (e.dataTransfer?.types?.includes('Files')) {
          e.preventDefault();
          dropWell?.classList.add('dragover');
        }
      });
    });
    newTaskDialog.addEventListener('dragleave', e => {
      if (!newTaskDialog.contains(e.relatedTarget)) dropWell?.classList.remove('dragover');
    });
    newTaskDialog.addEventListener('drop', e => {
      if (!e.dataTransfer?.files?.length) return;
      e.preventDefault();
      e.stopPropagation();
      dropWell?.classList.remove('dragover');
      if (e.target.closest('#sub-dropzone')) selectSubtitleFile(e.dataTransfer.files[0]);
      else handleDroppedFiles(e.dataTransfer.files);
    });
  }

  document.getElementById('btn-pick-local')?.addEventListener('click', async () => {
    if (state.isBackendConnected) {
      try {
        const res = await apiFetch('/api/dialogs/pick-files', {
          method: 'POST',
          body: JSON.stringify({ kind: 'media', multi: false }),
        });
        if (res.paths?.length) {
          droppedFile = null;
          const inputEl = document.getElementById('input-source-path');
          delete inputEl.dataset.droppedLabel;
          inputEl.value = res.paths[0];
          dropWell?.classList.add('has-file');
          return;
        }
        if (Array.isArray(res.paths)) return;
      } catch (err) {
        appendLog(
          'warn',
          'FILE',
          `调用系统文件选择器失败，已切换为浏览器文件选择: ${err.message}`,
          true,
          undefined,
          { taskId: '', persist: false }
        );
      }
    }
    localVideoFileInput.click();
  });

  document.getElementById('btn-pick-sub')?.addEventListener('click', async () => {
    if (state.isBackendConnected) {
      try {
        const res = await apiFetch('/api/dialogs/pick-files', {
          method: 'POST',
          body: JSON.stringify({ kind: 'subtitle', multi: false }),
        });
        if (res.paths?.length) {
          droppedSubFile = null;
          const subEl = document.getElementById('input-sub-path');
          delete subEl.dataset.droppedLabel;
          subEl.value = res.paths[0];
          syncSubClearButton();
          return;
        }
        if (Array.isArray(res.paths)) return;
      } catch (err) {
        appendLog(
          'warn',
          'FILE',
          `调用系统字幕选择器失败，已切换为浏览器文件选择: ${err.message}`,
          true,
          undefined,
          { taskId: '', persist: false }
        );
      }
    }
    localSubFileInput.click();
  });

  document.querySelectorAll('input[name="task-scope"]').forEach(radio => {
    radio.addEventListener('change', syncNewTaskPauseCheckbox);
  });
  syncNewTaskPauseCheckbox();

  document.getElementById('btn-toggle-new-task')?.addEventListener('click', () => {
    closeAllNtDropdowns(newTaskDialog);
    syncNewTaskPauseCheckbox();
    newTaskDialog?.showModal();
  });
  document.getElementById('btn-close-intake')?.addEventListener('click', () => {
    closeTopLayer(newTaskDialog);
  });
  document.getElementById('btn-cancel-intake')?.addEventListener('click', () => {
    closeTopLayer(newTaskDialog);
  });

  document.getElementById('btn-open-glossary-center')?.addEventListener('click', () => {
    const sourceVal = document.getElementById('input-source-path')?.value?.trim() || '';
    if ((sourceVal || droppedFile) && !window.confirm('离开将关闭新建窗口，确定前往词表管理吗？')) {
      return;
    }
    closeTopLayer(newTaskDialog);
    document.getElementById('tab-glossary')?.click();
  });

  document.getElementById('btn-create-task')?.addEventListener('click', async e => {
    const createBtn = e.currentTarget;
    const origBtnText = createBtn.textContent;
    const sourceInput = document.getElementById('input-source-path');
    const subInput = document.getElementById('input-sub-path');
    const rawPath = sourceInput.value.trim().replace(/^["']+|["']+$/g, '').trim();
    if (!rawPath && !droppedFile) {
      sourceInput.focus();
      setAutosaveStatus('error', '请选择媒体文件');
      return;
    }
    const cleanedSub = subInput.value.trim().replace(/^["']+|["']+$/g, '').trim();
    const subPath = cleanedSub || null;
    const srcLang = document.getElementById('task-src-lang').value || 'en';
    const tgtLang = document.getElementById('task-tgt-lang').value.trim() || '简体中文';
    const domain = document.getElementById('task-domain').value;
    const scopeRadio = document.querySelector('input[name="task-scope"]:checked');
    const stage = scopeRadio ? scopeRadio.value : 'subtitles';
    const pauseChk = document.getElementById('task-pause-terms-chk');

    if (state.isBackendConnected) {
      createBtn.disabled = true;
      try {
        if (pauseChk && !pauseChk.disabled) {
          await apiFetch('/api/config', {
            method: 'PUT',
            body: JSON.stringify({ updates: { pause_before_translate: pauseChk.checked } }),
          });
        }
        let created = null;
        if (droppedFile) {
          createBtn.textContent = '正在上传…';
          const form = new FormData();
          form.append('file', droppedFile);
          if (droppedSubFile) form.append('subtitles', droppedSubFile);
          else if (subPath) form.append('subtitles_path', subPath);
          form.append('source_language', srcLang);
          form.append('target_language', tgtLang);
          form.append('glossary_domain', domain);
          form.append('stage', stage);
          form.append('autorun', 'false');
          created = await apiFetch('/api/tasks/upload', { method: 'POST', body: form });
          droppedFile = null;
          droppedSubFile = null;
          delete sourceInput.dataset.droppedLabel;
          delete subInput.dataset.droppedLabel;
          dropWell?.classList.remove('has-file');
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
              autorun: false,
            }),
          });
        }
        if (created && created.task_id) {
          state.activeTaskId = created.task_id;
          sourceInput.value = '';
          subInput.value = '';
          dropWell?.classList.remove('has-file');
          syncSubClearButton();
          closeTopLayer(newTaskDialog);
          setAutosaveStatus('completed', '任务已创建');
          appendLog(
            'asr',
            'TASK',
            `已创建工作区 output/${created.task_id}/，点击「启动任务」开始执行。`,
            false,
            undefined,
            { taskId: created.task_id }
          );
          await loadBackendTasks();
        }
      } catch (err) {
        reportActionError('TASK', err);
      } finally {
        createBtn.disabled = false;
        createBtn.textContent = origBtnText;
      }
    }
  });
}
