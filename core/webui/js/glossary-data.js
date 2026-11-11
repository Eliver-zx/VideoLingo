import { state, TASK_DETAIL_CACHE } from './state.js';
import { apiFetch, escapeHtml, escapeAttr, setAutosaveStatus } from './dom-utils.js';
import { appendLog, reportActionError } from './console-log.js';

let glossaryAutosaveTimer = null;
let dictFetchController = null;
let onRenderTablesHook = null;

export function registerGlossaryRenderHook(fn) {
  onRenderTablesHook = fn;
}

export function getActiveDomainTerms() {
  if (!state.activeGlossaryDomain) return [];
  if (!state.domainTermsMap[state.activeGlossaryDomain]) {
    state.domainTermsMap[state.activeGlossaryDomain] = [];
  }
  return state.domainTermsMap[state.activeGlossaryDomain];
}

export function syncDomainSelectOptions() {
  const cfgSelect = document.querySelector('select[data-cfg="glossary_domain"]');
  const currentDefault = String(state.currentConfig.glossary_domain ?? 'destiny2');
  if (cfgSelect) {
    const opts = ['<option value="">不挂载词表</option>'];
    state.domainListMeta.forEach(d => {
      const sel = d.domain === currentDefault ? ' selected' : '';
      opts.push(`<option value="${escapeAttr(d.domain)}"${sel}>${escapeHtml(d.domain)}</option>`);
    });
    cfgSelect.innerHTML = opts.join('');
  }

  const menuList = document.getElementById('nt-domain-menu-list');
  const taskDomInput = document.getElementById('task-domain');
  const slugEl = document.getElementById('nt-domain-slug');
  const subEl = document.getElementById('nt-domain-sub');
  if (menuList && taskDomInput) {
    let selectedVal = taskDomInput.dataset.userEdited
      ? taskDomInput.value
      : state.currentConfig.glossary_domain !== undefined
        ? String(state.currentConfig.glossary_domain || '')
        : taskDomInput.value;
    let matched = state.domainListMeta.find(d => d.domain === selectedVal);
    if (
      selectedVal &&
      !matched &&
      (state.domainListMeta.length > 0 || state.isBackendConnected)
    ) {
      selectedVal = '';
      matched = undefined;
    }
    taskDomInput.value = selectedVal;
    const itemsHtml = state.domainListMeta.map(d => {
      const totalWords = Number(d.terms_count || 0) + Number(d.dict_count || 0);
      const subText = totalWords >= 10000 ? `${(totalWords / 10000).toFixed(1)}万词` : `${totalWords} 词`;
      const isAct = d.domain === selectedVal;
      return `
        <button type="button" class="nt-menu-item ${isAct ? 'active' : ''}" role="option" aria-selected="${isAct}" data-value="${escapeAttr(d.domain)}" data-sub="${escapeAttr(subText)}">
          <span class="nt-item-main mono">${escapeHtml(d.domain)}</span>
          <span class="nt-item-sub">${escapeHtml(subText)}</span>
        </button>
      `;
    });
    const noneAct = selectedVal === '';
    itemsHtml.push(
      '<div class="nt-menu-sep" role="separator"></div>',
      `<button type="button" class="nt-menu-item ${noneAct ? 'active' : ''}" role="option" aria-selected="${noneAct}" data-value="" data-sub="通用">
        <span class="nt-item-main">不挂载词表</span>
        <span class="nt-item-sub">通用</span>
      </button>`
    );
    menuList.innerHTML = itemsHtml.join('');
    if (slugEl && subEl) {
      if (matched) {
        const totalWords = Number(matched.terms_count || 0) + Number(matched.dict_count || 0);
        slugEl.textContent = matched.domain;
        slugEl.classList.add('mono');
        subEl.textContent = totalWords >= 10000 ? `${(totalWords / 10000).toFixed(1)}万词` : `${totalWords} 词`;
      } else {
        slugEl.textContent = '不挂载词表';
        slugEl.classList.remove('mono');
        subEl.textContent = '通用';
      }
    }
  }
}

export function renderDomainList() {
  const box = document.getElementById('domain-list-box');
  syncDomainSelectOptions();
  if (!box) return;
  if (state.domainListMeta.length === 0) {
    box.innerHTML = `
      <div class="rail-empty">
        <div>暂无领域词表</div>
        <div style="margin-block-start:4px;">点击上方「新建」创建专属词库</div>
      </div>
    `;
    return;
  }
  const defaultDom = String(state.currentConfig.glossary_domain ?? '');
  box.innerHTML = state.domainListMeta
    .map(d => {
      const active = d.domain === state.activeGlossaryDomain ? 'active' : '';
      const isDefault = d.domain === defaultDom;
      const dictLabel =
        d.dict_count >= 10000
          ? `${(d.dict_count / 10000).toFixed(1)} 万词典`
          : `${Number(d.dict_count || 0).toLocaleString()} 词典`;
      const hint = d.has_dict ? `${d.terms_count || 0} 条术语 · ${dictLabel}` : `${d.terms_count || 0} 条术语`;
      return `
        <div class="domain-item ${active}" data-domain="${escapeAttr(d.domain)}">
          <div class="domain-item-top">
            <span class="domain-item-name mono">${escapeHtml(d.domain)}</span>
            ${isDefault ? '<span class="status-label completed">默认</span>' : ''}
          </div>
          <div class="domain-item-meta">${escapeHtml(hint)}</div>
        </div>
      `;
    })
    .join('');
}

export function syncGlossaryHeaderControls(dom) {
  const titleEl = document.getElementById('glossary-active-title');
  if (titleEl) titleEl.textContent = dom || '未选择领域';

  ['btn-delete-domain', 'btn-add-domain-term', 'btn-save-domain-csv'].forEach(id => {
    const btn = document.getElementById(id);
    if (btn) btn.disabled = !dom;
  });
  ['btn-import-glossary', 'btn-import-dict'].forEach(id => {
    const lbl = document.getElementById(id);
    if (lbl) lbl.setAttribute('aria-disabled', String(!dom));
  });
  ['input-import-glossary', 'input-import-dict'].forEach(id => {
    const inp = document.getElementById(id);
    if (inp) inp.disabled = !dom;
  });

  const activeMeta = state.domainListMeta.find(d => d.domain === dom);
  const termsCountEl = document.getElementById('glossary-terms-count');
  if (termsCountEl) {
    const count = state.domainTermsMap[dom] ? state.domainTermsMap[dom].length : Number(activeMeta?.terms_count || 0);
    termsCountEl.textContent = String(count);
  }

  const dictPill = document.getElementById('dict-count-pill');
  if (dictPill) {
    if (activeMeta?.has_dict) {
      const cnt = Number(activeMeta.dict_count || 0);
      dictPill.className = 'status-label completed';
      dictPill.textContent = cnt >= 10000 ? `${(cnt / 10000).toFixed(1)} 万词` : `${cnt.toLocaleString()} 词`;
    } else {
      dictPill.className = 'status-label';
      dictPill.textContent = dom ? '未导入' : '';
    }
  }
}

export async function loadDomainDictEntries(dom, keyword = '') {
  if (dictFetchController) {
    dictFetchController.abort();
    dictFetchController = null;
  }
  if (state.isBackendConnected && dom) {
    const controller = new AbortController();
    dictFetchController = controller;
    try {
      const data = await apiFetch(
        `/api/glossary/domains/${encodeURIComponent(dom)}/dict?q=${encodeURIComponent(keyword)}`,
        { signal: controller.signal }
      );
      if (dictFetchController === controller) dictFetchController = null;
      if (Array.isArray(data.entries)) {
        state.domainDictEntries = data.entries;
        onRenderTablesHook?.({ dictOnly: true, keyword });
        return;
      }
    } catch (err) {
      if (err?.name === 'AbortError') return;
      appendLog('warn', 'DICT', `检索参考词典失败 (${dom}): ${err.message}`, true, undefined, {
        taskId: '',
        persist: false,
      });
    }
  }
  state.domainDictEntries = [];
  onRenderTablesHook?.({ dictOnly: true, keyword });
}

export async function selectGlossaryDomain(dom) {
  state.activeGlossaryDomain = dom;
  syncGlossaryHeaderControls(dom);

  if (state.isBackendConnected && dom) {
    try {
      const res = await apiFetch(`/api/glossary/domains/${encodeURIComponent(dom)}`);
      const terms = Array.isArray(res.terms) ? res.terms : [];
      state.domainTermsMap[dom] = terms.map(t => ({
        src: t.src || '',
        tgt: t.tgt || '',
        aliases: Array.isArray(t.aliases) ? t.aliases : [],
        note: t.note || '',
        fix_source: Boolean(t.fix_source),
        promote: true,
      }));
      state.hydratedGlossaryDomain = dom;
    } catch (err) {
      reportActionError('GLOSSARY', err);
    }
  } else {
    state.hydratedGlossaryDomain = dom;
  }

  const activeMeta = state.domainListMeta.find(d => d.domain === dom);
  const list = getActiveDomainTerms();
  if (activeMeta) activeMeta.terms_count = list.length;
  syncGlossaryHeaderControls(dom);
  renderDomainList();
  onRenderTablesHook?.({ csvOnly: true });
  const dictQuery = document.getElementById('dict-search-input')?.value || '';
  await loadDomainDictEntries(dom, dictQuery);
}

export async function ensureGlossaryDomainHydrated() {
  if (!state.activeGlossaryDomain) return;
  if (state.hydratedGlossaryDomain === state.activeGlossaryDomain) return;
  await selectGlossaryDomain(state.activeGlossaryDomain);
}

export async function loadBackendGlossaryDomains(options = {}) {
  const forceHydrate = Boolean(options.forceHydrate) || state.activeView === 'glossary';
  try {
    const data = await apiFetch('/api/glossary/domains');
    state.isBackendConnected = true;
    state.domainListMeta = Array.isArray(data.domains)
      ? data.domains.map(d => ({
          domain: d.domain,
          terms_count: d.terms_count ?? 0,
          has_dict: Boolean(d.has_dict),
          dict_count: d.dict_count ?? 0,
        }))
      : [];
    if (state.domainListMeta.length > 0) {
      const preferred =
        data.active_domain !== undefined
          ? state.domainListMeta.some(x => x.domain === data.active_domain)
            ? data.active_domain
            : ''
          : state.currentConfig.glossary_domain !== undefined &&
              state.domainListMeta.some(x => x.domain === state.currentConfig.glossary_domain)
            ? String(state.currentConfig.glossary_domain)
            : state.domainListMeta[0].domain;
      const taskDomInput = document.getElementById('task-domain');
      if (taskDomInput && !taskDomInput.dataset.userEdited) {
        taskDomInput.value =
          state.currentConfig.glossary_domain !== undefined
            ? state.domainListMeta.some(x => x.domain === state.currentConfig.glossary_domain)
              ? String(state.currentConfig.glossary_domain)
              : ''
            : preferred;
      }
      if (!state.domainListMeta.some(x => x.domain === state.activeGlossaryDomain)) {
        state.activeGlossaryDomain = preferred || state.domainListMeta[0].domain;
        state.hydratedGlossaryDomain = '';
      } else if (!forceHydrate) {
        state.hydratedGlossaryDomain = '';
      }
      if (forceHydrate) {
        await selectGlossaryDomain(state.activeGlossaryDomain);
      } else {
        syncGlossaryHeaderControls(state.activeGlossaryDomain);
        renderDomainList();
      }
    } else {
      state.activeGlossaryDomain = '';
      state.hydratedGlossaryDomain = '';
      syncGlossaryHeaderControls('');
      renderDomainList();
      onRenderTablesHook?.({ keyword: '' });
    }
  } catch (err) {
    appendLog('warn', 'GLOSSARY', `加载领域词表列表失败: ${err.message}`, true, undefined, {
      taskId: '',
      persist: false,
    });
  }
}

export async function persistActiveDomainTerms() {
  clearTimeout(glossaryAutosaveTimer);
  if (!state.activeGlossaryDomain) return;
  const validList = getActiveDomainTerms().filter(
    t => String(t.src || '').trim() || String(t.tgt || '').trim()
  );
  const meta = state.domainListMeta.find(d => d.domain === state.activeGlossaryDomain);
  const countChanged = Boolean(meta && meta.terms_count !== validList.length);
  if (meta) meta.terms_count = validList.length;
  const termsCountEl = document.getElementById('glossary-terms-count');
  if (termsCountEl) termsCountEl.textContent = String(validList.length);
  if (countChanged) renderDomainList();

  if (state.isBackendConnected) {
    await apiFetch(`/api/glossary/domains/${encodeURIComponent(state.activeGlossaryDomain)}`, {
      method: 'PUT',
      body: JSON.stringify({
        terms: validList.map(t => ({
          src: String(t.src || '').trim(),
          tgt: String(t.tgt || '').trim(),
          aliases: Array.isArray(t.aliases) ? t.aliases : [],
          note: String(t.note || ''),
          fix_source: Boolean(t.fix_source),
        })),
      }),
    });
  }
  TASK_DETAIL_CACHE.clear();
  setAutosaveStatus('completed', '词表已保存');
}

export function scheduleGlossaryAutosave() {
  if (!state.activeGlossaryDomain) return;
  setAutosaveStatus('paused', '保存中…');
  clearTimeout(glossaryAutosaveTimer);
  glossaryAutosaveTimer = setTimeout(async () => {
    try {
      await persistActiveDomainTerms();
    } catch (err) {
      reportActionError('GLOSSARY', err);
    }
  }, 450);
}

export async function handleGlossaryFileImport(inputEl, target) {
  const file = inputEl?.files?.[0];
  if (!file) return;
  try {
    if (!state.activeGlossaryDomain && state.isBackendConnected) {
      await loadBackendGlossaryDomains({ forceHydrate: true });
    }
    if (!state.activeGlossaryDomain) {
      setAutosaveStatus('error', '请先新建或选择领域');
      return;
    }
    if (state.isBackendConnected) {
      const form = new FormData();
      form.append('file', file);
      form.append('mode', 'merge');
      form.append('target', target);
      await apiFetch(`/api/glossary/domains/${encodeURIComponent(state.activeGlossaryDomain)}/import`, {
        method: 'POST',
        body: form,
      });
      TASK_DETAIL_CACHE.clear();
      setAutosaveStatus('completed', target === 'dict' ? '词典导入成功' : '术语导入成功');
      await loadBackendGlossaryDomains({ forceHydrate: true });
    }
  } catch (err) {
    reportActionError('IMPORT', err);
  } finally {
    inputEl.value = '';
  }
}
