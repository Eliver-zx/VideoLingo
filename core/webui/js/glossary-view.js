import { state, TASK_DETAIL_CACHE } from './state.js';
import { apiFetch, debounce, escapeHtml, escapeAttr, setAutosaveStatus } from './dom-utils.js';
import { reportActionError } from './console-log.js';
import {
  registerGlossaryRenderHook,
  getActiveDomainTerms,
  syncDomainSelectOptions,
  renderDomainList,
  selectGlossaryDomain,
  ensureGlossaryDomainHydrated,
  loadBackendGlossaryDomains,
  loadDomainDictEntries,
  persistActiveDomainTerms,
  scheduleGlossaryAutosave,
  handleGlossaryFileImport,
} from './glossary-data.js';

export {
  syncDomainSelectOptions,
  renderDomainList,
  selectGlossaryDomain,
  ensureGlossaryDomainHydrated,
  loadBackendGlossaryDomains,
  loadDomainDictEntries,
};

let renderedDictSlice = [];

registerGlossaryRenderHook(({ dictOnly = false, csvOnly = false, keyword = '' } = {}) => {
  if (!dictOnly) renderDomainCsvTable();
  if (!csvOnly) renderDictSearch(keyword, true);
});

function switchGlossaryTab(tab) {
  state.activeGlossaryTab = tab === 'dict' ? 'dict' : 'terms';
  document.querySelectorAll('[data-glossary-tab]').forEach(btn => {
    const on = btn.getAttribute('data-glossary-tab') === state.activeGlossaryTab;
    btn.classList.toggle('active', on);
    btn.setAttribute('aria-selected', String(on));
  });
  document.getElementById('glossary-pane-terms')?.classList.toggle('hidden', state.activeGlossaryTab !== 'terms');
  document.getElementById('glossary-terms-toolbar')?.classList.toggle('hidden', state.activeGlossaryTab !== 'terms');
  document.getElementById('glossary-pane-dict')?.classList.toggle('hidden', state.activeGlossaryTab !== 'dict');
  document.getElementById('glossary-dict-toolbar')?.classList.toggle('hidden', state.activeGlossaryTab !== 'dict');
}

export function renderDomainCsvTable() {
  const tbody = document.getElementById('domain-csv-tbody');
  if (!tbody) return;
  const q = (document.getElementById('glossary-csv-search')?.value || '').trim().toLowerCase();
  const list = getActiveDomainTerms();
  const termsCountEl = document.getElementById('glossary-terms-count');
  if (termsCountEl) termsCountEl.textContent = String(list.length);

  const rows = list
    .map((t, idx) => ({ t, idx }))
    .filter(({ t }) => {
      if (!q) return true;
      return (
        t.src.toLowerCase().includes(q) ||
        t.tgt.toLowerCase().includes(q) ||
        (t.note || '').toLowerCase().includes(q) ||
        t.aliases.some(a => a.toLowerCase().includes(q))
      );
    });

  if (rows.length === 0) {
    const emptyHint = !state.activeGlossaryDomain
      ? '请先在左侧选择或新建一个领域词表'
      : list.length === 0
        ? '当前领域暂无自定义术语，点击右上角「新增术语」或「导入」添加'
        : '未匹配到符合搜索条件的自定义术语';
    tbody.innerHTML = `
      <tr>
        <td colspan="5" style="padding:36px 16px; text-align:center; color:var(--text-muted); font-size:var(--fs-xs);">
          ${escapeHtml(emptyHint)}
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = rows
    .map(
      ({ t, idx }) => `
      <tr>
        <td><input type="text" class="cell-input mono" value="${escapeAttr(t.src)}" placeholder="原文术语" data-csv-field="src" data-idx="${idx}"></td>
        <td><input type="text" class="cell-input is-tgt" value="${escapeAttr(t.tgt)}" placeholder="目标译名" data-csv-field="tgt" data-idx="${idx}"></td>
        <td>
          <div class="alias-inline-bar">
            ${t.aliases
              .map(
                (al, aIdx) => `
              <span class="alias-pill">
                <span>${escapeHtml(al)}</span>
                <button type="button" class="alias-del" data-del-csv-alias="${idx}:${aIdx}" title="移除别名">×</button>
              </span>
            `
              )
              .join('')}
            <input type="text" class="alias-add-input mono" placeholder="+ 别名" data-add-csv-alias="${idx}">
          </div>
        </td>
        <td><input type="text" class="cell-input" value="${escapeAttr(t.note)}" placeholder="选填备注" data-csv-field="note" data-idx="${idx}"></td>
        <td>
          <div class="term-actions">
            <button type="button" class="term-icon-btn is-del" data-del-csv="${idx}" title="移除该条术语" aria-label="移除术语">
              <svg class="icon" viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
            <button type="button" class="term-icon-btn" data-insert-csv="${idx}" title="在下方插入一行" aria-label="在下方插入一行">
              <svg class="icon" viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            </button>
          </div>
        </td>
      </tr>
    `
    )
    .join('');
}

export function renderDictSearch(keyword = '', prefiltered = false) {
  const tbody = document.getElementById('dict-search-tbody');
  if (!tbody) return;
  const q = keyword.trim().toLowerCase();
  const sourceList = state.domainDictEntries;
  const matched = prefiltered
    ? sourceList
    : sourceList.filter(x => !q || x.k.toLowerCase().includes(q) || x.v.toLowerCase().includes(q));
  renderedDictSlice = matched;

  if (matched.length === 0) {
    const activeMeta = state.domainListMeta.find(d => d.domain === state.activeGlossaryDomain);
    const emptyHint = !state.activeGlossaryDomain
      ? '请先在左侧选择一个领域词表'
      : !activeMeta?.has_dict
        ? '当前领域尚未挂载参考词典，可点击右上角「导入词典」上传 JSON 或双语词表'
        : '未匹配到符合检索条件的词条';
    tbody.innerHTML = `
      <tr>
        <td colspan="3" style="padding:36px 16px; text-align:center; color:var(--text-muted); font-size:var(--fs-xs);">
          ${escapeHtml(emptyHint)}
        </td>
      </tr>
    `;
    return;
  }

  const existingSet = new Set(
    getActiveDomainTerms()
      .map(t => String(t.src || '').trim().toLowerCase())
      .filter(Boolean)
  );

  tbody.innerHTML = matched
    .map((item, idx) => {
      const inCustom = existingSet.has(String(item.k || '').trim().toLowerCase());
      return `
        <tr>
          <td class="mono" style="font-weight:500;">${escapeHtml(item.k)}</td>
          <td style="font-weight:600; color:var(--text-primary);">${escapeHtml(item.v)}</td>
          <td style="text-align:right;">
            ${
              inCustom
                ? '<button type="button" class="xs ghost" disabled>已收录</button>'
                : `<button type="button" class="xs" data-promote-dict-idx="${idx}">+ 加入术语</button>`
            }
          </td>
        </tr>
      `;
    })
    .join('');
}

export function initGlossaryEvents() {
  document.querySelectorAll('[data-glossary-tab]').forEach(btn => {
    btn.addEventListener('click', () => {
      switchGlossaryTab(btn.getAttribute('data-glossary-tab') || 'terms');
    });
  });

  document.getElementById('domain-list-box')?.addEventListener('click', e => {
    const el = e.target.closest('[data-domain]');
    if (!el) return;
    selectGlossaryDomain(el.getAttribute('data-domain') || '');
  });

  const csvTbody = document.getElementById('domain-csv-tbody');
  csvTbody?.addEventListener('input', e => {
    const inp = e.target.closest('[data-csv-field]');
    if (!inp) return;
    const list = getActiveDomainTerms();
    const rIdx = Number(inp.getAttribute('data-idx'));
    const field = inp.getAttribute('data-csv-field');
    if (!list[rIdx] || !field) return;
    list[rIdx][field] = inp.value;
    scheduleGlossaryAutosave();
  });

  csvTbody?.addEventListener('keydown', e => {
    const inp = e.target.closest('[data-add-csv-alias]');
    if (!inp || e.key !== 'Enter' || !inp.value.trim()) return;
    e.preventDefault();
    const list = getActiveDomainTerms();
    const rIdx = Number(inp.getAttribute('data-add-csv-alias'));
    if (!list[rIdx]) return;
    list[rIdx].aliases.push(inp.value.trim());
    renderDomainCsvTable();
    scheduleGlossaryAutosave();
  });

  csvTbody?.addEventListener('click', e => {
    const list = getActiveDomainTerms();
    const delAliasBtn = e.target.closest('[data-del-csv-alias]');
    if (delAliasBtn) {
      const [rIdx, aIdx] = delAliasBtn.getAttribute('data-del-csv-alias').split(':').map(Number);
      if (!list[rIdx]) return;
      list[rIdx].aliases.splice(aIdx, 1);
      renderDomainCsvTable();
      scheduleGlossaryAutosave();
      return;
    }
    const delBtn = e.target.closest('[data-del-csv]');
    if (delBtn) {
      list.splice(Number(delBtn.getAttribute('data-del-csv')), 1);
      renderDomainCsvTable();
      scheduleGlossaryAutosave();
      return;
    }
    const insBtn = e.target.closest('[data-insert-csv]');
    if (insBtn) {
      const idx = Number(insBtn.getAttribute('data-insert-csv'));
      const insertAt = idx >= 0 && idx < list.length ? idx + 1 : list.length;
      list.splice(insertAt, 0, { src: '', tgt: '', aliases: [], note: '', fix_source: false, promote: true });
      renderDomainCsvTable();
      document.querySelector(`#domain-csv-tbody input[data-csv-field="src"][data-idx="${insertAt}"]`)?.focus();
    }
  });

  document.getElementById('dict-search-tbody')?.addEventListener('click', e => {
    const btn = e.target.closest('[data-promote-dict-idx]');
    if (!btn || !state.activeGlossaryDomain) return;
    const entry = renderedDictSlice[Number(btn.getAttribute('data-promote-dict-idx'))];
    if (!entry) return;
    const list = getActiveDomainTerms();
    if (!list.some(t => String(t.src || '').trim().toLowerCase() === entry.k.trim().toLowerCase())) {
      list.unshift({ src: entry.k, tgt: entry.v, aliases: [], note: '', fix_source: false, promote: true });
      renderDomainCsvTable();
      renderDictSearch(document.getElementById('dict-search-input')?.value || '', true);
      scheduleGlossaryAutosave();
    }
  });

  document.getElementById('glossary-csv-search')?.addEventListener('input', debounce(renderDomainCsvTable, 120));
  document.getElementById('dict-search-input')?.addEventListener(
    'input',
    debounce(e => {
      loadDomainDictEntries(state.activeGlossaryDomain, e.target.value);
    }, 180)
  );

  document.getElementById('btn-add-domain-term')?.addEventListener('click', () => {
    if (!state.activeGlossaryDomain) return;
    switchGlossaryTab('terms');
    getActiveDomainTerms().unshift({ src: '', tgt: '', aliases: [], note: '', fix_source: false, promote: true });
    renderDomainCsvTable();
    document.querySelector('#domain-csv-tbody input[data-csv-field="src"]')?.focus();
  });

  document.getElementById('btn-save-domain-csv')?.addEventListener('click', async () => {
    if (!state.activeGlossaryDomain) return;
    try {
      await persistActiveDomainTerms();
    } catch (err) {
      reportActionError('GLOSSARY', err);
    }
  });

  document.getElementById('input-import-glossary')?.addEventListener('change', e => {
    handleGlossaryFileImport(e.currentTarget, 'terms');
  });
  document.getElementById('input-import-dict')?.addEventListener('change', e => {
    handleGlossaryFileImport(e.currentTarget, 'dict');
  });

  const newDomainCard = document.getElementById('new-domain-card');
  const newDomainInput = document.getElementById('new-domain-input');
  document.getElementById('btn-toggle-new-domain')?.addEventListener('click', () => {
    if (!newDomainCard) return;
    const willOpen = newDomainCard.classList.contains('hidden');
    newDomainCard.classList.toggle('hidden', !willOpen);
    if (willOpen) newDomainInput?.focus();
  });
  document.getElementById('btn-cancel-domain')?.addEventListener('click', () => {
    if (newDomainInput) newDomainInput.value = '';
    newDomainCard?.classList.add('hidden');
  });
  newDomainInput?.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      document.getElementById('btn-create-domain')?.click();
    } else if (e.key === 'Escape') {
      document.getElementById('btn-cancel-domain')?.click();
    }
  });

  document.getElementById('btn-create-domain')?.addEventListener('click', async () => {
    const slug = (newDomainInput?.value || '').trim().toLowerCase().replace(/[^a-z0-9_-]/g, '_');
    if (!slug) {
      newDomainInput?.focus();
      return;
    }
    if (state.isBackendConnected) {
      try {
        await apiFetch('/api/glossary/domains', { method: 'POST', body: JSON.stringify({ domain: slug }) });
        if (newDomainInput) newDomainInput.value = '';
        newDomainCard?.classList.add('hidden');
        state.activeGlossaryDomain = slug;
        setAutosaveStatus('completed', `已创建 ${slug}`);
        await loadBackendGlossaryDomains({ forceHydrate: true });
        return;
      } catch (err) {
        reportActionError('GLOSSARY', err);
        return;
      }
    }
    if (!state.domainListMeta.some(d => d.domain === slug)) {
      state.domainListMeta.push({ domain: slug, terms_count: 0, has_dict: false, dict_count: 0 });
      state.domainTermsMap[slug] = [];
    }
    if (newDomainInput) newDomainInput.value = '';
    newDomainCard?.classList.add('hidden');
    await selectGlossaryDomain(slug);
  });

  document.getElementById('btn-delete-domain')?.addEventListener('click', async () => {
    const deletedDom = state.activeGlossaryDomain;
    if (!deletedDom) return;
    if (!window.confirm(`确定要永久删除领域词表「${deletedDom}」吗？该操作不可撤销。`)) return;
    delete state.domainTermsMap[deletedDom];
    if (state.hydratedGlossaryDomain === deletedDom) state.hydratedGlossaryDomain = '';
    if (state.isBackendConnected) {
      try {
        await apiFetch(`/api/glossary/domains/${encodeURIComponent(deletedDom)}`, { method: 'DELETE' });
        TASK_DETAIL_CACHE.clear();
        setAutosaveStatus('completed', '领域已删除');
        await loadBackendGlossaryDomains({ forceHydrate: true });
        return;
      } catch (err) {
        reportActionError('GLOSSARY', err);
        return;
      }
    }
    state.domainListMeta = state.domainListMeta.filter(d => d.domain !== deletedDom);
    await selectGlossaryDomain(state.domainListMeta[0]?.domain || '');
  });
}
