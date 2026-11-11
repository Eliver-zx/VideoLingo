import { state, REAL_TASKS } from './state.js';
import { autoResizeThemeTextarea } from './dom-utils.js';
import { flushLogsDomIfNeeded } from './console-log.js';
import { ensureGlossaryDomainHydrated } from './glossary-view.js';
import { flushTermsDomIfNeeded } from './terms-review.js';

function getViewElements() {
  return {
    workbench: document.getElementById('view-workbench'),
    burn: document.getElementById('view-burn'),
    glossary: document.getElementById('view-glossary'),
    settings: document.getElementById('view-settings'),
  };
}

export function switchView(target) {
  const views = getViewElements();
  if (!views[target]) target = 'workbench';
  state.activeView = target;
  document.querySelectorAll('[data-nav]').forEach(b => {
    b.classList.toggle('active', b.getAttribute('data-nav') === target);
  });
  Object.entries(views).forEach(([k, el]) => {
    if (el) el.classList.toggle('hidden', k !== target);
  });
  if (target === 'glossary') {
    ensureGlossaryDomainHydrated();
  }
}

export function syncPrimaryCtaFocus() {
  const mainStartBtn = document.getElementById('btn-start-resume-task');
  if (!mainStartBtn) return;
  const task = REAL_TASKS.find(x => x.id === state.activeTaskId);
  const termsOwnsPrimary = Boolean(
    state.activeBatchId === 'terms-03' &&
      task &&
      task.rawState === 'paused' &&
      task.stateLabel === '待审术语'
  );
  const subsOwnsPrimary = Boolean(
    state.activeBatchId !== 'terms-03' &&
      state.activeBatchId !== 'runtime-log' &&
      task &&
      task.rawState === 'paused' &&
      task.stateLabel !== '待审术语'
  );
  mainStartBtn.classList.toggle('primary', !termsOwnsPrimary && !subsOwnsPrimary);
}

export function syncDetailPane() {
  const paneSubtitle = document.getElementById('subtitle-detail-pane');
  const paneTerms = document.getElementById('terms-review-panel');
  const consoleDock = document.getElementById('console-dock');
  const isTerms = state.activeBatchId === 'terms-03';
  const isLog = state.activeBatchId === 'runtime-log';
  paneSubtitle?.classList.toggle('hidden', isTerms || isLog);
  paneTerms?.classList.toggle('hidden', !isTerms);
  consoleDock?.classList.toggle('hidden', !isLog);
  const syncSrtBtn = document.getElementById('btn-sync-subtitles-from-srt');
  if (syncSrtBtn) {
    syncSrtBtn.classList.toggle('hidden', state.activeBatchId !== 'all');
  }
  syncPrimaryCtaFocus();
  if (isTerms) {
    flushTermsDomIfNeeded();
    autoResizeThemeTextarea(document.getElementById('terms-theme-input'));
  }
  if (isLog && consoleDock) {
    flushLogsDomIfNeeded();
  }
}

export function initRouterEvents() {
  document.querySelectorAll('[data-nav]').forEach(btn => {
    btn.addEventListener('click', () => {
      switchView(btn.getAttribute('data-nav') || 'workbench');
    });
  });
}
