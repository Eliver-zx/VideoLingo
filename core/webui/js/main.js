import { state, REAL_TASKS } from './state.js';
import { renderSystemLogs, initConsoleLogEvents } from './console-log.js';
import { syncDetailPane, initRouterEvents } from './router.js';
import { renderStreamTable, initStreamTableSearch } from './stream-table.js';
import { renderTaskTerms, initTermsReviewEvents } from './terms-review.js';
import {
  renderTaskRail,
  renderBatchTree,
  syncStartResumeButton,
  loadBackendTasks,
  initTaskRailEvents,
} from './task-rail.js';
import { initIntakeModalEvents } from './intake-modal.js';
import { initVideoPreviewEvents } from './video-preview.js';
import {
  renderDomainList,
  renderDomainCsvTable,
  renderDictSearch,
  loadBackendGlossaryDomains,
  initGlossaryEvents,
} from './glossary-view.js';
import {
  switchSettingsSection,
  updateAssPreview,
  loadBackendConfig,
  initSettingsViewEvents,
} from './settings-view.js';
import {
  renderBurnControls,
  renderBurnLogs,
  loadBackendBurnStatus,
  initBurnViewEvents,
} from './burn-view.js';
import { connectEventStream } from './sse.js';

initRouterEvents();
initConsoleLogEvents({
  onToggleConsole: () => {
    state.activeBatchId = state.activeBatchId === 'runtime-log' ? 'all' : 'runtime-log';
    renderBatchTree();
    syncDetailPane();
  },
});
initStreamTableSearch();
initTermsReviewEvents({ reloadTasks: loadBackendTasks });
initTaskRailEvents();
initIntakeModalEvents();
initVideoPreviewEvents();
initGlossaryEvents();
initSettingsViewEvents();
initBurnViewEvents({ onJumpAssStyle: switchSettingsSection });

syncStartResumeButton(null);
renderTaskRail();
renderBatchTree();
renderStreamTable();
renderTaskTerms();
renderSystemLogs();
renderDomainList();
renderDomainCsvTable();
renderDictSearch('', true);
renderBurnControls();
renderBurnLogs();
updateAssPreview();

(async () => {
  const cfgPromise = loadBackendConfig().then(() => {
    if (REAL_TASKS.length > 0) {
      let changed = false;
      for (let i = 0; i < REAL_TASKS.length; i += 1) {
        const raw = REAL_TASKS[i].raw;
        if (raw && raw.config_override?.glossary_domain === undefined) {
          const nextDom = String(state.currentConfig.glossary_domain ?? 'destiny2');
          if (REAL_TASKS[i].domain !== nextDom) {
            REAL_TASKS[i].domain = nextDom;
            changed = true;
          }
        }
      }
      if (changed) {
        renderTaskRail();
        const active = REAL_TASKS.find(x => x.id === state.activeTaskId);
        const domPill = document.getElementById('detail-domain-pill');
        if (active && domPill) domPill.textContent = active.domain || '通用';
      }
    }
  });
  await Promise.all([
    cfgPromise,
    loadBackendGlossaryDomains(),
    loadBackendBurnStatus(),
    loadBackendTasks(),
  ]);
  connectEventStream();
})();
