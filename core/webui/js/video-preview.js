import { state, REAL_TASKS } from './state.js';
import { escapeHtml, escapeAttr, closeTopLayer, wireModalDialog } from './dom-utils.js';

function releasePreviewVideo(container) {
  const videoEl = container?.querySelector('video');
  if (videoEl) {
    videoEl.pause();
    videoEl.removeAttribute('src');
    videoEl.load();
  }
}

export function openVideoPreviewModal({
  title = '视频预览',
  sources = [],
  emptyTitle = '暂未生成成片视频',
  emptyDesc = '完成压制字幕或合成配音后，可在此直接预览成片效果',
}) {
  const previewDialog = document.getElementById('video-preview-dialog');
  if (!previewDialog) return;
  const titleEl = document.getElementById('preview-dialog-title');
  const tabsEl = document.getElementById('preview-source-tabs');
  const bodyEl = document.getElementById('video-preview-body');
  if (titleEl) titleEl.textContent = title;

  if (sources.length > 0 && bodyEl) {
    const renderVideo = url => {
      releasePreviewVideo(bodyEl);
      bodyEl.innerHTML = `
        <div style="border-radius:var(--radius-md); overflow:hidden; background:#000;">
          <video controls preload="metadata" style="inline-size:100%; display:block; aspect-ratio:16/9;" src="${escapeAttr(url)}"></video>
        </div>
      `;
    };
    renderVideo(sources[0].url);
    if (tabsEl) {
      if (sources.length > 1) {
        tabsEl.classList.remove('hidden');
        tabsEl.innerHTML = sources
          .map(
            (s, idx) => `
          <button type="button" class="seg-btn ${idx === 0 ? 'active' : ''}" role="tab" data-preview-idx="${idx}">${escapeHtml(s.label)}</button>
        `
          )
          .join('');
      } else {
        tabsEl.classList.add('hidden');
        tabsEl.innerHTML = '';
      }
      tabsEl._previewSources = sources;
      tabsEl._renderVideo = renderVideo;
    }
  } else if (bodyEl) {
    releasePreviewVideo(bodyEl);
    if (tabsEl) {
      tabsEl.classList.add('hidden');
      tabsEl.innerHTML = '';
      tabsEl._previewSources = null;
      tabsEl._renderVideo = null;
    }
    bodyEl.innerHTML = `
      <div class="preview-empty-state">
        <svg class="icon preview-empty-icon" viewBox="0 0 24 24" aria-hidden="true"><rect x="2" y="4" width="20" height="16" rx="2"/><polygon points="10 9 15 12 10 15 10 9"/></svg>
        <div class="preview-empty-title" id="preview-video-tag">${escapeHtml(emptyTitle)}</div>
        <div class="preview-empty-desc" id="preview-modal-status">${escapeHtml(emptyDesc)}</div>
      </div>
    `;
  }
  previewDialog.showModal();
}

export function initVideoPreviewEvents() {
  const previewDialog = document.getElementById('video-preview-dialog');
  wireModalDialog(previewDialog);

  previewDialog?.addEventListener('close', () => {
    releasePreviewVideo(previewDialog);
  });

  document.getElementById('preview-source-tabs')?.addEventListener('click', e => {
    const tabsEl = e.currentTarget;
    const btn = e.target.closest('[data-preview-idx]');
    if (!btn || !tabsEl._previewSources || !tabsEl._renderVideo) return;
    const idx = Number(btn.getAttribute('data-preview-idx'));
    tabsEl.querySelectorAll('[data-preview-idx]').forEach((b, i) => b.classList.toggle('active', i === idx));
    const chosen = tabsEl._previewSources[idx];
    if (chosen) tabsEl._renderVideo(chosen.url);
  });

  document.getElementById('btn-preview-video-modal')?.addEventListener('click', () => {
    const task = REAL_TASKS.find(x => x.id === state.activeTaskId);
    if (!task) {
      openVideoPreviewModal({
        title: '视频预览',
        sources: [],
        emptyTitle: '未选择任务',
        emptyDesc: '请先在左侧选择或新建一个任务',
      });
      return;
    }
    const sd = task.raw?.stages_done || {};
    const files = task.raw?.files || [];
    const sources = [];
    const base = `/api/tasks/${encodeURIComponent(task.id)}/files/`;
    if (sd.sub_video || files.includes('output_sub.mp4')) {
      sources.push({ label: '字幕成片', url: `${base}output_sub.mp4?inline=1&t=${Date.now()}` });
    }
    if (sd.dub_video || files.includes('output_dub.mp4')) {
      sources.push({ label: '配音成片', url: `${base}output_dub.mp4?inline=1&t=${Date.now()}` });
    }
    const rawMedia = task.raw?.media?.filename;
    if (rawMedia && files.includes(rawMedia) && /\.(mp4|mov|webm|mkv)$/i.test(rawMedia)) {
      sources.push({ label: '原始视频', url: `${base}${encodeURIComponent(rawMedia)}?inline=1` });
    }
    openVideoPreviewModal({
      title: task.title,
      sources,
      emptyTitle: '暂未生成可预览视频',
      emptyDesc: '完成「压制字幕」或「合成配音」后，可在此直接对比预览',
    });
  });

  document.getElementById('btn-close-preview-modal')?.addEventListener('click', () => {
    closeTopLayer(previewDialog);
  });
}
