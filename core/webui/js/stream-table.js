import {
  state,
  REAL_TASKS,
  REAL_TASK_TERMS,
  DEFAULT_PIPELINE_BATCH_NODES,
  PIPELINE_BATCH_NODES,
  STREAM_TABLE_ROWS,
  ASR_SEGMENT_ROWS,
  SUBTITLE_CUE_ROWS,
  REAL_SYSTEM_LOGS,
} from './state.js';
import { debounce, escapeHtml, escapeAttr, formatTimecode, formatWordTimestamp } from './dom-utils.js';

const CASE_TOKEN_RE = /^[A-Z][a-zA-Z0-9'-]*$/;
const PUNCT_TOKEN_SRC =
  '([A-Za-z0-9\'-]+|[\\u3400-\\u9FFF\\u3040-\\u30FF\\uAC00-\\uD7AF]|[^\\s,.;:?!，。；：？！—–…])([,.;:?!，。；：？！—–…]+)|([,.;:?!，。；：？！—–…]+)|([A-Za-z0-9\'-]+)';

let pendingStreamRaf = 0;
let pendingStreamJob = null;

export function cancelPendingStreamChunk() {
  if (pendingStreamRaf && typeof cancelAnimationFrame === 'function') {
    cancelAnimationFrame(pendingStreamRaf);
  }
  pendingStreamRaf = 0;
  pendingStreamJob = null;
}

function renderChunkedTableRows(tbody, items, renderItemHtml, baseChunkSize = 120, nextChunkSize = 200) {
  cancelPendingStreamChunk();
  const scrollEl = tbody.closest('.table-scroll');
  let firstChunkSize = baseChunkSize;
  if (scrollEl && scrollEl.scrollTop > 0) {
    const minForScroll = Math.ceil((scrollEl.scrollTop + (scrollEl.clientHeight || 800)) / 28) + 20;
    firstChunkSize = Math.max(baseChunkSize, minForScroll);
  }
  if (items.length <= firstChunkSize || typeof requestAnimationFrame !== 'function') {
    tbody.innerHTML = items.map(renderItemHtml).join('');
    return;
  }
  let initialHtml = '';
  for (let i = 0; i < firstChunkSize; i += 1) {
    initialHtml += renderItemHtml(items[i], i);
  }
  tbody.innerHTML = initialHtml;
  let cursor = firstChunkSize;
  const job = () => {
    if (pendingStreamJob !== job) return;
    const end = Math.min(cursor + nextChunkSize, items.length);
    let chunkHtml = '';
    for (let i = cursor; i < end; i += 1) {
      chunkHtml += renderItemHtml(items[i], i);
    }
    tbody.insertAdjacentHTML('beforeend', chunkHtml);
    cursor = end;
    if (cursor < items.length) {
      pendingStreamRaf = requestAnimationFrame(job);
    } else {
      pendingStreamRaf = 0;
      pendingStreamJob = null;
    }
  };
  pendingStreamJob = job;
  pendingStreamRaf = requestAnimationFrame(job);
}

export function formatPunctuatedHtml(punctText, rawText) {
  const text = String(punctText || '');
  if (!text) return '';
  const rawTokens = new Set(
    String(rawText || '')
      .split(/\s+/)
      .filter(Boolean)
  );
  const renderWord = word => {
    const safe = escapeHtml(word);
    if (
      CASE_TOKEN_RE.test(word) &&
      rawTokens.size > 0 &&
      !rawTokens.has(word) &&
      rawTokens.has(word.toLowerCase())
    ) {
      return `<span class="case-fix-token" title="大小写与专有名词恢复">${safe}</span>`;
    }
    return safe;
  };
  const pattern = new RegExp(PUNCT_TOKEN_SRC, 'g');
  let out = '';
  let lastIdx = 0;
  let m;
  while ((m = pattern.exec(text)) !== null) {
    if (m.index > lastIdx) {
      out += escapeHtml(text.slice(lastIdx, m.index));
    }
    if (m[1] !== undefined && m[2] !== undefined) {
      out += `<span class="punct-word">${renderWord(m[1])}<span class="punct-mark">${escapeHtml(m[2])}</span></span>`;
    } else if (m[3] !== undefined) {
      out += `<span class="punct-mark">${escapeHtml(m[3])}</span>`;
    } else if (m[4] !== undefined) {
      out += renderWord(m[4]);
    }
    lastIdx = pattern.lastIndex;
  }
  if (lastIdx < text.length) {
    out += escapeHtml(text.slice(lastIdx));
  }
  return out;
}

export function syncBatchNodesFromLoadedRows(sd = {}, totalWordsArg = 0) {
  if (STREAM_TABLE_ROWS.length > 0) {
    let countedWords = 0;
    let splitCount = 0;
    for (let i = 0; i < STREAM_TABLE_ROWS.length; i += 1) {
      countedWords += STREAM_TABLE_ROWS[i].wordCount || 0;
      if (STREAM_TABLE_ROWS[i].isSplit) splitCount += 1;
    }
    const totalWords = Number(totalWordsArg || 0) || countedWords;
    const asrMeta = sd.asr ? (totalWords > 0 ? `${totalWords} 词` : '已完成') : '未开始';
    const splitMeta = sd.subtitles
      ? splitCount > 0
        ? `${splitCount} 句拆分`
        : `${SUBTITLE_CUE_ROWS.length} 轴`
      : '0 句';
    const baseNodes = [
      { id: 'all', group: 'overview', kind: 'summary', step: 'SRT', title: '成品字幕', meta: `${SUBTITLE_CUE_ROWS.length} 轴`, time: '', state: 'idle', desc: '成品字幕' },
      { id: 'runtime-log', group: 'overview', kind: 'summary', step: 'LOG', title: '运行日志', meta: `${REAL_SYSTEM_LOGS.length} 条`, time: '', state: 'idle', desc: '运行日志' },
      { id: 'asr-01', group: 'pipeline', kind: 'step', step: '01', title: '语音识别', meta: asrMeta, time: '', state: 'idle', desc: '语音识别' },
      { id: 'nlp-all', group: 'pipeline', kind: 'step', step: '02', title: '标点与断句', meta: sd.split ? `${STREAM_TABLE_ROWS.length} 句` : '0 句', time: '', state: 'idle', desc: '标点与断句' },
      { id: 'terms-03', group: 'pipeline', kind: 'step', step: '03', title: '术语审校', meta: `${REAL_TASK_TERMS.length} 条`, time: '', state: 'idle', desc: '术语审校' },
      { id: 'step-04-group', group: 'pipeline', kind: 'step', step: '04', title: '字幕翻译', meta: sd.translate ? `${STREAM_TABLE_ROWS.length} 句` : '0 句', time: '', state: 'idle', desc: '字幕翻译' },
      { id: 'split-04', group: 'pipeline', kind: 'step', step: '05', title: '长句拆行', meta: splitMeta, time: '', state: 'idle', desc: '长句拆行' },
    ];
    PIPELINE_BATCH_NODES.splice(0, PIPELINE_BATCH_NODES.length, ...baseNodes);
    if (!PIPELINE_BATCH_NODES.some(n => n.id === state.activeBatchId)) {
      state.activeBatchId = 'all';
    }
    return;
  }

  const totalWordsOnly =
    Number(totalWordsArg || 0) || ASR_SEGMENT_ROWS.reduce((acc, r) => acc + r.wordCount, 0);
  const nodes = DEFAULT_PIPELINE_BATCH_NODES.map(n => {
    if (n.id === 'runtime-log') return { ...n, meta: `${REAL_SYSTEM_LOGS.length} 条` };
    if (n.id === 'asr-01' && (sd.asr || ASR_SEGMENT_ROWS.length > 0)) {
      return { ...n, meta: totalWordsOnly > 0 ? `${totalWordsOnly} 词` : '已完成' };
    }
    if (n.id === 'terms-03' && REAL_TASK_TERMS.length > 0) {
      return { ...n, meta: `${REAL_TASK_TERMS.length} 条` };
    }
    return { ...n };
  });
  PIPELINE_BATCH_NODES.splice(0, PIPELINE_BATCH_NODES.length, ...nodes);
}

export function mapAsrSegmentsToRows(asrSegments) {
  return (Array.isArray(asrSegments) ? asrSegments : []).map((seg, i) => {
    const numStr = `#${String(i + 1).padStart(3, '0')}`;
    const startSec = Number(seg.start || 0);
    const endSec = Number(seg.end || startSec);
    const durSec = Math.max(0, endSec - startSec);
    const gapAfter = Number(seg.gap_after || 0);
    const span =
      Array.isArray(seg.word_span) && seg.word_span.length >= 2
        ? [Number(seg.word_span[0] || 0), Number(seg.word_span[1] || 0)]
        : null;
    const wordsList = Array.isArray(seg.words) ? seg.words : [];
    const srcRawText = String(seg.src_raw || seg.src || '');
    const wordCount =
      wordsList.length > 0
        ? wordsList.length
        : span && span[1] > span[0]
          ? span[1] - span[0]
          : srcRawText
            ? srcRawText.trim().split(/\s+/).length
            : 0;
    return {
      idx: numStr,
      tc: `${formatTimecode(startSec)} → ${formatTimecode(endSec)}`,
      startSec,
      endSec,
      durSec,
      gapAfter,
      wordSpan: span,
      wordCount,
      srcRawText,
      words: wordsList,
      uttIndex: i,
    };
  });
}

export function rebuildStreamRowsFromUtterances(utts, sd = {}, asrSegments = [], totalWordsArg = 0) {
  if (asrSegments !== null && asrSegments !== undefined) {
    const mappedAsrRows = mapAsrSegmentsToRows(asrSegments);
    ASR_SEGMENT_ROWS.splice(0, ASR_SEGMENT_ROWS.length, ...mappedAsrRows);
  }

  if (utts.length > 0) {
    const cueRows = [];
    let totalWords = Number(totalWordsArg || 0);
    let countedWords = 0;
    const rows = utts.map((u, i) => {
      const numStr = `#${String(i + 1).padStart(3, '0')}`;
      const startSec = Number(u.start || 0);
      const endSec = Number(u.end || startSec);
      const durSec = Math.max(0, endSec - startSec);
      const gapAfter = Number(u.gap_after || 0);
      const tcStr = `${formatTimecode(startSec)} → ${formatTimecode(endSec)}`;
      const rawCues =
        Array.isArray(u.sub_cues) && u.sub_cues.length > 0
          ? u.sub_cues
          : [{ src: u.src, tgt: u.tgt, start: u.start, end: u.end }];
      const hasCues = rawCues.length > 1;
      const srcText = String(u.src ?? '');
      const srcRawText = String(u.src_raw || srcText);
      const srcPunctText = String(u.src_punct || srcText);
      const tgtText = String(u.tgt ?? '');
      const ttsText = String(u.tts_text ?? tgtText);
      const span =
        Array.isArray(u.word_span) && u.word_span.length >= 2
          ? [Number(u.word_span[0] || 0), Number(u.word_span[1] || 0)]
          : null;
      const wordCount =
        span && span[1] > span[0]
          ? span[1] - span[0]
          : srcRawText
            ? srcRawText.trim().split(/\s+/).length
            : 0;
      countedWords += wordCount;

      const cues = rawCues
        .map(c => {
          const cStart = Number(c.start ?? startSec);
          const cEnd = Number(c.end ?? endSec);
          return {
            src: String(c.src ?? '').trim(),
            tgt: String(c.tgt ?? c.trans ?? c.text ?? '').trim(),
            startSec: cStart,
            endSec: cEnd,
            durSec: Math.max(0, cEnd - cStart),
            tc: `${formatTimecode(cStart)} → ${formatTimecode(cEnd)}`,
          };
        })
        .filter(c => c.src || c.tgt);

      if (cues.length > 0) {
        cues.forEach(c => {
          const cueNum = `#${String(cueRows.length + 1).padStart(3, '0')}`;
          cueRows.push({
            idx: cueNum,
            uttIdx: numStr,
            uttIndex: i,
            tc: c.tc,
            startSec: c.startSec,
            endSec: c.endSec,
            durSec: c.durSec,
            srcText: c.src,
            tgtText: c.tgt,
          });
        });
      } else if (srcText || tgtText) {
        const cueNum = `#${String(cueRows.length + 1).padStart(3, '0')}`;
        cueRows.push({
          idx: cueNum,
          uttIdx: numStr,
          uttIndex: i,
          tc: tcStr,
          startSec,
          endSec,
          durSec,
          srcText,
          tgtText,
        });
      }

      return {
        idx: numStr,
        stage: hasCues ? 'split' : 'translate',
        isSplit: hasCues,
        tc: tcStr,
        startSec,
        endSec,
        durSec,
        gapAfter,
        wordSpan: span,
        wordCount,
        srcRawText,
        srcPunctText,
        srcText,
        tgtText,
        ttsText,
        cues,
        uttIndex: i,
        rawUtt: { ...u, src: srcText, tgt: tgtText, tts_text: ttsText },
      };
    });
    if (!totalWords) totalWords = countedWords;
    STREAM_TABLE_ROWS.splice(0, STREAM_TABLE_ROWS.length, ...rows);
    SUBTITLE_CUE_ROWS.splice(0, SUBTITLE_CUE_ROWS.length, ...cueRows);
    syncBatchNodesFromLoadedRows(sd, totalWords);
  } else {
    STREAM_TABLE_ROWS.splice(0, STREAM_TABLE_ROWS.length);
    SUBTITLE_CUE_ROWS.splice(0, SUBTITLE_CUE_ROWS.length);
    syncBatchNodesFromLoadedRows(sd, totalWordsArg);
  }
}

function syncStreamThead(thead, mode, col2Title, col3Title) {
  if (!thead || thead.dataset.mode === mode) return;
  thead.dataset.mode = mode;
  thead.innerHTML = `
    <tr>
      <th style="width:60px;">#</th>
      <th style="width:184px;">${col2Title}</th>
      <th>${col3Title}</th>
    </tr>
  `;
}

export function renderStreamTable() {
  const tbody = document.getElementById('structured-stream-tbody');
  const thead = document.getElementById('structured-stream-thead');
  if (!tbody) return;
  const searchEl = document.getElementById('subtitle-stream-search');
  const query = (searchEl?.value || '').trim().toLowerCase();
  const activeNode =
    PIPELINE_BATCH_NODES.find(n => n.id === state.activeBatchId) || PIPELINE_BATCH_NODES[0];

  const headingEl = document.getElementById('active-batch-heading');
  const tagEl = document.getElementById('active-batch-tag');
  if (headingEl && activeNode) headingEl.textContent = activeNode.title;
  if (tagEl && activeNode) {
    tagEl.textContent = activeNode.kind === 'step' ? `阶段 ${activeNode.step}` : '';
  }

  if (state.activeBatchId === 'all') {
    syncStreamThead(thead, 'all', '时间轴', '双语字幕（上译文 / 下原文）');
    const filteredCues = SUBTITLE_CUE_ROWS.filter(c => {
      if (!query) return true;
      return (
        c.idx.toLowerCase().includes(query) ||
        (c.tc || '').toLowerCase().includes(query) ||
        c.srcText.toLowerCase().includes(query) ||
        c.tgtText.toLowerCase().includes(query)
      );
    });
    document.getElementById('stream-row-counter').textContent = query
      ? `${filteredCues.length} / ${SUBTITLE_CUE_ROWS.length} 轴`
      : `共 ${SUBTITLE_CUE_ROWS.length} 轴`;

    if (filteredCues.length === 0) {
      cancelPendingStreamChunk();
      const emptyHint =
        REAL_TASKS.length === 0
          ? '暂无任务，点击左上角「新建」导入视频或字幕文件'
          : SUBTITLE_CUE_ROWS.length === 0
            ? '当前任务尚未生成成品字幕数据'
            : '未匹配到符合搜索条件的字幕轴';
      tbody.innerHTML = `
        <tr>
          <td colspan="3" style="padding:40px 16px; text-align:center; color:var(--text-muted); font-size:var(--fs-xs);">
            ${escapeHtml(emptyHint)}
          </td>
        </tr>
      `;
      return;
    }

    renderChunkedTableRows(
      tbody,
      filteredCues,
      c => `
        <tr>
          <td class="sub-idx">${escapeHtml(c.idx)}</td>
          <td class="sub-tc">${escapeHtml(c.tc || '00:00:00 → 00:00:00')}</td>
          <td>
            <div class="sub-bilingual-stack">
              <div class="sub-line-tgt">${escapeHtml(c.tgtText || '—')}</div>
              <div class="sub-line-src">${escapeHtml(c.srcText)}</div>
            </div>
          </td>
        </tr>
      `
    );
    return;
  }

  if (state.activeBatchId === 'split-04') {
    syncStreamThead(thead, 'split', '原句时间', '长句多轴拆分结果');
    const splitRows = STREAM_TABLE_ROWS.filter(r => {
      if (!r.isSplit) return false;
      if (!query) return true;
      return (
        r.idx.toLowerCase().includes(query) ||
        r.srcText.toLowerCase().includes(query) ||
        r.tgtText.toLowerCase().includes(query)
      );
    });
    document.getElementById('stream-row-counter').textContent = query
      ? `${splitRows.length} 句匹配`
      : `共 ${splitRows.length} 句拆分`;

    if (splitRows.length === 0) {
      cancelPendingStreamChunk();
      const emptyHint =
        STREAM_TABLE_ROWS.length === 0
          ? '当前阶段尚未生成字幕数据'
          : '当前任务无超长句拆分（全部单轴可读），或未匹配到搜索条件';
      tbody.innerHTML = `
        <tr>
          <td colspan="3" style="padding:40px 16px; text-align:center; color:var(--text-muted); font-size:var(--fs-xs);">
            ${escapeHtml(emptyHint)}
          </td>
        </tr>
      `;
      return;
    }

    renderChunkedTableRows(tbody, splitRows, r => {
      const cues =
        Array.isArray(r.cues) && r.cues.length > 0
          ? r.cues
          : [{ src: r.srcText, tgt: r.tgtText, tc: r.tc }];
      return `
        <tr data-utt-row="${r.uttIndex}">
          <td class="sub-idx">${escapeHtml(r.idx)}</td>
          <td class="sub-tc">${escapeHtml(r.tc || '00:00:00 → 00:00:00')}</td>
          <td>
            <div class="sub-cue-group">
              ${cues
                .map(
                  (c, cIdx) => `
                <div class="sub-cue-item">
                  <span class="sub-cue-num">${cIdx + 1}</span>
                  <span class="sub-cue-tc">${escapeHtml(c.tc || r.tc || '')}</span>
                  <div class="sub-bilingual-stack">
                    <div class="sub-line-tgt">${escapeHtml(c.tgt || '—')}</div>
                    <div class="sub-line-src">${escapeHtml(c.src)}</div>
                  </div>
                </div>
              `
                )
                .join('')}
            </div>
          </td>
        </tr>
      `;
    });
    return;
  }

  const isAsrStage = state.activeBatchId === 'asr-01';
  const isNlpStage = state.activeBatchId === 'nlp-all';
  if (isAsrStage) {
    syncStreamThead(thead, 'asr', '片段时间', '识别词流（悬停查看词级时间）');
  } else if (isNlpStage) {
    syncStreamThead(thead, 'nlp', '句子时间', '语义断句与标点恢复');
  } else {
    syncStreamThead(thead, 'translate', '句子时间', '翻译结果（上译文 / 下原文）');
  }

  const sourceRows = isAsrStage && ASR_SEGMENT_ROWS.length > 0 ? ASR_SEGMENT_ROWS : STREAM_TABLE_ROWS;

  const filtered = sourceRows.filter(r => {
    if (!query) return true;
    const srcPlain = String(r.srcText ?? '').toLowerCase();
    const rawPlain = String(r.srcRawText ?? '').toLowerCase();
    const punctPlain = String(r.srcPunctText ?? '').toLowerCase();
    const tgtPlain = String(r.tgtText ?? '').toLowerCase();
    return (
      r.idx.toLowerCase().includes(query) ||
      srcPlain.includes(query) ||
      rawPlain.includes(query) ||
      punctPlain.includes(query) ||
      (!isAsrStage && !isNlpStage && tgtPlain.includes(query))
    );
  });

  const totalWordsCount = sourceRows.reduce((acc, r) => acc + (r.wordCount || 0), 0);
  if (isAsrStage) {
    document.getElementById('stream-row-counter').textContent = query
      ? `${filtered.length} / ${sourceRows.length} 段`
      : `共 ${sourceRows.length} 段切片 · ${totalWordsCount} 词`;
  } else {
    document.getElementById('stream-row-counter').textContent = query
      ? `${filtered.length} / ${sourceRows.length} 句`
      : `共 ${filtered.length} 句`;
  }

  if (filtered.length === 0) {
    cancelPendingStreamChunk();
    const emptyHint =
      REAL_TASKS.length === 0
        ? '暂无任务，点击左上角「新建」导入视频或字幕文件'
        : sourceRows.length === 0
          ? '当前阶段尚未生成字幕数据'
          : '未匹配到符合搜索条件的记录';
    tbody.innerHTML = `
      <tr>
        <td colspan="3" style="padding:40px 16px; text-align:center; color:var(--text-muted); font-size:var(--fs-xs);">
          ${escapeHtml(emptyHint)}
        </td>
      </tr>
    `;
    return;
  }

  if (isAsrStage) {
    renderChunkedTableRows(
      tbody,
      filtered,
      r => {
        const uIdx = Number(r.uttIndex ?? 0);
        const rawVal = String(r.srcRawText || r.srcText || '').trim();
        const wordsList =
          Array.isArray(r.words) && r.words.length > 0
            ? r.words
            : rawVal
                .split(/\s+/)
                .filter(Boolean)
                .map(t => ({ text: t, start: null }));
        return `
          <tr data-utt-row="${uIdx}">
            <td class="sub-idx">${escapeHtml(r.idx)}</td>
            <td class="sub-tc">${escapeHtml(r.tc || '00:00:00 → 00:00:00')}</td>
            <td>
              <div class="asr-stream-prose">${wordsList
                .map(w => {
                  const hasTs = w.start !== null && w.start !== undefined;
                  return hasTs
                    ? `<span class="asr-word" data-ts="${escapeAttr(formatWordTimestamp(w.start))}">${escapeHtml(w.text)}</span>`
                    : `<span class="asr-word">${escapeHtml(w.text)}</span>`;
                })
                .join(' ')}</div>
            </td>
          </tr>
        `;
      },
      80,
      120
    );
    return;
  }

  if (isNlpStage) {
    renderChunkedTableRows(tbody, filtered, r => {
      const uIdx = Number(r.uttIndex ?? 0);
      const punctVal = String(r.srcPunctText || r.srcText || '').trim();
      const rawVal = String(r.srcRawText || '').trim();
      return `
        <tr data-utt-row="${uIdx}">
          <td class="sub-idx">${escapeHtml(r.idx)}</td>
          <td class="sub-tc">${escapeHtml(r.tc || '00:00:00 → 00:00:00')}</td>
          <td>
            <div class="sub-bilingual-stack">
              <div class="sub-line-tgt">${formatPunctuatedHtml(punctVal, rawVal)}</div>
            </div>
          </td>
        </tr>
      `;
    });
    return;
  }

  renderChunkedTableRows(tbody, filtered, r => {
    const uIdx = Number(r.uttIndex ?? 0);
    const srcVal = r.srcText !== undefined ? r.srcText : (r.rawUtt?.src ?? '');
    const tgtVal = r.tgtText !== undefined ? r.tgtText : (r.rawUtt?.tgt ?? '');
    return `
      <tr data-utt-row="${uIdx}">
        <td class="sub-idx">${escapeHtml(r.idx)}</td>
        <td class="sub-tc">${escapeHtml(r.tc || '00:00:00 → 00:00:00')}</td>
        <td>
          <div class="sub-bilingual-stack">
            <div class="sub-line-tgt">${escapeHtml(tgtVal || '—')}</div>
            <div class="sub-line-src">${escapeHtml(srcVal)}</div>
          </div>
        </td>
      </tr>
    `;
  });
}

export function initStreamTableSearch() {
  document
    .getElementById('subtitle-stream-search')
    ?.addEventListener('input', debounce(renderStreamTable, 120));
}
