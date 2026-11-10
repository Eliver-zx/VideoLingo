// @ts-check
// =========================================================================
    // 0. 运行时状态容器 (由后端 API 与 SSE 实时同步，无演示假数据)
    // =========================================================================
    /** @type {Array<Record<string, any>>} */
    const REAL_TASKS = [];

    let activeTaskId = '';
    let activeBatchId = 'all';
    let isEditingTerms = false;
    let showExistingKbTerms = false;

    /** @type {Array<{src: string, tgt: string, aliases: string[], note: string, fix_source: boolean, promote: boolean, in_domain_kb?: boolean, source?: string, dirty?: boolean}>} */
    const REAL_TASK_TERMS = [];

    /** @type {Array<{id: string, kind: string, step: string, title: string, meta: string, time: string, desc: string}>} */
    const DEFAULT_PIPELINE_BATCH_NODES = [
      { id: 'runtime-log', kind: 'summary', step: 'LOG', title: '运行日志', meta: '0 条', time: '', desc: '运行日志 (runtime.log)' },
      { id: 'all', kind: 'summary', step: 'SRT', title: '成品字幕', meta: '0 轴', time: '', desc: '成品字幕 (0 轴)' },
      { id: 'asr-01', kind: 'step', step: '01', title: '语音识别', meta: '未开始', time: '', desc: '语音识别' },
      { id: 'nlp-all', kind: 'step', step: '02', title: '标点与断句', meta: '0 句', time: '', desc: '标点与断句' },
      { id: 'terms-03', kind: 'step', step: '03', title: '术语审校', meta: '0 条', time: '', desc: '术语审校' },
      { id: 'step-04-group', kind: 'step', step: '04', title: '字幕翻译', meta: '0 句', time: '', desc: '字幕翻译' },
      { id: 'split-04', kind: 'step', step: '05', title: '长句拆行', meta: '0 句', time: '', desc: '长句拆行' }
    ];

    const PIPELINE_BATCH_NODES = DEFAULT_PIPELINE_BATCH_NODES.map(n => ({ ...n }));

    /** @type {Array<Record<string, any>>} */
    const STREAM_TABLE_ROWS = [];

    /** @type {Array<Record<string, any>>} */
    const ASR_SEGMENT_ROWS = [];

    /** @type {Array<Record<string, any>>} */
    const SUBTITLE_CUE_ROWS = [];

    /** @type {Array<{time: string, tag: string, label: string, msg: string, warn: boolean}>} */
    const REAL_SYSTEM_LOGS = [];

    /** @type {Array<{k: string, v: string}>} */
    const REAL_DICT_ENTRIES = [];
