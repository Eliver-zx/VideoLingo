// @ts-check

/** @type {Array<Record<string, any>>} */
export const REAL_TASKS = [];

/** @type {Array<{src: string, tgt: string, aliases: string[], note: string, fix_source: boolean, promote: boolean, in_domain_kb?: boolean, source?: string, dirty?: boolean}>} */
export const REAL_TASK_TERMS = [];

/** @type {Array<{id: string, group: string, kind: string, step: string, title: string, meta: string, time: string, state?: string, desc: string}>} */
export const DEFAULT_PIPELINE_BATCH_NODES = [
  { id: 'all', group: 'overview', kind: 'summary', step: 'SRT', title: '成品字幕', meta: '0 轴', time: '', state: 'idle', desc: '成品字幕' },
  { id: 'runtime-log', group: 'overview', kind: 'summary', step: 'LOG', title: '运行日志', meta: '0 条', time: '', state: 'idle', desc: '运行日志' },
  { id: 'asr-01', group: 'pipeline', kind: 'step', step: '01', title: '语音识别', meta: '未开始', time: '', state: 'idle', desc: '语音识别' },
  { id: 'nlp-all', group: 'pipeline', kind: 'step', step: '02', title: '标点与断句', meta: '0 句', time: '', state: 'idle', desc: '标点与断句' },
  { id: 'terms-03', group: 'pipeline', kind: 'step', step: '03', title: '术语审校', meta: '0 条', time: '', state: 'idle', desc: '术语审校' },
  { id: 'step-04-group', group: 'pipeline', kind: 'step', step: '04', title: '字幕翻译', meta: '0 句', time: '', state: 'idle', desc: '字幕翻译' },
  { id: 'split-04', group: 'pipeline', kind: 'step', step: '05', title: '长句拆行', meta: '0 句', time: '', state: 'idle', desc: '长句拆行' },
];

export const PIPELINE_BATCH_NODES = DEFAULT_PIPELINE_BATCH_NODES.map(n => ({ ...n }));

/** @type {Array<Record<string, any>>} */
export const STREAM_TABLE_ROWS = [];

/** @type {Array<Record<string, any>>} */
export const ASR_SEGMENT_ROWS = [];

/** @type {Array<Record<string, any>>} */
export const SUBTITLE_CUE_ROWS = [];

/** @type {Array<{time: string, tag: string, label: string, msg: string, warn: boolean}>} */
export const REAL_SYSTEM_LOGS = [];

/** @type {Map<string, Array<{time: string, tag: string, label: string, msg: string, warn: boolean}>>} */
export const TASK_LOGS_BY_ID = new Map();

/** @type {Map<string, Record<string, any>>} */
export const TASK_DETAIL_CACHE = new Map();

export const burnRuntimeState = {
  state: 'idle',
  progress: 0,
  sub_step: '',
  video_path: '',
  subtitle_path: '',
  secondary_subtitle_path: '',
  output_path: '',
  completed_output_path: '',
  error: '',
};

/** @type {Array<{time: string, tag: string, label: string, msg: string, warn: boolean}>} */
export const burnLogs = [];

export const state = {
  activeTaskId: '',
  activeBatchId: 'all',
  activeView: 'workbench',
  isEditingTerms: false,
  showExistingKbTerms: false,
  isBackendConnected: false,
  activeGlossaryDomain: '',
  activeGlossaryTab: 'terms',
  hydratedGlossaryDomain: '',
  /** @type {Record<string, Array<any>>} */
  domainTermsMap: {},
  /** @type {Array<{k: string, v: string}>} */
  domainDictEntries: [],
  /** @type {Array<{domain: string, terms_count: number, has_dict: boolean, dict_count: number}>} */
  domainListMeta: [],
  /** @type {Record<string, any>} */
  currentConfig: {},
  burnGpuTouched: false,
  warnOnlyLog: false,
  logsDomDirty: false,
  termsDomDirty: false,
  loadedAsrTaskId: '',
  hydratedTaskId: '',
};
