// =========================================================================
    // 真实仓库基准数据与默认任务演示 (来自 output/test_srt_benchmark 与 glossary/destiny2)
    // =========================================================================
    const REAL_TASKS = [
      {
        id: 'Destiny2_Attrition_Orbs_Guide',
        title: 'Destiny2_Attrition_Orbs_Guide.mp4',
        domain: 'destiny2',
        state: 'completed',
        stateLabel: '字幕就绪',
        stepSummary: '189 句 · 待合成',
        progress: 82,
      },
      {
        id: 'Qwen3_ASR_Architecture_Review',
        title: 'Qwen3_ASR_Architecture_Review.mp4',
        domain: 'ai_ml',
        state: 'paused',
        stateLabel: '待审术语',
        stepSummary: '24 条术语待确认',
        progress: 45,
      },
      {
        id: 'Apple_MLX_Optimization_Talk',
        title: 'Apple_MLX_Optimization_Talk.mov',
        domain: 'ai_ml',
        state: 'completed',
        stateLabel: '已完成',
        stepSummary: '成片已输出',
        progress: 100,
      }
    ];

    let activeTaskId = REAL_TASKS[0].id;
    let activeBatchId = 'all';

    const REAL_TASK_TERMS = [
      { src: 'Attrition Orbs', tgt: '羸弱能量球', aliases: ["attrition orb's", 'Attrition Orb', 'Trition Orbs', 'Trition Orbs proc'], note: '《命运2》武器perk，持续命中生成光球；官方译名为羸弱能量球', fix_source: true, promote: true },
      { src: 'Multimach CCX', tgt: '多机系统CCX', aliases: ['Multimach', 'Multi-Makh', 'Multimoc', 'Multi-Moc'], note: '《命运2》铁旗冲锋枪武器名，常作为对比基准', fix_source: true, promote: true },
      { src: 'D.F.A.', tgt: '死从天降', aliases: ['DFA'], note: '《命运2》日落手炮武器名', fix_source: false, promote: true },
      { src: 'Anonymous Autumn', tgt: '无名之秋', aliases: [], note: '《命运2》熔炉竞技场侧枪武器名', fix_source: false, promote: true },
      { src: 'Liminal Vigil', tgt: '阈限守夜', aliases: ['Liminal'], note: '《命运2》守望者尖塔地牢侧枪武器名', fix_source: false, promote: true },
      { src: 'Kinetic Tremors', tgt: '动能震颤', aliases: ['KT'], note: '《命运2》动能武器perk，与羸弱能量球有协同作用', fix_source: false, promote: true },
      { src: 'Explosive Payload', tgt: '高爆载荷', aliases: ['EP'], note: '《命运2》爆炸弹头perk，单发算作2次命中进度', fix_source: false, promote: true },
      { src: 'Indebted Kindness', tgt: '蒙恩', aliases: [], note: '《命运2》火箭辅助手枪武器名', fix_source: false, promote: true }
    ];

    // 左侧流水线阶段与批次树节点 (按 Step 01 → 04 从上到下顺序排列)
    const PIPELINE_BATCH_NODES = [
      { id: 'all', kind: 'summary', step: 'ALL', title: '全部字幕', meta: '189 句', time: '48.2s', desc: '全部字幕 (189 句)' },
      { id: 'asr-01', kind: 'step', step: '01', title: '语音识别', meta: '2,340 词', time: '14.2s', desc: '语音识别 · Qwen3-ASR 1.7B' },
      { id: 'nlp-all', kind: 'step', step: '02', title: '标点与断句', meta: '189 句', time: '11.8s', desc: '标点与断句 · 189 句' },
      { id: 'terms-03', kind: 'step', step: '03', title: '术语审校', meta: '8 条', time: '7.1s', desc: '术语审校' },
      { id: 'step-04-group', kind: 'group', step: '04', title: '字幕翻译', meta: '13 批', time: '15.1s', desc: '字幕翻译 · 13 批' },
      { id: 'batch-01', kind: 'child', step: 'B01', title: '#001–015', meta: '4 术语', time: '3.4s', desc: '第 001–015 句' },
      { id: 'batch-02', kind: 'child', step: 'B02', title: '#016–030', meta: '谐音修正', time: '3.1s', desc: '第 016–030 句 · 谐音修正' },
      { id: 'batch-05', kind: 'child', step: 'B05', title: '#061–075', meta: '2 术语', time: '3.0s', desc: '第 061–075 句' },
      { id: 'batch-08', kind: 'child', step: 'B08', title: '#106–120', meta: '2 术语', time: '2.9s', desc: '第 106–120 句' },
      { id: 'batch-13', kind: 'child', step: 'B13', title: '#181–189', meta: '3 术语', time: '2.4s', desc: '第 181–189 句' },
      { id: 'split-04', kind: 'child', step: '切行', title: '长句拆行', meta: '#071, #113', time: '1.8s', desc: '长句拆行 (#071, #113)' }
    ];

    // 右侧高密度表格数据行 (Detail: 支持百句级无缝滚动不崩坏)
    const STREAM_TABLE_ROWS = [
      {
        idx: '#001',
        stage: 'translate',
        batchId: 'batch-01',
        tc: '00:00:01 → 00:00:05',
        meta: 'B01 · 3 术语',
        srcHtml: `So, starting off, you might notice at the top here, we have <span class="term-token">DFA</span> with <span class="term-token">explosive payload</span> and <span class="term-token">attrition orbs</span>.`,
        tgtHtml: `首先，你可能注意到排在榜首的是带有<span class="term-token">高爆载荷</span>和<span class="term-token">羸弱能量球</span>的<span class="term-token">死从天降</span>。`
      },
      {
        idx: '#002',
        stage: 'translate',
        batchId: 'batch-01',
        tc: '00:00:05 → 00:00:11',
        meta: 'B01 · 2 术语',
        srcHtml: `Hey everyone, so while I was doing the primary tier list, we went through all the <span class="term-token">attrition orbs</span> classics, like AOKT, <span class="term-token">multimach</span>.`,
        tgtHtml: `大家好，我在做主武器分级列表时，我们回顾了所有<span class="term-token">羸弱能量球</span>的经典武器，比如 AOKT、<span class="term-token">多机系统CCX</span>。`
      },
      {
        idx: '#003',
        stage: 'translate',
        batchId: 'batch-01',
        tc: '00:00:11 → 00:00:16',
        meta: 'B01 · 2 术语',
        srcHtml: `And right next to it, <span class="term-token">Anonymous Autumn</span> and <span class="term-token">Liminal Vigil</span> also trigger orbs surprisingly fast in PvE.`,
        tgtHtml: `紧挨着它的<span class="term-token">无名之秋</span>和<span class="term-token">阈限守夜</span>在 PvE 中生成光球的速度也快得惊人。`
      },
      {
        idx: '#016',
        stage: 'translate',
        batchId: 'batch-02',
        tc: '00:01:04 → 00:01:09',
        meta: 'B02 · 谐音修正',
        srcHtml: `So, <span class="term-token">Indebted Kindness</span>, shout out to that. It rolls like Air Trigger and <span class="term-token" title="ASR 谐音变体自动映射 → Attrition Orbs">Trition Orbs</span>, which I think is pretty good.`,
        tgtHtml: `那么，<span class="term-token">蒙恩</span>，得提一下这把枪。它能刷出空中扳机和<span class="term-token">羸弱能量球</span>，我觉得挺不错的。`
      },
      {
        idx: '#017',
        stage: 'translate',
        batchId: 'batch-02',
        tc: '00:01:09 → 00:01:15',
        meta: 'B02 · 2 术语',
        srcHtml: `Because rocket sidearms deal area damage, <span class="term-token">Explosive Payload</span> counts as two hits per shot toward the orb progress.`,
        tgtHtml: `因为火箭辅助手枪造成范围伤害，<span class="term-token">高爆载荷</span>每一发都会算作两次命中进度。`
      },
      {
        idx: '#071',
        stage: 'split',
        batchId: 'split-04',
        tc: '00:04:18 → 00:04:24',
        meta: 'B05 · 拆为 2 轴',
        srcHtml: `Unfortunately, I wouldn't really recommend using a four-round magazine sniper with <span class="term-token">Explosive Payload</span> and <span class="term-token">Attrition Orbs</span> just to make an orb.`,
        tgtHtml: `不幸的是，我不太推荐为了生成光球而去用四发弹匣容量、<span class="br-marker" title="长句拆行位置">↵</span>带<span class="term-token">高爆载荷</span>和<span class="term-token">羸弱能量球</span>的天钿RR4。`
      },
      {
        idx: '#113',
        stage: 'split',
        batchId: 'split-04',
        tc: '00:06:51 → 00:06:57',
        meta: 'B08 · 拆为 2 轴',
        srcHtml: `Heavy grenade launchers do not follow the expected magazine percentage formula when triggering <span class="term-token">Attrition Orbs</span> on bosses.`,
        tgtHtml: `重型榴弹发射器不遵循武器触发<span class="term-token">羸弱能量球</span>时<span class="br-marker" title="长句拆行位置">↵</span>通常预期的弹匣容量百分比公式。`
      },
      {
        idx: '#188',
        stage: 'translate',
        batchId: 'batch-13',
        tc: '00:11:32 → 00:11:38',
        meta: 'B13 · 3 术语',
        srcHtml: `So if you are farming Iron Banner this week, definitely keep an eye out for <span class="term-token">Multimach CCX</span> with <span class="term-token">Attrition Orbs</span> and <span class="term-token">Kinetic Tremors</span>.`,
        tgtHtml: `所以如果你这周在刷铁旗，一定要留心带有<span class="term-token">羸弱能量球</span>和<span class="term-token">动能震颤</span>的<span class="term-token">多机系统CCX</span>。`
      },
      {
        idx: '#189',
        stage: 'translate',
        batchId: 'batch-13',
        tc: '00:11:38 → 00:11:43',
        meta: 'B13 · 收尾句',
        srcHtml: `Drop a comment down below if you found another weird interaction, and I'll catch you guys in the next guide.`,
        tgtHtml: `如果你发现了其他奇妙的机制联动，欢迎在下方评论区留言，我们下期攻略再见。`
      },
      {
        idx: 'NLP-01',
        stage: 'nlp',
        batchId: 'nlp-all',
        tc: 'Chunk 01/11 · 1.1s',
        meta: 'Step 02 · 标点恢复',
        srcHtml: `<span class="mono" style="color:var(--text-muted);">[RAW]</span> Hey everyone <span class="punct-chip">‖</span> so while I was doing the primary tier list <span class="punct-chip">‖</span> we went through all the attrition orbs classics`,
        tgtHtml: `<span class="mono" style="color:var(--emerald-bright);">[OUT]</span> Hey everyone<span class="punct-chip">,</span> so while I was doing the primary tier list<span class="punct-chip">,</span> we went through all the attrition orbs classics<span class="punct-chip">.</span>`
      },
      {
        idx: 'ASR-01',
        stage: 'asr',
        batchId: 'asr-01',
        tc: '2,340 词 · 14.2s',
        meta: 'Step 01 · 词级对齐',
        srcHtml: `本地 <code class="code-tag">Qwen3-ASR 1.7B</code> + <code class="code-tag">ForcedAligner-0.6B</code> 完成词级时间戳对齐，识别出 48 处自然换气停顿 (<span class="punct-chip">‖</span>)。`,
        tgtHtml: `已写入 <code class="code-tag">output/Destiny2_Attrition_Orbs_Guide/log/words.json</code>，供下游 LLM 断句与 SRT 精确时间轴映射。`
      }
    ];

    const REAL_SYSTEM_LOGS = [
      { time: '14:20:01', tag: 'asr', label: 'ASR', msg: 'Transcribing audio with local Qwen3-ASR 1.7b + ForcedAligner-0.6B (engine=mlx)...', warn: false },
      { time: '14:20:15', tag: 'asr', label: 'ASR', msg: 'Saved 2,340 word timestamps to output/Destiny2_Attrition_Orbs_Guide/log/words.json', warn: false },
      { time: '14:20:16', tag: 'nlp', label: 'NLP', msg: 'Splitting sentences by LLM-restored punctuation (language: en, 11 chunks, max_workers=4)...', warn: false },
      { time: '14:20:28', tag: 'nlp', label: 'NLP', msg: '189 sentences saved to log/split_by_nlp.txt and log/utterances.json', warn: false },
      { time: '14:20:29', tag: 'llm', label: 'TERMS', msg: 'Loaded domain glossary [destiny2]: 8 custom terms + 39 matched entries from dict.json (42,810 total)', warn: false },
      { time: '14:20:36', tag: 'llm', label: 'TERMS', msg: 'Extracted 63 task terms into log/terms.json (alias normalization applied: Trition Orbs -> Attrition Orbs)', warn: false },
      { time: '14:21:02', tag: 'llm', label: 'TRANS', msg: 'Translating 13 batches concurrently (max_workers=4, reflect_translate=false)...', warn: false },
      { time: '14:21:19', tag: 'warn', label: 'WARN', msg: 'Line #71 and #113 exceed max_length=75 (target_multiplier=1.3); invoking LLM batch split [br]...', warn: true },
      { time: '14:21:22', tag: 'llm', label: 'SRT', msg: 'Generated output/src.srt, output/trans.srt, output/bi.srt, and output/bi.ass (1920x1080)', warn: false },
      { time: '14:21:22', tag: 'warn', label: 'PAUSE', msg: 'Paused at checkpoint [pause_after_translate]: Edit trans.srt in Subtitle Edit, then click Burn Video or Dubbing.', warn: true }
    ];

    const REAL_DICT_ENTRIES = [
      { k: 'Attrition Orbs', v: '羸弱能量球' },
      { k: 'Attrition', v: '消耗 / 衰减' },
      { k: 'Orb of Power', v: '力量光球' },
      { k: 'Kinetic Tremors', v: '动能震颤' },
      { k: 'Explosive Payload', v: '高爆载荷' },
      { k: 'Multimach CCX', v: '多机系统CCX' },
      { k: 'Fatebringer', v: '命运带原者' },
      { k: 'Gjallarhorn', v: '加拉尔号角' }
    ];
