import { syncBurnPreviewHint } from './burn-view.js';

let assPreviewRaf = 0;

export function normalizeAssColorToHex(raw, fallback = '#FFFF00') {
  const s = String(raw || '').trim();
  const hexMatch = s.match(/^#([0-9a-fA-F]{6})$/);
  if (hexMatch) return `#${hexMatch[1].toUpperCase()}`;
  const assMatch = s.match(/^&[Hh](?:[0-9a-fA-F]{2})?([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$/);
  if (assMatch) {
    const [, bb, gg, rr] = assMatch;
    return `#${rr}${gg}${bb}`.toUpperCase();
  }
  return fallback;
}

export function parseAssBackColor(raw) {
  const s = String(raw || '').trim();
  const assAlphaMatch = s.match(/^&[Hh]([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$/);
  if (assAlphaMatch) {
    const [, aa, bb, gg, rr] = assAlphaMatch;
    const opacity = Math.round((1 - parseInt(aa, 16) / 255) * 100);
    return { hex: `#${rr}${gg}${bb}`.toUpperCase(), opacity: Math.max(0, Math.min(100, opacity)) };
  }
  return { hex: normalizeAssColorToHex(s, '#000000'), opacity: 80 };
}

export function composeAssBackColor(hex, opacityPct) {
  const cleanHex = normalizeAssColorToHex(hex, '#000000').slice(1);
  const clamped = Math.max(0, Math.min(100, Number(opacityPct ?? 80)));
  const aa = Math.round((1 - clamped / 100) * 255)
    .toString(16)
    .toUpperCase()
    .padStart(2, '0');
  return `&H${aa}${cleanHex.slice(4, 6)}${cleanHex.slice(2, 4)}${cleanHex.slice(0, 2)}`;
}

function hexToRgba(hex, opacityFrac) {
  const h = normalizeAssColorToHex(hex, '#000000').slice(1);
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, opacityFrac)).toFixed(2)})`;
}

function buildCaptionShadow(outlineHex, outlineW, shadowRgba = '') {
  const parts = [];
  const w = Math.max(0, Number(outlineW || 0));
  if (w > 0) {
    const px = Math.max(0.8, Math.min(5, w * 1.05)).toFixed(1);
    const diag = (Number(px) * 0.72).toFixed(1);
    parts.push(
      `${px}px 0 0 ${outlineHex}`,
      `-${px}px 0 0 ${outlineHex}`,
      `0 ${px}px 0 ${outlineHex}`,
      `0 -${px}px 0 ${outlineHex}`,
      `${diag}px ${diag}px 0 ${outlineHex}`,
      `-${diag}px ${diag}px 0 ${outlineHex}`,
      `${diag}px -${diag}px 0 ${outlineHex}`,
      `-${diag}px -${diag}px 0 ${outlineHex}`
    );
  }
  if (shadowRgba) parts.push(`2px 2px 4px ${shadowRgba}`);
  return parts.length > 0 ? parts.join(', ') : 'none';
}

export function updateAssPreview() {
  const mode = document.getElementById('ass-burn-mode')?.value || 'translation';
  const trFont = (document.getElementById('ass-tr-font')?.value || '').trim();
  const srcFont = (document.getElementById('ass-src-font')?.value || '').trim();
  const trSize = Number(document.getElementById('ass-tr-size')?.value) || 17;
  const srcSize = Number(document.getElementById('ass-src-size')?.value) || 15;
  const trColor = normalizeAssColorToHex(document.getElementById('ass-tr-color')?.value, '#FFFF00');
  const srcColor = normalizeAssColorToHex(document.getElementById('ass-src-color')?.value, '#FFFFFF');
  const trOutlineColor = normalizeAssColorToHex(document.getElementById('ass-tr-outline-color')?.value, '#000000');
  const srcOutlineColor = normalizeAssColorToHex(document.getElementById('ass-src-outline-color')?.value, '#000000');
  const trOutlineW = Number(document.getElementById('ass-tr-outline-w')?.value ?? 1.5);
  const srcOutlineW = Number(document.getElementById('ass-src-outline-w')?.value ?? 1);
  const trMargin = Number(document.getElementById('ass-tr-margin')?.value ?? 5);

  const parsedBack = parseAssBackColor(document.getElementById('ass-tr-back-color')?.value || '&H33000000');
  const syncSwatch = (id, hex) => {
    const el = document.getElementById(id);
    if (el) el.value = hex.toLowerCase();
  };
  syncSwatch('ass-tr-color-picker', trColor);
  syncSwatch('ass-src-color-picker', srcColor);
  syncSwatch('ass-tr-outline-picker', trOutlineColor);
  syncSwatch('ass-src-outline-picker', srcOutlineColor);
  syncSwatch('ass-tr-back-picker', parsedBack.hex);
  const backOpacityEl = document.getElementById('ass-tr-back-opacity');
  if (backOpacityEl && document.activeElement !== backOpacityEl) {
    backOpacityEl.value = String(parsedBack.opacity);
  }

  document.querySelectorAll('#ass-style-table [data-style-col="tr"]').forEach(cell => {
    cell.classList.toggle('is-col-dimmed', mode === 'source');
  });
  document.querySelectorAll('#ass-style-table [data-style-col="src"]').forEach(cell => {
    cell.classList.toggle('is-col-dimmed', mode === 'translation');
  });

  const trShadowRgba = parsedBack.opacity > 0 ? hexToRgba(parsedBack.hex, parsedBack.opacity / 100) : '';
  const trTextShadow = buildCaptionShadow(trOutlineColor, trOutlineW, trShadowRgba);
  const srcTextShadow = buildCaptionShadow(srcOutlineColor, srcOutlineW, '');
  const trFamily = trFont ? `"${trFont.replace(/"/g, '')}", var(--font-sans)` : 'var(--font-sans)';
  const srcFamily = srcFont ? `"${srcFont.replace(/"/g, '')}", var(--font-sans)` : 'var(--font-sans)';
  const marginPx = `${Math.max(4, Math.round(trMargin * 0.75))}px`;

  const trEl = document.getElementById('ass-preview-tr');
  const srcEl = document.getElementById('ass-preview-src');
  const boxEl = document.getElementById('ass-preview-box');
  if (trEl) {
    trEl.style.fontFamily = trFamily;
    trEl.style.fontSize = `${Math.round(trSize * 1.18)}px`;
    trEl.style.fontWeight = '700';
    trEl.style.color = trColor;
    trEl.style.textShadow = trTextShadow;
    trEl.style.display = mode === 'source' ? 'none' : 'block';
  }
  if (srcEl) {
    srcEl.style.fontFamily = srcFamily;
    srcEl.style.fontSize = `${Math.round(srcSize * 1.06)}px`;
    srcEl.style.fontWeight = '500';
    srcEl.style.color = srcColor;
    srcEl.style.textShadow = srcTextShadow;
    srcEl.style.display = mode === 'translation' ? 'none' : 'block';
  }
  if (boxEl) boxEl.style.marginBlockEnd = marginPx;

  const burnTrEl = document.getElementById('burn-preview-tr');
  const burnSrcEl = document.getElementById('burn-preview-src');
  const burnBoxEl = document.getElementById('burn-preview-box');
  if (burnTrEl) {
    burnTrEl.style.fontFamily = trFamily;
    burnTrEl.style.fontSize = `${Math.round(trSize * 1.12)}px`;
    burnTrEl.style.fontWeight = '700';
    burnTrEl.style.color = trColor;
    burnTrEl.style.textShadow = trTextShadow;
  }
  if (burnSrcEl) {
    burnSrcEl.style.fontFamily = srcFamily;
    burnSrcEl.style.fontSize = `${Math.round(srcSize * 1.02)}px`;
    burnSrcEl.style.fontWeight = '500';
    burnSrcEl.style.color = srcColor;
    burnSrcEl.style.textShadow = srcTextShadow;
  }
  if (burnBoxEl) burnBoxEl.style.marginBlockEnd = marginPx;
  syncBurnPreviewHint();
}

export function scheduleAssPreview() {
  if (assPreviewRaf) return;
  assPreviewRaf = requestAnimationFrame(() => {
    assPreviewRaf = 0;
    updateAssPreview();
  });
}

export function initAssStyleEvents({ onBatchReset } = {}) {
  [
    'ass-burn-mode',
    'ass-tr-font',
    'ass-tr-size',
    'ass-tr-color',
    'ass-tr-outline-color',
    'ass-tr-outline-w',
    'ass-tr-margin',
    'ass-src-font',
    'ass-src-size',
    'ass-src-color',
    'ass-src-outline-color',
    'ass-src-outline-w',
  ].forEach(id => {
    const el = document.getElementById(id);
    const evtName = el?.tagName === 'SELECT' ? 'change' : 'input';
    el?.addEventListener(evtName, scheduleAssPreview);
  });

  const wireColorSwatchPair = (pickerId, hexInputId) => {
    const picker = document.getElementById(pickerId);
    const hexInp = document.getElementById(hexInputId);
    picker?.addEventListener('input', e => {
      if (!hexInp) return;
      hexInp.value = e.target.value.toUpperCase();
      hexInp.dispatchEvent(new Event('input', { bubbles: true }));
    });
  };
  wireColorSwatchPair('ass-tr-color-picker', 'ass-tr-color');
  wireColorSwatchPair('ass-src-color-picker', 'ass-src-color');
  wireColorSwatchPair('ass-tr-outline-picker', 'ass-tr-outline-color');
  wireColorSwatchPair('ass-src-outline-picker', 'ass-src-outline-color');

  const syncBackColorHidden = () => {
    const picker = document.getElementById('ass-tr-back-picker');
    const opEl = document.getElementById('ass-tr-back-opacity');
    const hiddenEl = document.getElementById('ass-tr-back-color');
    if (!hiddenEl) return;
    hiddenEl.value = composeAssBackColor(picker?.value || '#000000', opEl?.value ?? 80);
    hiddenEl.dispatchEvent(new Event('input', { bubbles: true }));
    scheduleAssPreview();
  };
  document.getElementById('ass-tr-back-picker')?.addEventListener('input', syncBackColorHidden);
  document.getElementById('ass-tr-back-opacity')?.addEventListener('input', syncBackColorHidden);

  document.querySelectorAll('[data-ass-bg]').forEach(btn => {
    btn.addEventListener('click', () => {
      const bg = btn.getAttribute('data-ass-bg') || 'dark';
      document.querySelectorAll('[data-ass-bg]').forEach(b => b.classList.toggle('active', b === btn));
      const stage = document.getElementById('ass-live-stage');
      if (stage) stage.dataset.backdrop = bg;
    });
  });

  document.getElementById('btn-reset-ass-style')?.addEventListener('click', () => {
    const resets = [
      ['ass-tr-font', '抖音美好体'],
      ['ass-tr-size', '17'],
      ['ass-tr-color', '#FFFF00'],
      ['ass-tr-outline-color', '#000000'],
      ['ass-tr-outline-w', '1.5'],
      ['ass-tr-margin', '5'],
      ['ass-src-font', ''],
      ['ass-src-size', '15'],
      ['ass-src-color', '#FFFFFF'],
      ['ass-src-outline-color', '#000000'],
      ['ass-src-outline-w', '1'],
      ['ass-tr-back-color', '&H33000000'],
    ];
    onBatchReset?.(resets);
    updateAssPreview();
  });
}
