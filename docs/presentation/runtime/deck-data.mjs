/* Deck data: number formats and the consistency check (ADR-0097).
 *
 * Adapted from sunumatik presets/deck_starter/deste-veri.mjs (commit d40d33c), whose idea is kept:
 * every number on the slides comes from ONE place, and denetle() says when a slide disagrees
 * with the data. Here the one place is deck-data.json (written by
 * tools/build_presentation_data.py from a fixed-seed run of the library); each number on a
 * slide is an element <span data-k="dotted.key" data-fmt="...">.
 *
 *   bicim(value, fmt, lang)          -> display text; mirrors format_value() in the Python tool
 *   bak(data, key)                   -> value at a dotted key (undefined when missing)
 *   denetle(slides, data, lang)      -> warning sentences (empty: the deck agrees with the data)
 *
 * `slides` is [{ id, type, headline, bullets, text, bound: [{ key, fmt, text }] }] in deck order;
 * the browser builds it from the DOM (deck.mjs), the Node check from the HTML text (denetle-cli.mjs).
 * Pure functions, no DOM. */

const FORMAT = /^(int|plus|str|f(\d)|pct(\d)|x(\d)|sig(\d)|musd(\d))$/;
const MINUS = '−';

function grup(digits, sep) {
  return digits.replace(/\B(?=(\d{3})+(?!\d))/g, sep);
}

/** Fixed decimals with the deck language's separators (Python: f"{v:,.{n}f}"). */
function sabit(v, n, lang) {
  const [tam, kesir] = Math.abs(v).toFixed(n).split('.');
  const [binlik, ondalik] = lang === 'tr' ? ['.', ','] : [',', '.'];
  return (v < 0 ? MINUS : '') + grup(tam, binlik) + (kesir !== undefined ? ondalik + kesir : '');
}

/** Display text of one value. Formats: int · plus · str · f<n> · pct<n> · x<n> · sig<n> · musd<n>. */
export function bicim(value, fmt = 'str', lang = 'en') {
  const m = FORMAT.exec(fmt);
  if (!m || !['en', 'tr'].includes(lang)) throw new Error(`unknown format ${fmt} or language ${lang}`);
  const sep = lang === 'en' ? ',' : '.';
  if (fmt === 'str') return String(value);
  if (fmt === 'int') return grup(String(Math.round(Number(value))), sep);
  if (fmt === 'plus') return grup(String(Math.trunc(Number(value))), sep) + '+';
  const n = Number(m.slice(2).find(g => g !== undefined));
  const v = Number(value);
  if (fmt.startsWith('f')) return sabit(v, n, lang);
  if (fmt.startsWith('pct')) { const t = sabit(100 * v, n, lang); return lang === 'tr' ? `%${t}` : `${t}%`; }
  if (fmt.startsWith('x')) return '×' + sabit(v, n, lang);
  if (fmt.startsWith('musd')) { const t = sabit(v, n, lang); return lang === 'tr' ? `${t} M$` : `$${t}M`; }
  const us = v ? Math.floor(Math.log10(Math.abs(v))) : 0;
  return sabit(v, Math.max(0, n - 1 - us), lang);
}

/** Value at a dotted key, or undefined. */
export function bak(data, key) {
  let v = data;
  for (const k of String(key).split('.')) {
    if (v === null || typeof v !== 'object' || !(k in v)) return undefined;
    v = v[k];
  }
  return v;
}

/* numbers in running text ("3.77%", "2,100+", "×233"): one number per token, either separator */
const SAYI = /(?<![\p{L}\p{N}])[−-]?\d+(?:[.,]\d+)*/gu;
const sayilar = (s, lang) => [...String(s ?? '').matchAll(SAYI)].map(x => {
  let t = x[0].replace(MINUS, '-');
  t = lang === 'tr' ? t.replace(/\./g, '').replace(',', '.') : t.replace(/,/g, '');
  return Number(t);
});

/** Consistency check of a rendered deck against its data -> warning sentences. */
export function denetle(slides, data, lang = 'en') {
  const u = [];
  const etiket = (s, i) => `${i + 1}. ${s.id || s.type}`;
  slides.forEach((s, i) => {
    for (const b of s.bound) {
      const v = bak(data, b.key);
      if (v === undefined) { u.push(`${etiket(s, i)}: key "${b.key}" is not in deck-data.json`); continue; }
      let beklenen;
      try { beklenen = bicim(v, b.fmt, lang); } catch (e) { u.push(`${etiket(s, i)}: ${e.message}`); continue; }
      if (beklenen !== b.text) u.push(`${etiket(s, i)}: "${b.key}" shows "${b.text}", the data give "${beklenen}"`);
    }
    if (s.headline && s.headline.length > 75) u.push(`${etiket(s, i)}: headline has ${s.headline.length} characters; one-line claims stay within 75`);
  });
  const kapak = slides.find(s => s.type === 'kapak');
  if (kapak && /\?\s*$/.test(kapak.headline)) u.push('cover: the headline is a claim, not a question');
  /* the conclusion adds no new information: each of its numbers appeared on an earlier slide */
  const i = slides.findIndex(s => s.type === 'sonuc');
  if (i >= 0) {
    const bilinen = new Set(slides.slice(0, i).flatMap(s => sayilar(s.text, lang)).map(x => Math.round(x * 100) / 100));
    for (const [k, m] of (slides[i].bullets || []).entries()) {
      for (const n of sayilar(m, lang)) {
        const r = Math.round(n * 100) / 100;
        if (![...bilinen].some(b => Math.abs(b - r) <= 0.005 + 1e-9)) u.push(`conclusion bullet ${k + 1}: ${n} does not appear on an earlier slide`);
      }
    }
    if ((slides[i].bullets || []).length !== 3) u.push(`conclusion: ${(slides[i].bullets || []).length} bullets; three are remembered`);
  }
  return u;
}
