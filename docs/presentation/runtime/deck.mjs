/* Stats-Lab overview deck: mounts the sunumatik deck shell and checks the slides against
 * deck-data.json (ADR-0097).
 *
 *   import { baslat } from './runtime/deck.mjs';
 *   baslat();
 *
 * - desteKur (deste-kabuk.mjs): next/previous, overview (O), notes (N), fullscreen (F), zoom (Z),
 *   print (P), help (?), #n deep links, ?export=1, reduced motion. English chrome labels for
 *   <html lang="en">.
 * - ?theme=light|dark overrides the system colour scheme (charts swap with it).
 * - denetle (deck-data.mjs): every <span data-k> must show the value deck-data.json gives;
 *   the result is logged and stored on <html data-veri="tutarli|N uyari">. The static HTML is
 *   already filled by tools/build_presentation_data.py, so the deck reads correctly without it. */
import { desteKur } from './deste-kabuk.mjs';
import { denetle } from './deck-data.mjs';

const EN = {
  'Deste kumandası': 'Deck controls', 'Önceki slayt (←)': 'Previous slide (←)', 'Sonraki slayt (→)': 'Next slide (→)',
  'Genel bakış': 'Overview', 'Genel bakış (O)': 'Overview (O)', 'Diğer ▾': 'More ▾', 'Diğer işlemler': 'More actions',
  'Sunucu notları  N': 'Speaker notes  N', 'Sunucu notlarını gizle  N': 'Hide speaker notes  N',
  'Tam ekran  F': 'Full screen  F', 'Yaklaştır  Z': 'Zoom  Z', 'Yaklaştırmayı kapat  Z': 'Stop zooming  Z',
  'PDF için yazdır  P': 'Print to PDF  P', 'Kısayollar  ?': 'Shortcuts  ?', 'Sunucu notları': 'Speaker notes',
  'Kısayollar': 'Shortcuts', 'İleri': 'Next', '→  ↓  PgDn  Boşluk': '→  ↓  PgDn  Space', 'Geri': 'Previous',
  'Baş / son': 'First / last', 'Tam ekran': 'Full screen', 'Yaklaştır (odak: tıkla · kat: + −)': 'Zoom (focus: click · scale: + −)',
  'PDF için yazdır': 'Print to PDF', 'Bu pencere': 'This window', 'Kapat': 'Close', 'Kapat (Esc)': 'Close (Esc)',
  'Kısayolları kapat (Esc)': 'Close shortcuts (Esc)', 'Bu slaytta not yok.': 'No notes on this slide.',
  'Tam ekran açılamadı': 'Full screen is not available', 'Genel bakışı kapat (Esc)': 'Close overview (Esc)',
  'Not · {n}. slayt': 'Notes · slide {n}', 'Genel bakış · {n} slayt': 'Overview · {n} slides', '{n}. slayt: {b}': 'Slide {n}: {b}',
  'Yaklaştırma {k} · tıkla: odak · +/−: kat · Esc: çık': 'Zoom {k} · click: focus · +/−: scale · Esc: exit',
};

/** Slides in deck order with their headline, bullets, visible text, and bound values. */
export function slaytlariTopla(sahne) {
  return [...sahne.querySelectorAll(':scope > section.slayt')].map(s => {
    const gorunur = s.cloneNode(true);
    gorunur.querySelectorAll('.notlar').forEach(n => n.remove());
    return {
      id: s.id,
      type: s.dataset.tur,
      headline: s.querySelector('h1, h2')?.textContent.replace(/\s+/g, ' ').trim() ?? '',
      bullets: [...s.querySelectorAll('li.madde')].map(li => li.textContent.replace(/\s+/g, ' ').trim()),
      text: gorunur.textContent.replace(/\s+/g, ' '),
      bound: [...s.querySelectorAll('[data-k]')].map(e => ({ key: e.dataset.k, fmt: e.dataset.fmt || 'str', text: e.textContent })),
    };
  });
}

export async function baslat() {
  const kok = document.documentElement;
  const tema = new URLSearchParams(location.search).get('theme');
  if (tema === 'light' || tema === 'dark') kok.dataset.theme = tema;
  const sahne = document.querySelector('#sahne');
  const lang = kok.lang === 'tr' ? 'tr' : 'en';
  const slaytlar = slaytlariTopla(sahne);
  const deste = desteKur(sahne, { baslik: document.title, gecis: 'none', metin: lang === 'en' ? EN : undefined });
  window.__deste = deste;
  try {
    const yanit = await fetch(new URL('../deck-data.json', import.meta.url));
    const uyarilar = denetle(slaytlar, await yanit.json(), lang);
    kok.dataset.veri = uyarilar.length ? `${uyarilar.length} uyari` : 'tutarli';
    if (uyarilar.length) console.warn('deck-data check:', uyarilar);
  } catch (hata) {
    kok.dataset.veri = 'okunamadi';
    console.info('deck-data.json could not be read; the slides show the values written at build time.', hata);
  }
  return deste;
}
