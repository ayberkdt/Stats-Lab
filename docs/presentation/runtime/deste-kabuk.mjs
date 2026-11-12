/* Copied from sunumatik presets/core/deste-kabuk.mjs (commit d40d33c), the deck runtime of the
 * project owner's presentation toolkit. Local change: every chrome label goes through T(), so a deck
 * can pass `metin: { 'Türkçe etiket': 'label' }` (docs/presentation/runtime/deck.mjs does for English).
 * Everything else is unchanged. */
/* DESTE KABUĞU — sunumatik desteleri için ortak çalışma zamanı (sunum düzeni).
 *
 * Neden: her demo deste (ornek-deste, tatli-deste, deck_starter) sahne ölçekleme,
 * tuş, ilerleme çubuğu, genel bakış ve not panelini kendi içinde yeniden yazıyordu;
 * aynı kusurlar (sarmal gezinti, ölçülmeyen küçük resim, düşen tuş) tekrarlıyordu.
 * Bu modül sabit sahne sözleşmesini (fixed-stage-contract.md) BİR kez uygular:
 * 1920×1080 sahne, transform-origin 0 0, slaytlar görünürlük/opaklık/işaretçi ile
 * değişir, kumanda sahnenin DIŞINDA ekran boyutunda kalır.
 *
 *   import { desteKur } from '../presets/core/deste-kabuk.mjs';
 *   const deste = desteKur(document.querySelector('#sahne'), {
 *     slaytSecici: '.slayt',          // sahnenin doğrudan çocukları
 *     gecis: 'fade-through',          // 'fade-through' | 'push' | 'none'
 *     gecisFn: transitionSlides,      // motion_core/slide-transitions.js; verilirse sınıf değişimini o yapar
 *     onEnter(slayt, i) {}, onLeave(slayt, i) {},
 *     baslik: 'Deste adı',            // yardım penceresinde görünür
 *     kumanda: true,                  // denetim çubuğu
 *   });
 *   deste → { git, ileri, geri, index, sayi, tamEkran, genelBakis, notlar, yaklas, yaklasSifirla, yardim, yokEt }
 *
 * Slayt: sahnenin doğrudan çocuğu; içinde `aside.notlar` (sunucu notu, sahnede gizli),
 *        `data-enter` (geçiş türü), `data-anchor` (zoom-into çapası) olabilir.
 * Tuşlar (KISAYOLLAR): → ↓ PgDn Boşluk ileri · ← ↑ PgUp geri · Home/End uçlar ·
 *        F tam ekran · O genel bakış · N notlar · Z yaklaştır · P yazdır · ? kısayollar · Esc kapat.
 *        Alt/Ctrl/Meta basılıyken, odak form alanı/dialog içindeyken ve
 *        [data-owns-arrows] / [data-owns-keys] alt ağacındayken ok ve Boşluk destede değil.
 * Adres: #n (1 tabanlı) derin bağlantı; her harekette replaceState; hashchange dinlenir.
 *        ?export=1 → html[data-export="true"]: krom yok, geçiş yok, yaklaştırma yok.
 * Yaklaştırma ("sayfa büyütme", Z): sahne s·kat ile odak noktası ekran ortasında kalacak
 *        biçimde büyür (yaklasKirp → yaklasDonusum); tıkla odak, +/− kat, sürükle kaydır,
 *        Ctrl+tekerlek, oklar 120 sahne px kaydırır.
 * Saf yardımcılar (olcek, adrestenIndeks, yaklasKirp, yaklasDonusum) DOM'a dokunmaz;
 * modülün içe alınmasının yan etkisi yoktur — Node'da sınanır (scripts/validate-deste-kabuk.mjs).
 * Kabuk hiçbir zaman kendiliğinden ilerlemez: her hareket kullanıcı girdisidir. */

export const SURUM = 1;

export const KISAYOLLAR = {
  ileri: ['ArrowRight', 'ArrowDown', 'PageDown', ' '],
  geri: ['ArrowLeft', 'ArrowUp', 'PageUp'],
  bas: ['Home'],
  son: ['End'],
  tamEkran: ['f', 'F'],
  genelBakis: ['o', 'O'],
  notlar: ['n', 'N'],
  yaklas: ['z', 'Z'],
  yazdir: ['p', 'P'],
  yardim: ['?'],
  kapat: ['Escape'],
};

const SAHNE_G = 1920, SAHNE_Y = 1080;

/** Sabit sahne ölçeği: s = min(g/1920, y/1080); x, y kutulama (letterbox) kaydırması. */
export function olcek(genislik, yukseklik) {
  const s = Math.min(genislik / SAHNE_G, yukseklik / SAHNE_Y);
  return { s, x: (genislik - SAHNE_G * s) / 2, y: (yukseklik - SAHNE_Y * s) / 2 };
}

/** '#3' → 2 (0 tabanlı); boş/bozuk → 0; aralık dışı → sınıra çekilir; n < 1 → 0. */
export function adrestenIndeks(hash, n) {
  const m = String(hash ?? '').match(/^#?(\d+)$/);
  if (!m || !(n >= 1)) return 0;
  return Math.min(n - 1, Math.max(0, Number(m[1]) - 1));
}

/** Yaklaştırma durumunu kırpar (saf). kat ∈ [min, max]; odak (x, y) sahne koordinatında ve
 *  büyütülmüş sahne hiç boş tuval göstermeyecek biçimde sınırlanır:
 *  x ∈ [960/kat, 1920 − 960/kat], y ∈ [540/kat, 1080 − 540/kat]. kat ≤ 1 → kimlik. */
export function yaklasKirp({ kat, x, y } = {}, { min = 1, max = 4 } = {}) {
  let k = Number(kat);
  if (!Number.isFinite(k)) k = 1;
  k = Math.min(max, Math.max(min, k));
  if (k <= 1) return { kat: 1, x: SAHNE_G / 2, y: SAHNE_Y / 2 };
  const yx = SAHNE_G / 2 / k, yy = SAHNE_Y / 2 / k;
  const ox = Number.isFinite(Number(x)) ? Number(x) : SAHNE_G / 2;
  const oy = Number.isFinite(Number(y)) ? Number(y) : SAHNE_Y / 2;
  return {
    kat: k,
    x: Math.min(SAHNE_G - yx, Math.max(yx, ox)),
    y: Math.min(SAHNE_Y - yy, Math.max(yy, oy)),
  };
}

const donusumDizgesi = (s, x, y) => `translate(${x}px, ${y}px) scale(${s})`;

/** Taban dönüşümü {s, x, y} (olcek) ve yaklaştırma durumu {kat, x, y} → CSS transform.
 *
 *  Türetme: taban dönüşümü sahne noktası p'yi ekrana  E(p) = T + s·p  ile götürür; sahne
 *  merkezi (960, 540) kutulama gereği görüntü alanının merkezine düşer: M = T + s·(960, 540).
 *  Yaklaştırmada ölçek s·kat olur ve odak noktası (x, y) merkezde kalmalıdır:
 *      T' + s·kat·(x, y) = M  ⇒  T' = T + s·(960 − kat·x, 540 − kat·y).
 *  kat = 1, odak (960, 540) ⇒ T' = T: tabanla birebir aynı dizge. */
export function yaklasDonusum({ kat = 1, x = SAHNE_G / 2, y = SAHNE_Y / 2 } = {}, taban) {
  const { s, x: tx, y: ty } = taban;
  const sk = s * kat;
  return donusumDizgesi(sk, tx + s * (SAHNE_G / 2 - kat * x), ty + s * (SAHNE_Y / 2 - kat * y));
}

/* ── çalışma zamanı ─────────────────────────────────────────────────────── */

const el = (etiket, ozellik = {}, metin) => {
  const e = Object.assign(document.createElement(etiket), ozellik);
  if (metin !== undefined) e.textContent = metin;
  return e;
};
const katYaz = k => `×${(Math.round(k * 10) / 10).toString().replace('.', ',')}`;

export function desteKur(sahne, secenek = {}) {
  const ayar = {
    slaytSecici: '.slayt', gecis: 'fade-through', gecisFn: null, onEnter: null, onLeave: null,
    baslik: '', kumanda: true, ...secenek,
  };
  /* yerel değişiklik: krom metinleri ayar.metin ile çevrilir (yoksa Türkçe) */
  const T = (s) => (ayar.metin && Object.hasOwn(ayar.metin, s) ? ayar.metin[s] : s);
  const kok = document.documentElement;
  const ihracat = new URLSearchParams(location.search).get('export') === '1';
  if (ihracat) kok.dataset.export = 'true';
  const azHareket = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

  sahne.classList.add('dk-sahne');
  const slaytlar = [...sahne.children].filter(e => e.matches(ayar.slaytSecici));
  const n = slaytlar.length;
  let simdi = -1, gecisSuruyor = false, bekleyen = -1;

  /* dinleyiciler ve üretilen DOM, yokEt için kayıt altında */
  const temizlik = [];
  const dinle = (hedef, tur, fn, opt) => {
    hedef.addEventListener(tur, fn, opt);
    temizlik.push(() => hedef.removeEventListener(tur, fn, opt));
  };
  const uretilen = [];
  const ekle = (e) => { document.body.appendChild(e); uretilen.push(e); return e; };

  /* ── ölçek + yaklaştırma durumu ── */
  let taban = olcek(innerWidth, innerHeight);
  let yakin = { kat: 1, x: SAHNE_G / 2, y: SAHNE_Y / 2 };
  let yaklasKipi = false;
  const uygula = () => { sahne.style.transform = yaklasDonusum(yakin, taban); };
  const olcekle = () => { taban = olcek(innerWidth, innerHeight); uygula(); };
  /* ekran noktası → sahne noktası (geçerli dönüşümden) */
  const sahneNoktasi = (cx, cy) => {
    const sk = taban.s * yakin.kat;
    const tx = taban.x + taban.s * (SAHNE_G / 2 - yakin.kat * yakin.x);
    const ty = taban.y + taban.s * (SAHNE_Y / 2 - yakin.kat * yakin.y);
    return { x: (cx - tx) / sk, y: (cy - ty) / sk };
  };

  /* ── krom: kumanda çubuğu, menü, ilerleme, notlar, genel bakış, yardım, çip, bildirim ── */
  const kromVar = !ihracat;
  const dugme = (metin, etiket, fn, sinif = '') => {
    const b = el('button', { type: 'button', title: etiket, className: sinif }, metin);
    b.setAttribute('aria-label', etiket);
    b.addEventListener('click', fn);
    return b;
  };

  let kumanda = null, geriB = null, ileriB = null, bakisB = null, digerB = null, sayac = null, menu = null, notMenu = null, yaklasMenu = null;
  if (kromVar && ayar.kumanda) {
    kumanda = el('nav', { className: 'dk-kumanda' });
    kumanda.setAttribute('aria-label', T('Deste kumandası'));
    geriB = dugme('‹', T('Önceki slayt (←)'), () => geri());
    sayac = el('span', { className: 'dk-sayac' });
    sayac.setAttribute('aria-live', 'polite');
    ileriB = dugme('›', T('Sonraki slayt (→)'), () => ileri(), 'birincil');
    bakisB = dugme(T('Genel bakış'), T('Genel bakış (O)'), () => genelBakis());
    bakisB.setAttribute('aria-pressed', 'false');
    digerB = dugme(T('Diğer ▾'), T('Diğer işlemler'), () => menuAc());
    digerB.setAttribute('aria-haspopup', 'menu');
    digerB.setAttribute('aria-expanded', 'false');
    kumanda.append(geriB, sayac, ileriB, bakisB, digerB);
    ekle(kumanda);

    menu = el('div', { className: 'dk-menu', hidden: true });
    menu.setAttribute('role', 'menu');
    menu.setAttribute('aria-label', T('Diğer işlemler'));
    const menuOge = (metin, fn) => {
      const b = el('button', { type: 'button' }, metin);
      b.setAttribute('role', 'menuitem');
      b.addEventListener('click', () => { menuKapat(); fn(); });
      menu.append(b);
      return b;
    };
    notMenu = menuOge(T('Sunucu notları  N'), () => notlar());
    menuOge(T('Tam ekran  F'), () => tamEkran());
    yaklasMenu = menuOge(T('Yaklaştır  Z'), () => yaklas());
    menuOge(T('PDF için yazdır  P'), () => yazdir());
    menuOge(T('Kısayollar  ?'), () => yardim());
    ekle(menu);
    /* menü içinde ok tuşlarıyla gezinti (role=menu sözleşmesi) */
    menu.addEventListener('keydown', e => {
      const ogeler = [...menu.querySelectorAll('[role="menuitem"]')];
      const i = ogeler.indexOf(document.activeElement);
      if (e.key === 'ArrowDown') { e.preventDefault(); ogeler[(i + 1) % ogeler.length]?.focus(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); ogeler[(i - 1 + ogeler.length) % ogeler.length]?.focus(); }
    });
    dinle(document, 'pointerdown', e => { if (menu && !menu.hidden && !menu.contains(e.target) && e.target !== digerB) menuKapat(); });
  }
  function menuAc() {
    if (!menu) return;
    if (!menu.hidden) return menuKapat();
    menu.hidden = false;
    digerB.setAttribute('aria-expanded', 'true');
    menu.querySelector('button')?.focus();
  }
  function menuKapat() {
    if (!menu) return;
    menu.hidden = true;
    digerB.setAttribute('aria-expanded', 'false');
  }

  /* ilerleme çubuğu: geçiş kapalıyken de slayt değişiminin tek görsel sinyali */
  const ilerleme = kromVar ? ekle(el('div', { className: 'dk-ilerleme' })) : null;
  ilerleme?.setAttribute('aria-hidden', 'true');

  const notPaneli = kromVar ? ekle(el('aside', { className: 'dk-notlar', hidden: true })) : null;
  notPaneli?.setAttribute('aria-label', T('Sunucu notları'));

  const bildirim = kromVar ? ekle(el('p', { className: 'dk-bildirim', hidden: true })) : null;
  bildirim?.setAttribute('role', 'status');
  let bildirimZaman = 0;
  function bildir(m) {
    if (!bildirim) return;
    bildirim.textContent = m;
    bildirim.hidden = false;
    clearTimeout(bildirimZaman);
    bildirimZaman = setTimeout(() => { bildirim.hidden = true; }, 3200);
  }

  const bakis = kromVar ? ekle(el('div', { className: 'dk-bakis', hidden: true })) : null;
  if (bakis) {
    bakis.setAttribute('role', 'dialog');
    bakis.setAttribute('aria-modal', 'true');
    bakis.setAttribute('aria-label', T('Genel bakış'));
  }

  const cip = kromVar ? ekle(el('div', { className: 'dk-yaklas-cip', hidden: true })) : null;
  cip?.setAttribute('role', 'status');

  /* yardım: tek tablo, ≤ 11 satır */
  let yardimDlg = null;
  if (kromVar) {
    yardimDlg = el('dialog', { className: 'dk-yardim' });
    yardimDlg.setAttribute('aria-labelledby', 'dk-yardim-baslik');
    const bas = el('h2', { id: 'dk-yardim-baslik' }, T('Kısayollar'));
    const ad = el('p', { className: 'dk-yardim-deste' }, ayar.baslik || document.title || '');
    const tablo = el('table');
    const SATIRLAR = [
      [T('İleri'), T('→  ↓  PgDn  Boşluk')], [T('Geri'), '←  ↑  PgUp'], [T('Baş / son'), 'Home / End'],
      [T('Tam ekran'), 'F'], [T('Genel bakış'), 'O'], [T('Sunucu notları'), 'N'],
      [T('Yaklaştır (odak: tıkla · kat: + −)'), 'Z'], [T('PDF için yazdır'), 'P'],
      [T('Bu pencere'), '?'], [T('Kapat'), 'Esc'],
    ];
    for (const [islev, tus] of SATIRLAR) {
      const tr = el('tr');
      tr.append(el('td', {}, islev), el('td', {}, tus));
      tablo.append(tr);
    }
    const kapatB = dugme(T('Kapat (Esc)'), T('Kısayolları kapat (Esc)'), () => yardimDlg.close());
    yardimDlg.append(bas, ad, tablo, kapatB);
    ekle(yardimDlg);
  }

  /* ── slayt değişimi ── */
  function dinlenmeDurumu() {
    slaytlar.forEach((s, k) => {
      const etkin = k === simdi;
      s.classList.toggle('etkin', etkin);
      s.setAttribute('aria-hidden', String(!etkin));
      s.inert = !etkin;
    });
  }
  function kromGuncelle() {
    if (ilerleme) ilerleme.style.width = `${((simdi + 1) / n) * 100}%`;
    if (sayac) sayac.textContent = `${simdi + 1} / ${n}`;
    if (geriB) geriB.disabled = simdi === 0;
    if (ileriB) ileriB.disabled = simdi === n - 1;
    if (notPaneli) {
      const not = slaytlar[simdi].querySelector('aside.notlar, .notlar');
      notPaneli.replaceChildren(el('b', {}, T('Not · {n}. slayt').replace('{n}', String(simdi + 1))), el('p', {}, not ? not.textContent.trim() : T('Bu slaytta not yok.')));
    }
  }
  const adresYaz = () => history.replaceState(null, '', `${location.pathname}${location.search}#${simdi + 1}`);

  async function git(i, { yaz = true, aniden = false } = {}) {
    if (!n) return;
    const hedef = Math.max(0, Math.min(n - 1, Number(i) || 0));
    if (hedef === simdi) return;
    if (gecisSuruyor) { bekleyen = hedef; return; }   // son tuşu kuyruğa al, düşürme
    const onceki = simdi;
    const kaynak = onceki >= 0 ? slaytlar[onceki] : null;
    const varis = slaytlar[hedef];
    const yon = hedef > onceki ? 1 : -1;
    simdi = hedef;
    if (yaklasKipi) yaklasSifirla();
    /* sayaç, ilerleme ve adres hemen değişir (Doherty: geri bildirim geçişi beklemez) */
    kromGuncelle();
    if (yaz) adresYaz();
    const gecisli = kaynak && typeof ayar.gecisFn === 'function' && ayar.gecis !== 'none' && !ihracat && !aniden && !azHareket();
    if (gecisli) {
      gecisSuruyor = true;
      try {
        await ayar.gecisFn(kaynak, varis, {
          kind: varis.dataset.enter || ayar.gecis, direction: yon, stage: sahne,
          anchor: varis.dataset.anchor, activeClass: 'etkin',
        });
      } catch { /* geçiş çökse de dinlenme durumu aşağıda kurulur */ }
      gecisSuruyor = false;
    }
    dinlenmeDurumu();
    if (kaynak) ayar.onLeave?.(kaynak, onceki);
    ayar.onEnter?.(varis, simdi);
    if (bekleyen >= 0) { const b = bekleyen; bekleyen = -1; git(b); }
  }
  const ileri = () => git(simdi + 1);
  const geri = () => git(simdi - 1);

  /* ── notlar ── */
  function notlar(ac = notPaneli ? notPaneli.hidden : false) {
    if (!notPaneli) return;
    notPaneli.hidden = !ac;
    if (notMenu) notMenu.textContent = ac ? T('Sunucu notlarını gizle  N') : T('Sunucu notları  N');
  }

  /* ── tam ekran ── */
  function tamEkran() {
    if (document.fullscreenElement) { document.exitFullscreen?.().catch(() => {}); return; }
    const istek = kok.requestFullscreen?.bind(kok);
    if (!istek) { bildir(T('Tam ekran açılamadı')); return; }
    Promise.resolve().then(istek).catch(() => bildir(T('Tam ekran açılamadı')));
  }
  /* bazı gömülü tarayıcılar sözü (promise) reddetmez, yalnız olay yayar: ikisi de bildirir */
  dinle(document, 'fullscreenerror', () => bildir(T('Tam ekran açılamadı')));

  /* ── yazdır ── */
  function yazdir() { notlar(false); window.print(); }

  /* ── yardım ── */
  function yardim() {
    if (!yardimDlg) return;
    menuKapat();
    if (yardimDlg.open) yardimDlg.close(); else yardimDlg.showModal();
  }

  /* ── genel bakış ── */
  let bakisDonus = null;
  const kilitle = (durum) => { sahne.inert = durum; if (kumanda) kumanda.inert = durum; };
  function genelBakis(ac = bakis ? bakis.hidden : false) {
    if (!bakis) return;
    menuKapat();
    if (!ac) {
      bakis.hidden = true;
      kilitle(false);
      bakisB?.setAttribute('aria-pressed', 'false');
      bakisDonus?.focus?.();
      bakisDonus = null;
      return;
    }
    if (yaklasKipi) yaklasSifirla();
    bakisDonus = document.activeElement;
    bakis.replaceChildren();
    const bas = el('header');
    bas.append(el('h2', {}, T('Genel bakış · {n} slayt').replace('{n}', String(n))));
    bas.append(dugme(T('Kapat (Esc)'), T('Genel bakışı kapat (Esc)'), () => genelBakis(false), 'dk-kapat'));
    const liste = el('ol', { className: 'dk-kucukler' });
    bakis.append(bas, liste);
    slaytlar.forEach((s, i) => {
      const li = el('li');
      const b = el('button', { type: 'button' });
      b.setAttribute('aria-current', String(i === simdi));
      const cerceve = el('div', { className: 'dk-kucuk-sahne' });
      const kopya = sahne.cloneNode(false);
      kopya.removeAttribute('style');
      kopya.removeAttribute('id');
      kopya.classList.remove('dk-yaklas', 'dk-surukle');
      kopya.setAttribute('aria-hidden', 'true');
      const sk = s.cloneNode(true);
      sk.removeAttribute('aria-hidden');
      sk.removeAttribute('aria-labelledby');
      sk.inert = true;
      /* kimlikler çoğalmasın; ağır gömülüler (iframe, video, canvas) küçük resimde yer tutucuya döner */
      for (const x of [sk, ...sk.querySelectorAll('[id]')]) x.removeAttribute('id');
      for (const g of sk.querySelectorAll('iframe, video, canvas')) {
        const yt = el('div', { className: 'dk-yer-tutucu' }, g.getAttribute('title') || '');
        g.replaceWith(yt);
      }
      kopya.append(sk);
      cerceve.append(kopya);
      const baslik = s.querySelector('h1, h2')?.textContent.trim() ?? '';
      b.append(cerceve, el('span', { className: 'dk-ad' }, `${i + 1}. ${baslik}`));
      b.setAttribute('aria-label', T('{n}. slayt: {b}').replace('{n}', String(i + 1)).replace('{b}', baslik));
      b.addEventListener('click', () => { git(i); genelBakis(false); });
      li.append(b);
      liste.append(li);
    });
    bakis.hidden = false;
    kilitle(true);
    /* ölçek görünür hâlde, eşzamanlı ölçülür (rAF gizli sekmede çalışmaz) */
    liste.querySelectorAll('.dk-kucuk-sahne').forEach(c => { c.firstChild.style.transform = `scale(${c.clientWidth / SAHNE_G})`; });
    bakisB?.setAttribute('aria-pressed', 'true');
    liste.querySelector('[aria-current="true"]')?.focus();
  }

  /* ── yaklaştırma: sayfa büyütme ── */
  function cipYaz() {
    if (!cip) return;
    cip.hidden = !yaklasKipi;
    cip.textContent = T('Yaklaştırma {k} · tıkla: odak · +/−: kat · Esc: çık').replace('{k}', katYaz(yakin.kat));
  }
  function yaklasAyarla(yeni) {
    yakin = yaklasKirp(yeni);
    uygula();
    cipYaz();
  }
  function yaklas(ac = !yaklasKipi) {
    if (ihracat) return;
    if (!ac) return yaklasSifirla();
    if (yaklasKipi) return;
    yaklasKipi = true;
    sahne.classList.add('dk-yaklas');
    sahne.classList.toggle('dk-gecisli', !azHareket());
    if (yaklasMenu) yaklasMenu.textContent = T('Yaklaştırmayı kapat  Z');
    yaklasAyarla({ kat: 2, x: SAHNE_G / 2, y: SAHNE_Y / 2 });
  }
  function yaklasSifirla() {
    yaklasKipi = false;
    yakin = { kat: 1, x: SAHNE_G / 2, y: SAHNE_Y / 2 };
    uygula();
    cipYaz();
    if (yaklasMenu) yaklasMenu.textContent = T('Yaklaştır  Z');
    /* geçiş bitince sınıf kalkar: dönüş hareketi de yumuşak */
    const bitir = () => sahne.classList.remove('dk-yaklas', 'dk-gecisli');
    if (azHareket()) bitir(); else setTimeout(bitir, 260);
  }
  /* işaretçi: tıkla = odak, sürükle = kaydır (dokunmatik dahil) */
  let surukle = null;
  dinle(sahne, 'pointerdown', e => {
    if (!yaklasKipi || e.button !== 0) return;
    surukle = { x: e.clientX, y: e.clientY, ox: yakin.x, oy: yakin.y, oynadi: false };
    sahne.classList.add('dk-surukle');
    sahne.setPointerCapture?.(e.pointerId);
  });
  dinle(sahne, 'pointermove', e => {
    if (!surukle) return;
    const dx = e.clientX - surukle.x, dy = e.clientY - surukle.y;
    if (!surukle.oynadi && Math.hypot(dx, dy) < 6) return;
    surukle.oynadi = true;
    const sk = taban.s * yakin.kat;
    yaklasAyarla({ kat: yakin.kat, x: surukle.ox - dx / sk, y: surukle.oy - dy / sk });
  });
  const surukleBitir = e => {
    if (!surukle) return;
    const s0 = surukle; surukle = null;
    sahne.classList.remove('dk-surukle');
    if (!s0.oynadi) { const p = sahneNoktasi(e.clientX, e.clientY); yaklasAyarla({ kat: yakin.kat, x: p.x, y: p.y }); }
  };
  dinle(sahne, 'pointerup', surukleBitir);
  dinle(sahne, 'pointercancel', () => { surukle = null; sahne.classList.remove('dk-surukle'); });
  dinle(window, 'wheel', e => {
    if (!yaklasKipi || !e.ctrlKey) return;
    e.preventDefault();
    const p = sahneNoktasi(e.clientX, e.clientY);
    yaklasAyarla({ kat: yakin.kat * (e.deltaY < 0 ? 1.2 : 1 / 1.2), x: p.x, y: p.y });
  }, { passive: false });

  /* ── klavye ── */
  const OKLAR = ['ArrowRight', 'ArrowLeft', 'ArrowUp', 'ArrowDown', ' '];
  dinle(window, 'keydown', e => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const hedef = e.target instanceof Element ? e.target : null;
    const k = e.key;
    /* Esc öncelik sırası: menü → yardım → genel bakış → yaklaştırma → notlar */
    if (KISAYOLLAR.kapat.includes(k)) {
      if (menu && !menu.hidden) { e.preventDefault(); menuKapat(); digerB.focus(); }
      else if (yardimDlg?.open) { e.preventDefault(); yardimDlg.close(); }
      else if (bakis && !bakis.hidden) { e.preventDefault(); genelBakis(false); }
      else if (yaklasKipi) { e.preventDefault(); yaklasSifirla(); }
      else notlar(false);
      return;
    }
    if (hedef?.closest('input, textarea, select, dialog, [contenteditable=true]')) return;
    if (bakis && !bakis.hidden) {
      if (KISAYOLLAR.genelBakis.includes(k)) { e.preventDefault(); genelBakis(false); }
      else if (k === 'ArrowRight' || k === 'ArrowLeft') {
        const d = [...bakis.querySelectorAll('.dk-kucukler button')];
        const i = d.indexOf(document.activeElement);
        if (i >= 0) { e.preventDefault(); d[Math.max(0, Math.min(d.length - 1, i + (k === 'ArrowRight' ? 1 : -1)))].focus(); }
      }
      return;
    }
    if (OKLAR.includes(k) && hedef?.closest('[data-owns-arrows], [data-owns-keys]')) return;
    if (yaklasKipi) {
      /* yaklaşmışken oklar slayt değil odak kaydırır; Enter/Boşluk bir şey yapmaz */
      const adim = 120;
      if (k === 'ArrowRight') { e.preventDefault(); yaklasAyarla({ ...yakin, x: yakin.x + adim }); return; }
      if (k === 'ArrowLeft') { e.preventDefault(); yaklasAyarla({ ...yakin, x: yakin.x - adim }); return; }
      if (k === 'ArrowDown') { e.preventDefault(); yaklasAyarla({ ...yakin, y: yakin.y + adim }); return; }
      if (k === 'ArrowUp') { e.preventDefault(); yaklasAyarla({ ...yakin, y: yakin.y - adim }); return; }
      if (k === ' ' || k === 'Enter') { e.preventDefault(); return; }
      if (k === '+' || k === '=') { e.preventDefault(); yaklasAyarla({ ...yakin, kat: yakin.kat * 1.5 }); return; }
      if (k === '-' || k === '_') { e.preventDefault(); yaklasAyarla({ ...yakin, kat: yakin.kat / 1.5 }); return; }
    }
    if (KISAYOLLAR.ileri.includes(k)) { e.preventDefault(); ileri(); }
    else if (KISAYOLLAR.geri.includes(k)) { e.preventDefault(); geri(); }
    else if (KISAYOLLAR.bas.includes(k)) { e.preventDefault(); git(0); }
    else if (KISAYOLLAR.son.includes(k)) { e.preventDefault(); git(n - 1); }
    else if (KISAYOLLAR.genelBakis.includes(k)) { e.preventDefault(); genelBakis(); }
    else if (KISAYOLLAR.notlar.includes(k)) { e.preventDefault(); notlar(); }
    else if (KISAYOLLAR.tamEkran.includes(k)) { e.preventDefault(); tamEkran(); }
    else if (KISAYOLLAR.yaklas.includes(k)) { e.preventDefault(); yaklas(); }
    else if (KISAYOLLAR.yazdir.includes(k)) { e.preventDefault(); yazdir(); }
    else if (KISAYOLLAR.yardim.includes(k)) { e.preventDefault(); yardim(); }
  });

  /* gömülü sahne (iframe) kendi betiğiyle odağı alırsa tuşlar desteye gelmez (iframe içindeki keydown
     üst pencereye çıkmaz). Klavyeyi sahiplenmek isteyen iframe data-owns-keys taşır; ötekilerden odak geri alınır. */
  dinle(window, 'blur', () => {
    setTimeout(() => {
      const a = document.activeElement;
      if (a?.tagName === 'IFRAME' && sahne.contains(a) && !a.closest('[data-owns-keys], [data-owns-arrows]')) a.blur();
    }, 0);
  });

  /* ── dokunmatik: yatay kaydırma ≥ 60 px gezinir; yaklaşmışken kaydırma işaretçiyle pan'dır ── */
  let dokunX = null, dokunY = null;
  dinle(window, 'touchstart', e => { dokunX = e.touches[0].clientX; dokunY = e.touches[0].clientY; }, { passive: true });
  dinle(window, 'touchend', e => {
    if (dokunX === null) return;
    const dx = e.changedTouches[0].clientX - dokunX, dy = e.changedTouches[0].clientY - dokunY;
    dokunX = dokunY = null;
    if (yaklasKipi) return;
    if (Math.abs(dx) >= 60 && Math.abs(dx) > Math.abs(dy)) git(simdi + (dx < 0 ? 1 : -1));
  }, { passive: true });

  /* ── kumanda 4 s hareketsizlikte söner; menü açıkken ya da odak çubuktayken sönmez ── */
  let sonme = 0;
  const uyan = () => {
    if (!kumanda) return;
    kumanda.classList.remove('soluk');
    clearTimeout(sonme);
    sonme = setTimeout(() => {
      if (menu.hidden && !kumanda.contains(document.activeElement)) kumanda.classList.add('soluk');
    }, 4000);
  };
  if (kumanda) {
    dinle(window, 'pointermove', uyan, { passive: true });
    dinle(window, 'pointerdown', uyan, { passive: true });
    dinle(window, 'keydown', uyan);
    dinle(kumanda, 'focusin', uyan);
    dinle(kumanda, 'focusout', uyan);
    uyan();
  }

  /* ── adres ve ölçek ── */
  dinle(window, 'hashchange', () => {
    const i = adrestenIndeks(location.hash, n);
    git(i, { yaz: false });
    if (location.hash !== `#${i + 1}`) adresYaz();
  });
  dinle(window, 'resize', olcekle);
  document.fonts?.ready.then(olcekle);

  function yokEt() {
    for (const t of temizlik) t();
    temizlik.length = 0;
    clearTimeout(sonme); clearTimeout(bildirimZaman);
    for (const e of uretilen) e.remove();
    uretilen.length = 0;
    sahne.classList.remove('dk-sahne', 'dk-yaklas', 'dk-gecisli', 'dk-surukle');
    sahne.style.removeProperty('transform');
    sahne.inert = false;
  }

  olcekle();
  git(adrestenIndeks(location.hash, n), { aniden: true });

  return {
    git, ileri, geri,
    index: () => simdi, sayi: () => n,
    tamEkran, genelBakis, notlar, yaklas, yaklasSifirla, yardim, yokEt,
  };
}
