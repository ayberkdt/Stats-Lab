#!/usr/bin/env node
/* Node gate for the deck's data consistency (ADR-0097).
 *
 *   node docs/presentation/runtime/denetle-cli.mjs docs/presentation/index.html [more.html ...]
 *
 * Reads each deck's HTML as text, collects its slides (section.slayt) with headline, bullets
 * (li.madde), visible text (speaker notes removed), and bound values (<span data-k>), and runs
 * denetle() from deck-data.mjs against ../deck-data.json. Exit code 1 when any warning is left. */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { denetle } from './deck-data.mjs';

const burasi = path.dirname(fileURLToPath(import.meta.url));
const veri = JSON.parse(fs.readFileSync(path.join(burasi, '..', 'deck-data.json'), 'utf8'));
const ac = s => s.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&nbsp;/g, ' ');
const duz = s => ac(s.replace(/<[^>]*>/g, ' ')).replace(/\s+/g, ' ').trim();

export function slaytlar(html) {
  const parcalar = html.split(/(?=<section\b[^>]*class="[^"]*\bslayt\b)/).slice(1);
  return parcalar.map(p => {
    const govde = p.slice(0, p.indexOf('</section>'));
    const notsuz = govde.replace(/<aside class="notlar">[\s\S]*?<\/aside>/g, '');
    const bas = /<h[12][^>]*>([\s\S]*?)<\/h[12]>/.exec(govde);
    return {
      id: /id="([^"]+)"/.exec(govde)?.[1] ?? '',
      type: /data-tur="([^"]+)"/.exec(govde)?.[1] ?? '',
      headline: bas ? duz(bas[1]) : '',
      bullets: [...govde.matchAll(/<li class="madde"[^>]*>([\s\S]*?)<\/li>/g)].map(m => duz(m[1])),
      text: duz(notsuz),
      bound: [...govde.matchAll(/<([a-z0-9]+)\b([^<>]*\bdata-k="([^"]+)"[^<>]*)>([^<]*)<\/\1>/g)].map(m => ({
        key: m[3], fmt: /data-fmt="([^"]+)"/.exec(m[2])?.[1] ?? 'str', text: ac(m[4]),
      })),
    };
  });
}

const dosyalar = process.argv.slice(2);
if (!dosyalar.length) { console.error('usage: node denetle-cli.mjs <deck.html> [...]'); process.exit(2); }
let toplam = 0;
for (const dosya of dosyalar) {
  const html = fs.readFileSync(dosya, 'utf8');
  const lang = /<html[^>]*\blang="([a-z]+)"/.exec(html)?.[1] ?? 'en';
  const s = slaytlar(html);
  const u = denetle(s, veri, lang);
  if (!s.length) u.push('no section.slayt found: nothing was checked');
  toplam += u.length;
  console.log(`${path.basename(dosya)}: ${s.length} slides, ${s.reduce((n, x) => n + x.bound.length, 0)} bound values, ${u.length} warnings`);
  for (const m of u) console.log(`  ${m}`);
}
process.exit(toplam ? 1 : 0);
