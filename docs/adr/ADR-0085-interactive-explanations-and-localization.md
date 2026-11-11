# ADR-0085: Interactive explanations and localization-ready prose

- Status: Accepted
- Date: 2026-11-11

## Context

The cross-cutting roadmap item "Accessibility (ADR-0078) and
localization-ready prose policy; interactive explanations remain" had two
open parts. ADR-0078 section 6 set a policy for prose (one formatted string
per sentence, explicit number formats) but did not centralize any string,
and section 7 left interactive explanations out of scope because the library
had no JavaScript or notebook layer.

Two constraints shaped this decision. The library is standard-library only
at run time, with Matplotlib optional, so an interactive page cannot depend
on a plotting or UI framework. And every number the library shows is traced
to an `EQ-*` record and a regression test, so a page must not re-implement
the mathematics in another language where it could drift from the Python
code.

The English study reports (`UQStudy.report()` and
`probstat.visualization.study.study_html`) are consumed by existing users and
tests; centralizing their strings must not change a byte of them.

## Decision

### 1. Interactive explainers as single offline HTML files

`probstat.visualization.interactive` writes self-contained explainers: one
HTML file each, with inline CSS, inline SVG charts, and a 37-line vanilla
script. There is no external script, stylesheet, font, image, or network
request, and no `http(s)` URL anywhere in the file.

Every number is computed in Python by the library function the page
explains and embedded twice as JSON:

- `explainer-data` holds the raw library outputs (for example every
  `kolmogorov_normal` value of `clt_error_profile`), so a reader or a test
  can check each number against a fresh library call;
- `explainer-view` holds the page states: for each slider position and mode,
  the SVG attributes and sentences to show.

The script only reads the integer slider position and the pressed mode
button and copies the selected state into the document. It evaluates no
distribution, bound, or estimator; `Math.` does not appear in it (tested).
The SVG path coordinates are also precomputed, so even the data-to-pixel
mapping happens in Python. The initial state is rendered into the markup,
so without script the page still shows the default numbers and a
`noscript` note says the controls are inactive.

Three explainers are provided, each tied to existing functionality:

| Function | Library calls | Control | Charts |
|---|---|---|---|
| `clt_explainer` | `clt_error_profile`, `standardized_mean_cdf` (Exponential(1) summands) | slider over 23 sample sizes, `n = 1 ... 1000` | `F_n(z) - Phi(z)` with dashed lines at `+-K_n`; log-log `K_n` and the Berry-Esseen bound with a marker at the selected `n` |
| `bayesian_updating_explainer` | `demand_failure_probability` (Beta posterior and credible interval), `wilson_score_interval`, `Beta.pdf` | slider over 0 to 40 observed trials; two buttons: credible or Wilson interval | prior, likelihood scaled to unit area, posterior, shaded interval, dashed true `p` |
| `monte_carlo_error_explainer` | `convergence_trace` of `pi_estimate_traces` (4 seeds, `N = 10 ... 50 000`) | slider over 12 checkpoints; four seed buttons | every seed's estimate in muted ink, the selected seed with its 95% band, dashed `pi` |

`write_explainers(directory, locale=...)` writes the three default pages.
Each page cites its `EQ-*` keys and `REF-*` sources visibly in a footer list
and in the method paragraphs. The likelihood drawn on the density axis is a
new registered equation, [EQ-VIZ-LIK-001].

Accessibility follows ADR-0078 and the WAI-ARIA and SVG specifications
[REF-W3C-WAI-ARIA-12-2023], [REF-W3C-SVG2-2018]:

- `lang` on the document, a skip link to `main`, and the landmarks
  `header`, `main` (two `section` elements labelled by their headings), and
  `footer`;
- native controls only: `input type="range"` with a `label`, an `output`,
  and an `aria-valuetext` that states the value (`n = 30`, not `12`), and
  `button` elements with `aria-pressed` inside a labelled `role="group"`;
  every control is at least 44 px high and has a 3 px focus ring
  (`:focus-visible`);
- an `aria-live="polite"` paragraph that states the current numbers in
  full sentences;
- every chart is `<svg role="img">` labelled by its `title` and a `desc`
  that the script updates with the state (WCAG 2.1 SC 1.1.1);
- colours are the chart palette tokens of ADR-0024: porcelain under the
  default scheme and graphite under `prefers-color-scheme: dark`
  [REF-W3C-MEDIAQUERIES-5]. Marks use the series hues (at least 3:1);
  direct labels use the series hue mixed toward the ink until it reaches
  4.5:1 (`legible_colour`), which also closes the light-palette label gap
  that ADR-0078 recorded for these pages.

Charts follow the figure rules: direct labels at line ends spread at least
20 user units apart so no two labels overlap, horizontal hairline grid, a
horizontal y title, uncertainty bands in the hue of their line at 0.16
opacity, accent only for reference lines (bound, true value, `pi`), and
colour never the only cue (dash patterns and the key). The SVG is drawn at
600 user units with 15 px text; it is at least 480 px wide (12 px text) and
scrolls inside its figure on narrower screens rather than shrinking the
text below the 9 pt floor of ADR-0083.

The lab `examples/interactive_explainers_lab.py` writes the pages in English
and Turkish to `examples/generated/`, prints their opening numbers, and
saves one static themed figure (`plot_monte_carlo_convergence` of seed 1)
through `save_figure`, so the lab satisfies the figure rule of the lab
contract with a file that carries its text description.

### 2. A message catalogue with explicit locales

`probstat.foundations.messages` holds the catalogue API; the entries live in
`_messages_en.py` and `_messages_tr.py`. It is in `foundations` (layer 0)
because both `workflow` (the Markdown report) and `visualization` (the HTML
report and the explainers) use it. No package `__all__` changed; the
modules are imported by their path.

- An entry maps a key to one `str.format` template with named placeholders,
  for example `"report.budget": "Evaluation budget: {budget:,} ({remaining:,} remaining)"`.
- `message(key, locale, **fields)` formats an entry; numeric fields are
  formatted with the specification written in the entry and then localized
  with the CLDR number symbols of the locale [REF-UNICODE-CLDR],
  [REF-UNICODE-TR35], [REF-PYTHON-FORMAT-SPEC]: Turkish uses the decimal
  comma and the group dot (`0,0243`, `9.309`). String fields are inserted
  unchanged.
- `html_message` escapes both the literal text and the fields, so markup is
  never part of an entry and a recorded label cannot inject markup.
- An entry whose wording depends on a count is a mapping from CLDR plural
  categories to templates; `message` chooses the form from the `count`
  field (`plural_category`: English `one` only for the integer 1, Turkish
  `one` for 1).
- `locale` is always an explicit keyword argument (`"en"` by default). The
  environment (`LANG`, `LC_ALL`, ...) is never read; a test sets Turkish
  environment variables and checks that the default report is still the
  English one.

`UQStudy.report(locale=...)`, `study_html(study, locale=...)`, and
`write_study_html(study, path, locale=...)` take the new argument. Every
sentence of both reports now comes from the catalogue; the English
catalogue reproduces the previous output byte for byte. A complete Turkish
catalogue covers every key of both reports and of the three explainers
(152 keys per locale).

What is not translated: recorded data (stage kinds such as `propagation`,
stage labels, summary keys, settings written as `name=value`, warning and
error texts recorded by the stages) and the Matplotlib figures embedded in
the HTML report, whose titles and descriptions stay English; in a
non-English report each such `figure` element carries `lang="en"` (WCAG 2.1
SC 3.1.2). Copula family names are identifiers and stay as recorded
(`Gaussian kopulası`).

### 3. Prose policy (replaces ADR-0078 section 6)

1. **One formatted string per sentence.** A sentence is one catalogue
   entry with named placeholders. Code never builds a sentence from
   translated fragments. A paragraph of several sentences is several
   entries joined with a space; a list of facts (`A · B · C`) joins whole
   entries with a separator, never parts of one sentence.
2. **Markup outside entries.** Bold, links, and HTML tags are added by the
   caller around a formatted entry, or passed as a field value; entries are
   plain text.
3. **Numbers by specification.** Every number is a placeholder with an
   explicit format specification (`{count:,}`, `{value:.3g}`); separators
   come from the locale. A literal number in an entry is written in that
   locale's notation (`C = 0,4748` in Turkish).
4. **Lists and intervals.** Where the decimal separator is a comma,
   interval limits and parameter lists are separated by `; `
   (`[0,251; 0,749]`, `Beta(7; 7)`).
5. **Plurals by category.** A count-dependent sentence is a `one`/`other`
   mapping, never `"trial(s)"` in new text. Both locales must define the same
   categories with the same placeholders (tested).
6. **No suffix on a placeholder.** A Turkish case suffix after a number
   (`%{share}'i`) depends on how the number is read aloud; entries are
   phrased so that no suffix attaches to a placeholder.
7. **Same placeholders everywhere.** For each key and plural form, every
   locale uses the same placeholder names with the same specifications
   (tested), so a translation cannot silently drop or reformat a number.
8. **Explicit locale.** Public functions take `locale=`; nothing reads the
   environment.

Existing English report wording that does not inflect ("1 cache hits",
"stage(s)") is kept so the output is unchanged; its `one` form equals its
`other` form, and correcting it later is an edit to one entry.

## Consequences

- The roadmap item has all three parts: accessible output (ADR-0078), a
  catalogue-backed prose policy with a complete second locale, and offline
  interactive explanations.
- Adding a language is adding one catalogue module and one `NUMBER_SYMBOLS`
  entry; the tests then require it to cover every key with matching
  placeholders.
- The explainer pages are large because every state is precomputed: about
  166 kB (CLT), 337 kB (Bayesian), and 43 kB (Monte Carlo). The Bayesian
  page stores 41 states of two 100-point densities twice (raw and as paths).
- The pages need a browser with SVG and ES5 (no build step); there is no
  hover or zoom, by design.
- With the lab seed 2026 the 40 Bernoulli(0.3) trials give 20 successes,
  a tail draw (`P(X >= 20) = 0.0063`), so after all trials both 95%
  intervals exclude `p = 0.3`. The seed is kept: the lab prints the verdict
  and states the probability, and the page shows what a miss looks like.
- New references: [REF-UNICODE-CLDR], [REF-UNICODE-TR35],
  [REF-PYTHON-FORMAT-SPEC], [REF-W3C-WAI-ARIA-12-2023], [REF-W3C-SVG2-2018],
  [REF-W3C-MEDIAQUERIES-5]. New equation: [EQ-VIZ-LIK-001].

## Verification

### Tests and checks

- `tests/test_localization.py` (14 tests): the English Markdown and HTML
  reports of a study that reaches every branch of both templates (failed
  runs, cache hits, budget, aborted stage, error, settings, warnings,
  non-finite values, a copula, more than seven stages, a surrogate child)
  are compared byte for byte with `tests/fixtures/study_report_en.txt` and
  `.html`, which were written by the code at `a71141b`, before the change
  (fixtures are stored with `-text` in `.gitattributes`, so checkouts do not
  rewrite their line ends). The same reports are unchanged under Turkish
  environment variables. The Turkish catalogue has every key, the same
  plural forms, and the same placeholders and specifications; Turkish
  numbers use the decimal comma; the Turkish reports translate every label.
- `tests/test_interactive_explainers.py` (17 tests): no `http(s)` URL,
  `<script src`, `<link`, `@import`, `@font-face`, `url(` other than a
  fragment, `fetch`, or `XMLHttpRequest`; the script contains no `Math.`
  call; the embedded data equals `clt_error_profile`,
  `standardized_mean_cdf`, `demand_failure_probability`,
  `wilson_score_interval`, `Beta.pdf`, and `convergence_trace` called again;
  every control is native and labelled; one `aria-live` region states the
  numbers; every SVG has `role="img"` with a title and a state-dependent
  description; light and dark tokens equal the porcelain and graphite
  palettes and label tokens reach 4.5:1; every cited key appears in the page
  text; every state addresses existing elements.
- `PYTHONPATH=src python -m pytest -q -p no:cacheprovider
  tests/test_documentation_contract.py tests/test_architecture.py
  tests/test_lab_contract.py tests/test_interactive_explainers.py
  tests/test_localization.py tests/test_workflow_study.py
  tests/test_visualization*.py tests/test_api_reference.py`: 149 passed.
- `python -m ruff check .` and `python -m mypy src/probstat`: no findings.
- Every lab runs with `MPLBACKEND=Agg`.

### Figure audit

The static figure of the lab (`interactive-monte-carlo-seed-1.png`) was
checked with `datawiz.figur_denetle(fig, ortam="web")` in both palettes:
0 HATA, 0 UYARI, 5 BİLGİ each (`type3-riski` for a PDF export that is not
made, `hedef-yok`, two `birim` notes for unitless axes, and
`belirsizlik-tanimi`, which the subtitle "Band: its reported confidence
interval" and the 95% title answer). The PNG was inspected: no overlapping
text, the band is in the line's hue, the reference label sits right of the
plot.

The SVG charts were checked against the same rules by construction and by
eye in headless Chrome and Edge screenshots of all three pages, in the light
and the dark scheme, at the default state and at a changed state (slider at
`n = 1`, 40 trials with the Wilson interval, `N = 50` with seed 2): direct
labels do not overlap each other or the marks, the density axis starts
at zero, the log axes are labelled at powers of ten, the shaded Bayesian interval
spans the full plot height from the axis (the pages draw no bars), text is
ink, muted ink, accent, or a label
token (all at least 4.5:1), and the series hues are the CVD-validated
palette of ADR-0024. At the narrowest window headless Chrome allows
(504 px), the page width is 489 px (no horizontal page scroll).

### UX audit

Task budget, written before the markup (identical for the three pages
except where noted):

```
GÖREV BÜTÇESİ — explainer page                   (yüzey: web)
Birincil görev(ler)       : 1 seçili n / deneme / N'de sayıları okumak; 2 eğriyle karşılaştırmak   Pareto · Occam
Birincil eylem            : kaydırıcıyı hareket ettirmek (vurgulu düğme yok)                       Von Restorff
Eylem / görünüm           : ≤ 5     · planlanan: CLT 0 · Bayes 2 · MC 4 düğme                     Hick
Üst düzey gezinti         : ≤ 7     · planlanan: 0 (yalnız atlama bağlantısı)                       Hick · Miller
Seçenek kümesi            : ≤ 5 ya da varsayılan işaretli · kaydırıcı varsayılanlı (n = 30, 10 deneme, N = 1 000)   Seçim Yükü
Form alanı / adım         : ≤ 7     · planlanan: 1 kaydırıcı + en çok 1 düğme grubu                Miller · Parkinson
Geri bildirim             : < 400 ms · önceden hesaplanmış durum seçimi, eşzamanlı                  Doherty
Stil bütçesi              : 4 punto · 1 düğme stili · 1 vurgu tonu · 0 gölge · 1 yarıçap           Benzerlik · Prägnanz
Sisteme yüklenen karmaşıklık : bütün sayılar Python'da hesaplanır; renk şeması işletim sisteminden; varsayılan durum seçili; kullanıcıya ayar sorulmaz   Tesler
```

The checker
`node ux-yasa-denetimi.mjs <file> --tablo` ran on all six generated pages:
**0 HATA · 0 UYARI** for each of `clt-explainer.html`,
`clt-explainer.tr.html`, `bayesian-updating-explainer.html`,
`bayesian-updating-explainer.tr.html`, `monte-carlo-error-explainer.html`,
and `monte-carlo-error-explainer.tr.html`. Eye tests were measured in
headless Chrome (window 1280 x 800, viewport 1262 x 704) with a probe
script reading `getBoundingClientRect`, and the state switch was timed over
every state with `performance.now()`. The English and Turkish files of one
explainer have the same structure; where a measurement differs, both values
are given (EN / TR).

Checker limits that apply here: series colours are CSS custom properties,
which the saturated-hue counter does not read, so row 10 counts hues by
eye; the static scan sees the full DOM because the script changes only
attributes and text, never structure.

YASA DENETİMİ — clt-explainer.html, clt-explainer.tr.html · 2026-11-11 ·
denetleyici koşuldu: 0 HATA · 0 UYARI

Sonuç: 20/20 yasa ölçüldü · 0 HATA · 0 UYARI · karar: TESLİM

| # | Yasa | Ölçüm | Eşik | Durum | Gerekçe (yalnız UYARI) |
|---|---|---|---|---|---|
| 1 | Hick | nav 0 madde · görünüm max 0 eylem · select 0 | ≤ 7 · ≤ 5 · ≤ 7 | geçti | |
| 2 | Fitts | kaydırıcı 728 × 44 px; CSS min 44 px; gönder eylemi yok (girdi→eylem mesafesi uygulanmaz) | ≥ 44 px · ≤ 24 px | geçti | |
| 3 | Jakob | imleç/kaydırma/seçim bulgusu 0; yerel `range`, ok/Home/End tuşları, Tab sırası belge sırası | 0 sapma | geçti | |
| 4 | Seçim Yükü | radyo 0; kaydırıcı 23 durum, varsayılan n = 30 seçili | ≤ 5 ya da varsayılan | geçti | |
| 5 | Yakınlık | etiket→kaydırıcı 4 px / denetim→okuma alanı 16 px = 1:4 | ≤ 1:2 | geçti | |
| 6 | Benzerlik | 4 punto · 1 yarıçap · 0 gölge | ≤ 8 · ≤ 4 · ≤ 2 | geçti | |
| 7 | Tekdüze Bağlılık | 0 kart, iç içe 0 | ≤ 12 · 0 | geçti | |
| 8 | Ortak Bölge | tek öğeli kutu 0; üst düzey bölge 3 (header, main, footer) | 0 · ≤ 4 | geçti | |
| 9 | Prägnanz | cam 0 · ışıma 0 · degrade 0 | 0 · 0 · ≤ 3 | geçti | |
| 10 | Von Restorff | birincil düğme 0; vurgu tonu 1 (accent: odak, kaydırıcı, okuma alanı kenarı); seri tonları yalnız grafikte | 1 · ≤ 2 | geçti | |
| 11 | Miller | form 0 alan · tablo 0 sütun; denetim 1 | ≤ 7 · ≤ 7 | geçti | |
| 12 | Sıra Konumu | nav yok; ilk öğe atlama bağlantısı → keşif bölümü, son: kaynaklar | birincil görevler | geçti | |
| 13 | Doherty | geçiş 0 ms; durum değişimi ortalama 4,1 / 5,0 ms, en kötü 84 / 103 ms (23 durum) | ≤ 300 · 0 · < 400 ms | geçti | |
| 14 | Zeigarnik | adım 0 | ≥ 3 adımda gösterge | geçti | |
| 15 | Zirve-Son | `aria-live` okuma alanı var; boş durum yok (her durum sayı taşır); betik kapalıyken `noscript` notu; n = 1 durumunda sınırın 1'i aştığı ayrı cümleyle söylenir | 3/3 ekran | geçti | |
| 16 | Parkinson | zorunlu 0/0 · autocomplete eksik 0 | ≤ %50 · 0 | geçti | |
| 17 | Tesler | zorunlu girdi 0; sayılar önceden hesaplı, varsayılan n seçili | her girdiye gerekçe | geçti | |
| 18 | Postel | pattern 0; kaydırıcı yalnız geçerli durumları üretir | 0 · normalizasyon | geçti | |
| 19 | Occam | sayfa 99 öğe · görünüm max 64; her öğe bir göreve (okuma, grafik, yöntem, kaynak) bağlı | ≤ 900 · ≤ 150 · 0 | geçti | |
| 20 | Pareto | ilk ekranın %71'i keşif bölümü (denetim, okuma alanı, ilk grafik başlığı) | ≥ %60 | geçti | |

YASA DENETİMİ — bayesian-updating-explainer.html,
bayesian-updating-explainer.tr.html · 2026-11-11 · denetleyici koşuldu:
0 HATA · 0 UYARI

Sonuç: 20/20 yasa ölçüldü · 0 HATA · 0 UYARI · karar: TESLİM

| # | Yasa | Ölçüm | Eşik | Durum | Gerekçe (yalnız UYARI) |
|---|---|---|---|---|---|
| 1 | Hick | nav 0 madde · görünüm max 2 eylem · select 0 | ≤ 7 · ≤ 5 · ≤ 7 | geçti | |
| 2 | Fitts | kaydırıcı 728 × 44 px; düğmeler 198 × 44 ve 184 × 44 / 188 × 44 ve 174 × 44 px; gönder eylemi yok | ≥ 44 px · ≤ 24 px | geçti | |
| 3 | Jakob | imleç/kaydırma/seçim bulgusu 0; yerel `range` ve `button`, `aria-pressed` geçiş düğmeleri | 0 sapma | geçti | |
| 4 | Seçim Yükü | radyo 0; 2 aralık düğmesi, varsayılan "güvenilir aralık" basılı; kaydırıcı varsayılanı 10 deneme | ≤ 5 ya da varsayılan | geçti | |
| 5 | Yakınlık | etiket→kaydırıcı 4 px / denetim grupları arası 16 px = 1:4; düğmeler arası 8 px | ≤ 1:2 | geçti | |
| 6 | Benzerlik | 4 punto · 1 yarıçap · 0 gölge · 1 düğme stili | ≤ 8 · ≤ 4 · ≤ 2 | geçti | |
| 7 | Tekdüze Bağlılık | 0 kart, iç içe 0 | ≤ 12 · 0 | geçti | |
| 8 | Ortak Bölge | tek öğeli kutu 0; üst düzey bölge 3 | 0 · ≤ 4 | geçti | |
| 9 | Prägnanz | cam 0 · ışıma 0 · degrade 0 | 0 · 0 · ≤ 3 | geçti | |
| 10 | Von Restorff | birincil düğme 0; vurgu tonu 1 (basılı düğme kenarı ve odak); seri tonları yalnız grafik ve anahtarda | 1 · ≤ 2 | geçti | |
| 11 | Miller | form 0 alan · tablo 0 sütun; anahtar 5 madde | ≤ 7 · ≤ 7 | geçti | |
| 12 | Sıra Konumu | nav yok; ilk öğe atlama bağlantısı, son: kaynaklar | birincil görevler | geçti | |
| 13 | Doherty | geçiş 0 ms; durum değişimi ortalama 1,6 / 1,3 ms, en kötü 108 / 88 ms (82 durum) | ≤ 300 · 0 · < 400 ms | geçti | |
| 14 | Zeigarnik | adım 0 | ≥ 3 adımda gösterge | geçti | |
| 15 | Zirve-Son | `aria-live` okuma alanı; 0 deneme durumu ayrı cümle ("Wilson aralığı yok, en az bir deneme gerekir") ve bant gizlenir; `noscript` notu | 3/3 ekran | geçti | |
| 16 | Parkinson | zorunlu 0/0 · autocomplete eksik 0 | ≤ %50 · 0 | geçti | |
| 17 | Tesler | zorunlu girdi 0; önsel, p, tohum kodda; varsayılanlar seçili | her girdiye gerekçe | geçti | |
| 18 | Postel | pattern 0; yalnız geçerli durumlar seçilebilir | 0 · normalizasyon | geçti | |
| 19 | Occam | sayfa 103 öğe · görünüm max 61 | ≤ 900 · ≤ 150 · 0 | geçti | |
| 20 | Pareto | ilk ekranın %71'i keşif bölümü | ≥ %60 | geçti | |

YASA DENETİMİ — monte-carlo-error-explainer.html,
monte-carlo-error-explainer.tr.html · 2026-11-11 · denetleyici koşuldu:
0 HATA · 0 UYARI

Sonuç: 20/20 yasa ölçüldü · 0 HATA · 0 UYARI · karar: TESLİM

| # | Yasa | Ölçüm | Eşik | Durum | Gerekçe (yalnız UYARI) |
|---|---|---|---|---|---|
| 1 | Hick | nav 0 madde · görünüm max 4 eylem · select 0 | ≤ 7 · ≤ 5 · ≤ 7 | geçti | |
| 2 | Fitts | kaydırıcı 728 × 44 px; tohum düğmeleri 85–87 × 44 / 100–101 × 44 px; gönder eylemi yok | ≥ 44 px · ≤ 24 px | geçti | |
| 3 | Jakob | imleç/kaydırma/seçim bulgusu 0; yerel `range` ve `button` | 0 sapma | geçti | |
| 4 | Seçim Yükü | 4 tohum düğmesi, varsayılan "Tohum 1" basılı; kaydırıcı varsayılanı N = 1 000 | ≤ 5 ya da varsayılan | geçti | |
| 5 | Yakınlık | etiket→kaydırıcı 4 px / denetim grupları arası 16 px = 1:4; düğmeler arası 8 px | ≤ 1:2 | geçti | |
| 6 | Benzerlik | 4 punto · 1 yarıçap · 0 gölge · 1 düğme stili | ≤ 8 · ≤ 4 · ≤ 2 | geçti | |
| 7 | Tekdüze Bağlılık | 0 kart, iç içe 0 | ≤ 12 · 0 | geçti | |
| 8 | Ortak Bölge | tek öğeli kutu 0; üst düzey bölge 3 | 0 · ≤ 4 | geçti | |
| 9 | Prägnanz | cam 0 · ışıma 0 · degrade 0 | 0 · 0 · ≤ 3 | geçti | |
| 10 | Von Restorff | birincil düğme 0; vurgu tonu 1; seçili tohum tek renkli çizgi, diğerleri soluk mürekkep | 1 · ≤ 2 | geçti | |
| 11 | Miller | form 0 alan · tablo 0 sütun; 4 tohum, anahtar 3 madde | ≤ 7 · ≤ 7 | geçti | |
| 12 | Sıra Konumu | nav yok; ilk öğe atlama bağlantısı, son: kaynaklar | birincil görevler | geçti | |
| 13 | Doherty | geçiş 0 ms; durum değişimi ortalama 3,2 / 2,5 ms, en kötü 125 / 101 ms (48 durum) | ≤ 300 · 0 · < 400 ms | geçti | |
| 14 | Zeigarnik | adım 0 | ≥ 3 adımda gösterge | geçti | |
| 15 | Zirve-Son | `aria-live` okuma alanı; aralık π'yi kaçırdığında ayrı cümle ("does not contain π"); `noscript` notu | 3/3 ekran | geçti | |
| 16 | Parkinson | zorunlu 0/0 · autocomplete eksik 0 | ≤ %50 · 0 | geçti | |
| 17 | Tesler | zorunlu girdi 0; tohumlar ve kontrol noktaları kodda | her girdiye gerekçe | geçti | |
| 18 | Postel | pattern 0; yalnız geçerli durumlar seçilebilir | 0 · normalizasyon | geçti | |
| 19 | Occam | sayfa 93 öğe · görünüm max 60 | ≤ 900 · ≤ 150 · 0 | geçti | |
| 20 | Pareto | ilk ekranın %66'sı / %71'i keşif bölümü (İngilizce başlık iki satır) | ≥ %60 | geçti | |

Risks that could change the decision: target sizes and spacing were
measured in Chromium only; the 480 px minimum chart width means the chart
scrolls inside its figure on a 375 px phone, which was not rendered (the
headless window cannot be narrower than 504 px); timings are from a
headless desktop run.
