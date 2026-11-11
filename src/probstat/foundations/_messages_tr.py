"""Turkish message catalogue (ADR-0085).

Every English key has a Turkish entry with the same placeholders and the
same format specifications; numbers get the Turkish decimal comma and
group dot from :mod:`probstat.foundations.messages`. Turkish nouns stay
singular after a numeral, so the ``one`` and ``other`` forms coincide.
Sentences avoid case suffixes on a placeholder (``%{share}'i``), whose
form would depend on how the number is read aloud.
"""

from __future__ import annotations

from collections.abc import Mapping

MESSAGES: Mapping[str, str | Mapping[str, str]] = {
    "study.title.forward": "Belirsizlik çalışması: {name}",
    "study.title.surrogate": "Vekil model çalışması: {name}",
    "study.dependence.independent": "bağımsız",
    "study.dependence.copula": "{family} kopulası",
    "study.stage_heading": "{index}. {stage}: {label}",
    "study.error": "Hata: {error}",
    "study.settings": "Ayarlar: {settings}",
    "study.warning_label": "Uyarı:",
    "study.table.quantity": "Nicelik",
    "study.table.value": "Değer",
    "study.status.complete": "tamamlandı",
    "study.status.aborted": "yarıda kesildi",
    "study.status.budget-exceeded": "bütçe aşıldı",
    "report.inputs": "Girdiler: {names}",
    "report.quantities": "İlgilenilen nicelikler: {names}",
    "report.dependence": "Bağımlılık: {dependence}",
    "report.input_law": "Girdi dağılımı: {label} (özet {hash})",
    "report.root_seed": "Kök tohum: {seed}",
    "report.total.forward": "İleri model değerlendirmeleri: {count:,}",
    "report.total.surrogate": "Vekil model değerlendirmeleri: {count:,}",
    "report.failed": (
        "Başarısız değerlendirmeler: {count:,} (denemeler içindeki payı %{share:.3g})"
    ),
    "report.cache_hits": "Önbellek isabetleri (değerlendirme sayılmaz): {count:,}",
    "report.budget": "Değerlendirme bütçesi: {budget:,} (kalan {remaining:,})",
    "report.software": "Yazılım: probstat {version} · Python {python} · {platform}",
    "report.created": "Oluşturulma (UTC): {created}",
    "report.stage.evaluations.forward": "Değerlendirmeler: {count:,}",
    "report.stage.evaluations.surrogate": "Vekil model değerlendirmeleri: {count:,}",
    "report.stage.failed": "başarısız: {count:,}",
    "report.stage.cache_hits": "önbellek isabeti: {count:,}",
    "report.stage.seed": "akış tohumu: {seed}",
    "report.stage.input_law": "girdi dağılımı: {label} ({hash})",
    "report.stage.status": "durum: {status}",
    "study_html.skip_link": "Aşama sonuçlarına geç",
    "study_html.nav_label": "Aşamalar",
    "study_html.nav_item": "{number}. {stage}",
    "study_html.more_stages": {
        "one": "Aşağıda {count} aşama daha var.",
        "other": "Aşağıda {count} aşama daha var.",
    },
    "study_html.figures_omitted": (
        "Şekiller eklenmedi: Matplotlib kurulu değil "
        "(pip install probstat-library[visualization])."
    ),
    "study_html.inputs": "Girdiler: {names}",
    "study_html.quantities": "Nicelikler: {names}",
    "study_html.total": {
        "one": "{count:,} model değerlendirmesi",
        "other": "{count:,} model değerlendirmesi",
    },
    "study_html.failed": {
        "one": "{count:,} başarısız çalıştırma",
        "other": "{count:,} başarısız çalıştırma",
    },
    "study_html.budget": "bütçe {budget:,} (kalan {remaining:,})",
    "study_html.input_law": "Girdi dağılımı: {label} ({hash})",
    "study_html.root_seed": "kök tohum {seed}",
    "study_html.library": "probstat {version}",
    "study_html.python": "Python {version}",
    "study_html.platform": "{platform}",
    "study_html.created": "oluşturulma {created} UTC",
    "study_html.stage.evaluations.forward": {
        "one": "{count:,} model değerlendirmesi",
        "other": "{count:,} model değerlendirmesi",
    },
    "study_html.stage.evaluations.surrogate": {
        "one": "{count:,} vekil model değerlendirmesi",
        "other": "{count:,} vekil model değerlendirmesi",
    },
    "study_html.stage.failed": "{count:,} başarısız",
    "study_html.stage.cache_hits": {
        "one": "{count:,} önbellek isabeti",
        "other": "{count:,} önbellek isabeti",
    },
    "study_html.stage.seed": "akış tohumu {seed}",
    "study_html.stage.input_law": "girdi dağılımı {label} ({hash})",
    "study_html.stage.status": "durum: {status}",
    "study_html.warning": "Uyarı: {message}",
    "study_html.child_title": "Aşama {number} üzerinden vekil model çalışması: {name}",
    "study_html.child_meta": {
        "one": (
            "{count:,} vekil model değerlendirmesi (ileri model değerlendirmesi değil)"
        ),
        "other": (
            "{count:,} vekil model değerlendirmesi (ileri model değerlendirmesi değil)"
        ),
    },
    "study_html.footer.generated": "probstat.workflow.UQStudy ile üretildi.",
    "study_html.footer.reproducible": (
        "Her sayı, kök tohum {seed} ve gösterilen akış tohumlarıyla yeniden "
        "üretilebilir."
    ),
    # Interval limits and parameter lists are separated by "; " because the
    # comma is the Turkish decimal separator.
    "explainer.skip_link": "Açıklamaya geç",
    "explainer.eyebrow": "Etkileşimli açıklama · probstat",
    "explainer.heading.explore": "Keşfet",
    "explainer.heading.method": "Sayılar nasıl hesaplanıyor",
    "explainer.heading.sources": "Denklemler ve kaynaklar",
    "explainer.noscript": (
        "JavaScript kapalı olduğu için denetimler yanıt vermiyor; gösterilen "
        "sayılar varsayılan ayara aittir."
    ),
    "explainer.offline": (
        "Bu sayfadaki her sayı, sayfa yazılmadan önce probstat {version} "
        "tarafından Python'da hesaplandı; sayfa çevrimdışı çalışır ve yalnızca bu "
        "önceden hesaplanmış durumlar arasından seçim yapar."
    ),
    "equation.EQ-LIM-001": "Shevtsova sabitiyle Berry–Esseen sınırı",
    "equation.EQ-LIM-002": "Standartlaştırılmış ortalamanın kesin dağılımları",
    "equation.EQ-LIM-003": "Normal dağılıma Kolmogorov uzaklığı",
    "equation.EQ-LIM-010": "Toplananların üçüncü merkezî momentleri",
    "equation.EQ-REL-DATA-001": "Binom verisinden bir olasılığın Beta sonsal dağılımı",
    "equation.EQ-DIST-BETA-001": "Beta yoğunluğu",
    "equation.EQ-DIST-INV-001": "Dağılım fonksiyonunu tersine çevirerek kantil",
    "equation.EQ-VIZ-LIK-001": "Birim alana ölçeklenmiş binom olabilirliği",
    "equation.EQ-INF-CI-WILSON-001": "Wilson skor aralığı",
    "equation.EQ-MC-MEAN-001": "Monte Carlo örneklem ortalaması kestiricisi",
    "equation.EQ-MC-SE-001": "Kestirilmiş Monte Carlo standart hatası",
    "equation.EQ-MC-ROOTN-001": "Monte Carlo hatasının kök-N ölçeklenmesi",
    "equation.EQ-RNG-SPLITMIX-001": "SplitMix64 ile alt tohum türetme",
    "clt.title": "Bir örneklem ortalaması ne kadar hızlı normalleşir?",
    "clt.lede": (
        "Kaydırıcıyı hareket ettirerek n üstel bekleme süresinin standartlaştırılmış "
        "ortalamasının dağılımı ile normal dağılım arasındaki kesin uzaklığın nasıl "
        "küçüldüğünü ve Berry–Esseen sınırının bunun ne kadar üstünde kaldığını "
        "görün."
    ),
    "clt.slider_label": "Örneklem büyüklüğü n",
    "clt.slider_value": "n = {n:,}",
    "clt.live.distance": (
        "n = {n:,} için standartlaştırılmış ortalama ile normal dağılım arasındaki "
        "kesin Kolmogorov uzaklığı {distance:.3g}."
    ),
    "clt.live.bound": (
        "Berry–Esseen sınırı {bound:.3g}, yani kesin uzaklığın {ratio:.1f} katı."
    ),
    "clt.live.bound_vacuous": (
        "Berry–Esseen sınırı {bound:.3g}; 1'in üstündeki bir sınır, dağılım "
        "fonksiyonları arasındaki uzaklık hakkında bilgi vermez."
    ),
    "clt.chart_a.heading": "Normal yaklaşım nerede yanılıyor",
    "clt.chart_a.title": "Kesin dağılım fonksiyonu eksi normal dağılım fonksiyonu",
    "clt.chart_a.desc": (
        "n = {n:,} için z −4 ile 4 arasındayken Fₙ(z) − Φ(z) çizgi grafiği: eğri "
        "±{distance:.3g} düzeyindeki kesikli çizgilerin arasında kalır ve onlara "
        "z = {z:.2f} yakınında en çok yaklaşır."
    ),
    "clt.chart_a.x": "Standartlaştırılmış değer z",
    "clt.chart_a.y": "Fₙ(z) − Φ(z)",
    "clt.chart_a.label.curve": "Fₙ − Φ",
    "clt.chart_a.label.distance": "±Kₙ",
    "clt.chart_b.heading": "Hata ne kadar hızlı küçülüyor",
    "clt.chart_b.title": "n'ye göre kesin uzaklık ve Berry–Esseen sınırı",
    "clt.chart_b.desc": (
        "n = {first:,} ile {last:,} arası için log–log çizgi grafiği: kesin uzaklık "
        "{k_first:.3g} değerinden {k_last:.3g} değerine, sınır {b_first:.3g} "
        "değerinden {b_last:.3g} değerine iner, ikisi de 1/√n gibi azalır ve "
        "işaretçi n = {n:,} noktasındadır."
    ),
    "clt.chart_b.x": "Örneklem büyüklüğü n (log ölçek)",
    "clt.chart_b.y": "Uzaklık (log ölçek)",
    "clt.chart_b.label.exact": "kesin Kₙ",
    "clt.chart_b.label.bound": "Berry–Esseen sınırı",
    "clt.method.law": (
        "Toplananlar bağımsız Üstel({rate:g}) bekleme süreleridir; bu yüzden n "
        "tanesinin toplamı gama dağılımına uyar ve standartlaştırılmış ortalamanın "
        "dağılım fonksiyonu Fₙ benzetim yapılmadan kesin olarak hesaplanır "
        "[EQ-LIM-002]."
    ),
    "clt.method.distance": (
        "Kolmogorov uzaklığı Kₙ, yani tüm z değerleri üzerindeki en büyük "
        "|Fₙ(z) − Φ(z)| farkı, ince bir ızgarada bulunur ve altın oran aramasıyla "
        "inceltilir [EQ-LIM-003]."
    ),
    "clt.method.bound": (
        "Berry–Esseen sınırı C·ρ/(σ³√n), Shevtsova sabiti C = 0,4748 ve bu "
        "toplananlar için ρ = E|X − μ|³ = {rho:.4f} ile hesaplanır [EQ-LIM-001, "
        "EQ-LIM-010]."
    ),
    "bayes.title": "Veri, bir olasılık hakkındaki inancı nasıl günceller?",
    "bayes.lede": (
        "Beta({a:g}; {b:g}) önsel dağılımı, başarı olasılığı {p:g} olan {trials} "
        "benzetim denemesiyle karşılaşır; denemeleri birer birer eklemek için "
        "kaydırıcıyı hareket ettirin ve Bayesçi güvenilir aralığı frekansçı Wilson "
        "aralığıyla karşılaştırın."
    ),
    "bayes.slider_label": "Gözlenen deneme sayısı",
    "bayes.slider_value": "n = {n}, başarı x = {x}",
    "bayes.mode_label": "Gösterilen aralık",
    "bayes.mode.credible": "%{percent:g} güvenilir aralık",
    "bayes.mode.wilson": "%{percent:g} Wilson aralığı",
    "bayes.live.credible_prior": (
        "Hiç deneme yokken sonsal dağılım önsel Beta({alpha:g}; {beta:g}) "
        "dağılımıdır ve %{percent:g} güvenilir aralığı [{lower:.3f}; {upper:.3f}]."
    ),
    "bayes.live.credible": {
        "one": (
            "1 denemeden sonra (başarı: {successes}) sonsal dağılım "
            "Beta({alpha:g}; {beta:g}), %{percent:g} güvenilir aralığı "
            "[{lower:.3f}; {upper:.3f}] ve ortalaması {mean:.3f}."
        ),
        "other": (
            "{count} denemeden sonra (başarı: {successes}) sonsal dağılım "
            "Beta({alpha:g}; {beta:g}), %{percent:g} güvenilir aralığı "
            "[{lower:.3f}; {upper:.3f}] ve ortalaması {mean:.3f}."
        ),
    },
    "bayes.live.wilson": {
        "one": (
            "1 denemeden sonra (başarı: {successes}) %{percent:g} Wilson aralığı, "
            "gözlenen oran {share:.3f} çevresinde [{lower:.3f}; {upper:.3f}]."
        ),
        "other": (
            "{count} denemeden sonra (başarı: {successes}) %{percent:g} Wilson "
            "aralığı, gözlenen oran {share:.3f} çevresinde "
            "[{lower:.3f}; {upper:.3f}]."
        ),
    },
    "bayes.live.wilson_none": (
        "Hiç deneme yokken Wilson aralığı yoktur, çünkü en az bir deneme gerektirir."
    ),
    "bayes.chart.heading": "Önsel, olabilirlik ve sonsal",
    "bayes.chart.title": "Başarı olasılığı p için yoğunluklar",
    "bayes.chart.desc_credible": {
        "one": (
            "1 denemeden sonra p'nin 0 ile 1 arasındaki önsel, olabilirlik ve "
            "sonsal yoğunluklarının çizgi grafiği: sonsal p = {mode:.2f} yakınında "
            "tepe yapar, gölgeli bant %{percent:g} güvenilir aralığı "
            "[{lower:.3f}; {upper:.3f}] gösterir ve kesikli çizgi gerçek "
            "p = {p:g} değerini işaretler."
        ),
        "other": (
            "{count} denemeden sonra p'nin 0 ile 1 arasındaki önsel, olabilirlik ve "
            "sonsal yoğunluklarının çizgi grafiği: sonsal p = {mode:.2f} yakınında "
            "tepe yapar, gölgeli bant %{percent:g} güvenilir aralığı "
            "[{lower:.3f}; {upper:.3f}] gösterir ve kesikli çizgi gerçek "
            "p = {p:g} değerini işaretler."
        ),
    },
    "bayes.chart.desc_wilson": {
        "one": (
            "1 denemeden sonra p'nin 0 ile 1 arasındaki önsel, olabilirlik ve "
            "sonsal yoğunluklarının çizgi grafiği: sonsal p = {mode:.2f} yakınında "
            "tepe yapar, gölgeli bant %{percent:g} Wilson aralığını "
            "[{lower:.3f}; {upper:.3f}] gösterir ve kesikli çizgi gerçek "
            "p = {p:g} değerini işaretler."
        ),
        "other": (
            "{count} denemeden sonra p'nin 0 ile 1 arasındaki önsel, olabilirlik ve "
            "sonsal yoğunluklarının çizgi grafiği: sonsal p = {mode:.2f} yakınında "
            "tepe yapar, gölgeli bant %{percent:g} Wilson aralığını "
            "[{lower:.3f}; {upper:.3f}] gösterir ve kesikli çizgi gerçek "
            "p = {p:g} değerini işaretler."
        ),
    },
    "bayes.chart.desc_none": (
        "Hiç deneme yokken p'nin 0 ile 1 arasındaki önsel, olabilirlik ve sonsal "
        "yoğunluklarının çizgi grafiği: sonsal önsele eşittir ve p = {mode:.2f} "
        "yakınında tepe yapar, hiçbir aralık gölgelenmez ve kesikli çizgi gerçek "
        "p = {p:g} değerini işaretler."
    ),
    "bayes.chart.x": "Başarı olasılığı p",
    "bayes.chart.y": "Yoğunluk",
    "bayes.key.prior": "önsel Beta({a:g}; {b:g})",
    "bayes.key.likelihood": "olabilirlik, birim alana ölçeklenmiş",
    "bayes.key.posterior": "sonsal",
    "bayes.key.truth": "gerçek p = {p:g}",
    "bayes.method.trials": (
        "Her deneme, kütüphanenin tohumlanmış rastgele akışlarından (kök tohum "
        "{seed}) çekilen ve başarı olasılığı {p:g} olan bir Bernoulli denemesidir "
        "[EQ-RNG-SPLITMIX-001]."
    ),
    "bayes.method.posterior": (
        "Beta(a; b) önseli ve n denemede x başarıyla sonsal dağılım "
        "Beta(a + x; b + n − x) olur [EQ-REL-DATA-001] ve eşit kuyruklu güvenilir "
        "aralığı Beta kantil fonksiyonundan gelir [EQ-DIST-BETA-001, "
        "EQ-DIST-INV-001]."
    ),
    "bayes.method.likelihood": (
        "Olabilirlik pˣ(1 − p)ⁿ⁻ˣ birim alana ölçeklenerek çizilir; bu, "
        "Beta(x + 1; n − x + 1) yoğunluğudur [EQ-VIZ-LIK-001]."
    ),
    "bayes.method.wilson": (
        "Wilson skor aralığı, aynı oran için önseli kullanmayan frekansçı bir "
        "aralıktır [EQ-INF-CI-WILSON-001]."
    ),
    "mc.title": "Monte Carlo hatası çekiliş sayısıyla nasıl küçülür?",
    "mc.lede": (
        "Birim karede düzgün dağılmış bir nokta çeyrek dairenin içine π/4 "
        "olasılıkla düşer, bu yüzden isabet oranının dört katı π'yi kestirir; "
        "çekiliş eklemek için kaydırıcıyı hareket ettirin ve bir koşuyu ve onun "
        "%{percent:g} güven bandını izlemek için bir tohum seçin."
    ),
    "mc.slider_label": "Çekiliş sayısı N",
    "mc.slider_value": "N = {n:,}",
    "mc.mode_label": "Rastgele tohum",
    "mc.mode.run": "Tohum {run}",
    "mc.live.covers": (
        "N = {n:,} çekilişle tohum {run}, π'yi {estimate:.4f} ± {half:.4f} olarak "
        "kestirir ve %{percent:g} aralığı [{lower:.4f}; {upper:.4f}] π'yi içerir."
    ),
    "mc.live.misses": (
        "N = {n:,} çekilişle tohum {run}, π'yi {estimate:.4f} ± {half:.4f} olarak "
        "kestirir ve %{percent:g} aralığı [{lower:.4f}; {upper:.4f}] π'yi içermez."
    ),
    "mc.live.spread": {
        "one": "Yalnızca bir tohum olduğu için tohumlar arası yayılım gösterilmez.",
        "other": (
            "{count} tohumda kestirimler {low:.4f} ile {high:.4f} arasında değişir."
        ),
    },
    "mc.chart.heading": "N'ye göre kestirim ve güven bandı",
    "mc.chart.title": "π'nin Monte Carlo kestirimleri",
    "mc.chart.desc": (
        "N = {first:,} ile {last:,} çekiliş arasında logaritmik eksenli, her tohum "
        "için bir çizgili çizgi grafiği: tohum {run}, ±{half_first:.3g} "
        "değerinden ±{half_last:.3g} değerine daralan %{percent:g} güven "
        "bandıyla vurgulanır, kesikli çizgi π'yi işaretler ve işaretçi N = {n:,} "
        "noktasındadır."
    ),
    "mc.chart.x": "Çekiliş sayısı N (log ölçek)",
    "mc.chart.y": "π kestirimi",
    "mc.label.pi": "π = {pi:.4f}",
    "mc.label.run": "tohum {run}",
    "mc.key.selected": "seçili tohum ve %{percent:g} bandı",
    "mc.key.others": "diğer tohumlar",
    "mc.key.pi": "kesin π",
    "mc.method.streams": (
        "Tohum k, noktalarını kök tohum {seed} değerinin k numaralı alt akışından "
        "çeker; bu yüzden koşular bağımsız ve yeniden üretilebilirdir "
        "[EQ-RNG-SPLITMIX-001]."
    ),
    "mc.method.estimate": (
        "Her kestirim 4·I(U₁² + U₂² ≤ 1) katkılarının ortalamasıdır "
        "[EQ-MC-MEAN-001]; standart hatası s/√N [EQ-MC-SE-001] ve aralığı normal "
        "yaklaşımla bulunur."
    ),
    "mc.method.rate": (
        "Aralığın yarı genişliği 1/√N gibi azalır, yani çekilişleri dört katına "
        "çıkarmak onu yarıya indirir [EQ-MC-ROOTN-001]."
    ),
}
