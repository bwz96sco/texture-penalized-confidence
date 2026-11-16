"""Post hoc readouts on retained texture-conf outputs for the GRSL letter (read-only).

Run from the repository root:
    uv run --no-sync python -I experiments/texture-conf-paper-readouts/readouts.py

Reads retained evidence files only and writes tasks/texture-conf-paper-readouts/evidence/readouts.json.
No imagery, no model, no texture computation, no new arm, no interval. Every readout first rests on a
reproduction of retained counts (R0); compute() asserts those reproductions.
"""
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / 'tasks'
U_EV = T / 'texture-s2-confirmation/evidence'
S_EV1 = T / 's2-independent-confirmation/evidence'
S_EV2 = T / 'texture-s2-supplementary/evidence'
P_EV = T / 'texture-permutation-premise/evidence'
OUT = T / 'texture-conf-paper-readouts/evidence/readouts.json'
COVERAGES = (0.5, 0.6, 0.7, 0.8, 0.9)
ARMS = ('confidence', 'ensemble', 'zero-label', 'veto-zl', 'texture-conf')
COUNT_KEYS = ('correct_cloud', 'false_cloud', 'correct_clear', 'missed_cloud', 'n_accepted_cloud_calls')


def r4(x):
    return round(x, 6)  # six decimals, so that values reported at three decimals are not rounded twice


def auroc(scores, pos):
    """Mann-Whitney AUROC with average ranks for ties; pos marks the positive (wrong) class."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    npos = sum(pos)
    nneg = len(pos) - npos
    return (sum(r for r, p in zip(ranks, pos) if p) - npos * (npos + 1) / 2) / (npos * nneg)


def load_untouched():
    fz = json.loads((U_EV / 'freeze.json').read_text())
    mp, sc = fz['cloud_call_mapping'], fz['scale']
    mu, sd, a, b = mp['mean'][0], mp['scale'][0], mp['coef'][0], mp['intercept']
    lab = {r['key']: (float(r['thick_frac']), float(r['thin_frac'])) for r in csv.DictReader(open(U_EV / 'labels.csv'))}
    rows = []
    for r in csv.DictReader(open(U_EV / 'features.csv')):
        conf, cf, sat = float(r['conf']), float(r['pred_cloud_frac']), float(r['sat_frac'])
        main = cf >= 0.5
        q = float(r['q_multiscale']) if main else None
        pcc = 1 / (1 + math.exp(-(a * (q - mu) / sd + b))) if main else 0.0
        th, tn = lab[r['key']]
        satf = sat > 0.1 and not main
        zl = 0.5 * (conf - sc['mean'][0]) / sc['sd'][0] - 0.5 * (cf - sc['mean'][1]) / sc['sd'][1]
        flag = main and pcc > 0.5
        rows.append(dict(unit=r['unit'], tile=int(r['tile']), cluster=r['cluster'], conf=conf, cf=cf, main=main,
                         q=q, pcc=pcc, th=th, tn=tn, A=th >= 0.5, B=th + tn >= 0.5, flag=flag,
                         s={'confidence': conf, 'ensemble': float(r['ens']), 'zero-label': zl,
                            'veto-zl': zl - 1e3 * flag, 'texture-conf': conf - 1e6 * satf - pcc},
                         s_within=conf - 1e6 * satf))
    return rows


def order_by(rows, arm):
    return sorted(range(len(rows)), key=lambda j: -rows[j]['s'][arm])  # stable: ties keep stored order


def counts(rows, acc, conv):
    c = dict(correct_cloud=0, false_cloud=0, correct_clear=0, missed_cloud=0)
    for j in acc:
        r = rows[j]
        if r['main']:
            c['correct_cloud' if r[conv] else 'false_cloud'] += 1
        else:
            c['missed_cloud' if r[conv] else 'correct_clear'] += 1
    c['n_accepted_cloud_calls'] = c['correct_cloud'] + c['false_cloud']
    return c


def r0_untouched(rows, retained):
    n = len(rows)
    k = math.ceil(0.8 * n)
    flags = sum(r['flag'] for r in rows)
    assert flags <= 0.2 * n  # the veto budget fallback does not apply
    acc = {arm: set(order_by(rows, arm)[:k]) for arm in ARMS}
    tc_cc = sum(rows[j]['main'] for j in acc['texture-conf'])
    clear = sorted((j for j in range(n) if not rows[j]['main']), key=lambda j: -rows[j]['s_within'])
    cloud = sorted((j for j in range(n) if rows[j]['main']), key=lambda j: -rows[j]['s_within'])
    acc['matched-control'] = set(clear[:k - tc_cc]) | set(cloud[:tc_cc])
    for conv, setting in (('A', 'cloudsen_A'), ('B', 'cloudsen_B')):
        for arm, a in acc.items():
            got = counts(rows, a, conv)
            want = {key: retained['settings'][setting]['arms'][arm][key] for key in COUNT_KEYS}
            assert got == want, (conv, arm, got, want)
    tc_scores = sorted((rows[j]['s']['texture-conf'] for j in range(n)), reverse=True)
    return acc, dict(
        n_tiles=n, n_accepted=k, n_cloud_calls=len(cloud), n_clear_calls=len(clear),
        arms_reproduced=sorted(acc), veto_flags=flags,
        texture_conf_clear_calls_accepted=sum(1 for j in clear if j in acc['texture-conf']),
        texture_conf_lowest_clear_score=r4(min(rows[j]['s']['texture-conf'] for j in clear)),
        texture_conf_cut_score=r4(tc_scores[k - 1]))


def r0_test_split():
    retained = json.loads((S_EV2 / 'results.json').read_text())
    lab = {r['key']: (float(r['thick_frac']), float(r['thin_frac']), r['cluster'])
           for r in csv.DictReader(open(S_EV1 / 'labels.csv'))}
    tc = {r['key']: int(r['accepted_texture_conf']) for r in csv.DictReader(open(S_EV2 / 'features.csv'))}
    conf, main = {}, {}
    for r in csv.DictReader(open(S_EV1 / 'decisions.csv')):
        if r['setting'] == 'cloudsen_A':
            conf[r['key']], main[r['key']] = int(r['confidence']), int(r['main'])
    assert len(lab) == len(tc) == len(conf) == 40425
    for mask, arm in ((conf, 'confidence'), (tc, 'texture-conf')):
        for conv, setting in (('A', 'cloudsen_A'), ('B', 'cloudsen_B')):
            got = dict(correct_cloud=0, false_cloud=0, correct_clear=0, missed_cloud=0)
            for key, accepted in mask.items():
                if not accepted:
                    continue
                th, tn, _ = lab[key]
                y = th >= 0.5 if conv == 'A' else th + tn >= 0.5
                if main[key]:
                    got['correct_cloud' if y else 'false_cloud'] += 1
                else:
                    got['missed_cloud' if y else 'correct_clear'] += 1
            got['n_accepted_cloud_calls'] = got['correct_cloud'] + got['false_cloud']
            want = {key: retained['settings'][setting]['arms'][arm][key] for key in COUNT_KEYS}
            assert got == want, (arm, conv, got, want)
    n_clear = sum(1 for key in main if not main[key])
    return lab, conf, tc, main, dict(
        n_tiles=len(lab), n_clear_calls=n_clear,
        texture_conf_clear_calls_accepted=sum(1 for key in main if not main[key] and tc[key]))


def kind(th, tn):
    if th >= 0.5:
        return 'cloud_under_A'
    if th + tn >= 0.5:
        return 'thin_cloud_tile'
    if th + tn == 0:
        return 'no_labelled_cloud'
    return 'partly_cloudy_below_0.5'


KINDS = ('no_labelled_cloud', 'partly_cloudy_below_0.5', 'thin_cloud_tile', 'cloud_under_A')


def r1(u_retained, lab, conf, tc, main):
    out = {}
    s = {c: u_retained['settings'][f'cloudsen_{c}']['arms'] for c in 'AB'}
    for base in ('confidence', 'matched-control'):
        fa = s['A'][base]['false_cloud'] - s['A']['texture-conf']['false_cloud']
        fb = s['B'][base]['false_cloud'] - s['B']['texture-conf']['false_cloud']
        out[f'untouched_net_false_clouds_removed_vs_{base}'] = dict(
            A=fa, B=fb, share_also_false_under_B=r4(fb / fa))
    groups = {'removed': lambda key: main[key] and conf[key] and not tc[key],
              'kept_by_texture_conf': lambda key: main[key] and tc[key]}
    comp = {}
    for g, f in groups.items():
        c = Counter(kind(*lab[key][:2]) for key in lab if f(key))
        comp[g] = {k: c[k] for k in KINDS}
        false_a = sum(c[k] for k in KINDS[:3])
        comp[g]['false_under_A'] = false_a
        comp[g]['shares_of_false_under_A'] = {k: r4(c[k] / false_a) for k in KINDS[:3]}
        near = sum(1 for key in lab if f(key) and 0.4 <= lab[key][0] < 0.5)
        comp[g]['near_threshold_thick_0.4_to_0.5'] = near
        comp[g]['near_threshold_share_of_false_under_A'] = r4(near / false_a)
    rois = Counter(lab[key][2] for key in lab if groups['removed'](key) and lab[key][0] < 0.5)
    top = sorted(rois.values(), reverse=True)
    comp['removed_false_under_A_rois'] = dict(n_rois=len(rois), of=165, top10_share=r4(sum(top[:10]) / sum(top)))
    out['test_split_per_tile'] = comp
    return out


def class_rates(items):
    """items: (pred_cloud_frac, thick, thin). Returns error rates of cloud calls and clear calls."""
    cc = [x for x in items if x[0] >= 0.5]
    cl = [x for x in items if x[0] < 0.5]
    res = dict(n=len(items), n_cloud_calls=len(cc))
    for conv, y in (('A', lambda x: x[1] >= 0.5), ('B', lambda x: x[1] + x[2] >= 0.5)):
        res[conv] = dict(cloud_calls_wrong=r4(sum(not y(x) for x in cc) / len(cc)),
                         clear_calls_wrong=r4(sum(y(x) for x in cl) / len(cl)))
    return res


def blocks(tiles):
    """tiles: unit -> {tile: (p, th, tn)}; 3x3 blocks on rows/cols {0,1,2},{3,4,5} of the 7x7 grid."""
    out = []
    for unit, t in tiles.items():
        assert len(t) == 49
        for bi in (0, 1):
            for bj in (0, 1):
                ids = [(3 * bi + i) * 7 + (3 * bj + j) for i in range(3) for j in range(3)]
                out.append(tuple(sum(t[k][m] for k in ids) / 9 for m in range(3)))
    return out


def r2(rows):
    out = {}
    cloud = [r for r in rows if r['main']]
    for conv in 'AB':
        wrong = [not r[conv] for r in cloud]
        out[f'untouched_auroc_{conv}'] = dict(
            n_cloud_calls=len(cloud), n_wrong=sum(wrong),
            q_multiscale=r4(auroc([r['q'] for r in cloud], wrong)),
            negative_confidence=r4(auroc([-r['conf'] for r in cloud], wrong)))
    cols = ('q_multi1', 'q_multi3', 'q_multi9', 'q_multiscale')
    src = [r for r in csv.DictReader(open(P_EV / 'source-features.csv')) if float(r['pred_cloud_frac']) >= 0.5]
    err = [r['correct'] == '0' for r in src]
    assert (len(src), sum(err)) == (1034, 341)
    dev_src = {c: r4(auroc([float(r[c]) for r in src], err)) for c in cols}
    s2 = [r for r in csv.DictReader(open(P_EV / 'tiles.csv'))
          if r['dataset'] == 'cloudsen' and float(r['pred_cloud_frac']) >= 0.5]
    ea = [float(r['label_cloud_frac']) < 0.5 for r in s2]
    eb = [float(r['label_cloud_frac']) + float(r['thin_frac']) < 0.5 for r in s2]
    dev_s2 = {c: dict(A=r4(auroc([float(r[c]) for r in s2], ea)), B=r4(auroc([float(r[c]) for r in s2], eb)))
              for c in cols}
    assert round(dev_src['q_multiscale'], 3) == 0.849
    assert (round(dev_s2['q_multiscale']['A'], 3), round(dev_s2['q_multiscale']['B'], 3)) == (0.946, 0.833)
    out['development_auroc_by_radius'] = dict(
        note='development data, where the feature was chosen; radius r px is 30r m on Landsat 8, 10r m on Sentinel-2',
        source_cloud_calls=dict(n=len(src), **dev_src), sentinel2_cloud_calls=dict(n=len(s2), **dev_s2))
    u_tiles = defaultdict(dict)
    for r in rows:
        u_tiles[r['unit']][r['tile']] = (r['cf'], r['th'], r['tn'])
    lab = {r['key']: (float(r['thick_frac']), float(r['thin_frac'])) for r in csv.DictReader(open(S_EV1 / 'labels.csv'))}
    s_tiles = defaultdict(dict)
    for r in csv.DictReader(open(S_EV1 / 'features-single-1.csv')):
        s_tiles[r['unit']][int(r['tile'])] = (float(r['pred_cloud_frac']),) + lab[r['key']]
    out['class_error_rates_all_tiles'] = dict(
        untouched_64px_tiles=class_rates([v for t in u_tiles.values() for v in t.values()]),
        untouched_1p92km_blocks=class_rates(blocks(u_tiles)),
        test_split_64px_tiles=class_rates([v for t in s_tiles.values() for v in t.values()]),
        test_split_1p92km_blocks=class_rates(blocks(s_tiles)))
    return out


def r3(rows):
    cloud = sorted((r for r in rows if r['main']), key=lambda r: r['pcc'])
    n = len(cloud)
    bins = []
    for d in range(10):
        seg = cloud[d * n // 10:(d + 1) * n // 10]
        bins.append(dict(decile=d + 1, n=len(seg), mean_pcc=r4(sum(r['pcc'] for r in seg) / len(seg)),
                         wrong_A=r4(sum(not r['A'] for r in seg) / len(seg)),
                         wrong_B=r4(sum(not r['B'] for r in seg) / len(seg))))
    out = dict(n_cloud_calls=n, mean_pcc=r4(sum(r['pcc'] for r in cloud) / n),
               wrong_share_A=r4(sum(not r['A'] for r in cloud) / n),
               wrong_share_B=r4(sum(not r['B'] for r in cloud) / n), deciles=bins)
    for conv in 'AB':
        out[f'ece10_{conv}'] = r4(sum(b['n'] * abs(b['mean_pcc'] - b[f'wrong_{conv}']) for b in bins) / n)
    return out


def r5(rows):
    """Cloud- and clear-call wrong shares within deciles of confidence over all untouched tiles (R5, 10 November)."""
    tiles = sorted(rows, key=lambda r: r['conf'])  # stable: ties keep stored order
    n = len(tiles)
    bins = []
    for d in range(10):
        seg = tiles[d * n // 10:(d + 1) * n // 10]
        b = dict(decile=d + 1, n=len(seg), conf_min=r4(seg[0]['conf']), conf_max=r4(seg[-1]['conf']))
        for side, is_cloud in (('cloud', True), ('clear', False)):
            calls = [r for r in seg if r['main'] == is_cloud]
            b[f'n_{side}_calls'] = len(calls)
            for conv in 'AB':
                b[f'{side}_wrong_{conv}'] = r4(sum(r[conv] != is_cloud for r in calls) / len(calls)) if calls else None
        bins.append(b)
    out = dict(n_tiles=n, deciles=bins)
    for conv in 'AB':
        out[f'cloud_above_clear_in_every_decile_{conv}'] = all(
            b[f'cloud_wrong_{conv}'] is not None and b[f'clear_wrong_{conv}'] is not None
            and b[f'cloud_wrong_{conv}'] > b[f'clear_wrong_{conv}'] for b in bins)
    return out


def r4_coverage(rows, retained):
    n = len(rows)
    out = {}
    for arm in ARMS:
        order = order_by(rows, arm)
        for conv in 'AB':
            cum, area, risk = 0, 0.0, {}
            for m, j in enumerate(order, 1):
                cum += rows[j]['main'] != rows[j][conv]
                area += cum / m
                for c in COVERAGES:
                    if m == math.ceil(c * n):
                        risk[str(c)] = r4(100 * cum / m)
            want = retained['settings'][f'cloudsen_{conv}']['arms'][arm]['n_accepted_wrong']
            assert round(risk['0.8'] * math.ceil(0.8 * n) / 100) == want, (arm, conv)
            out[f'{arm}_{conv}'] = dict(risk_percent_at=risk, aurc_percent=r4(100 * area / n))
    for conv in 'AB':
        out[f'order_holds_at_every_listed_coverage_{conv}'] = all(
            out[f'texture-conf_{conv}']['risk_percent_at'][c] < min(
                out[f'{g}_{conv}']['risk_percent_at'][c] for g in ('confidence', 'ensemble', 'zero-label'))
            for c in map(str, COVERAGES))
    return out


def compute():
    u_retained = json.loads((U_EV / 'results.json').read_text())
    rows = load_untouched()
    _, r0u = r0_untouched(rows, u_retained)
    lab, conf, tc, main, r0s = r0_test_split()
    return dict(
        task='texture-conf-paper-readouts',
        scope='post hoc, descriptive readouts on retained outputs of the frozen runs; no selection, arm, interval or imagery',
        R0_reproduction=dict(untouched=r0u, test_split=r0s),
        R1_removed_false_clouds=r1(u_retained, lab, conf, tc, main),
        R2_texture_separation_and_footprint=r2(rows),
        R3_pcc_reliability_untouched=r3(rows),
        R4_risk_away_from_80_untouched=r4_coverage(rows, u_retained),
        R5_class_reliability_at_equal_confidence_untouched=r5(rows))


if __name__ == '__main__':
    res = compute()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))
    sys.exit(0)
