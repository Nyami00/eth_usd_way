# 実装仕様 v1 — プロトタイプ・バックテスト & 統計的有意性ゲート

設計: Fable 5 / 実装: Opus 4.8。純Python標準ライブラリのみ(numpy/pandas不可)。

## モジュール構成

```
backtest/
  __init__.py
  data.py        # CSVロード
  indicators.py  # SMA/EMA/STDDEV/ATR/BB/Keltner/BandWidthパーセンタイル
  engine.py      # イベントループ型シミュレータ(1ポジション、リスクサイジング)
  strategy.py    # スクイーズ・ブレイクアウト戦略のシグナル生成
  metrics.py     # Sharpe(日次集計・年率)、DD、R統計、トレード集計
  stats.py       # ブートストラップ検定、ランダムエントリー検定
scripts/
  run_prototype.py  # v1固定パラメータで実行し、結果とp値をJSON+標準出力
```

## データ仕様
- `data/eth_usd_1h.csv` (timestamp,open,high,low,close,volume; ISO8601Z; 昇順)
- ウォームアップ: 2025-10-01以降のデータで指標計算。**成績評価は 2026-01-01T00:00:00Z 以降のバーのみ**
  (エクイティ計測・トレード計上とも評価期間内に限定。評価開始前にオープンしたポジションは持たない
  — エントリー判定自体を2026-01-01以降に限定してよい)

## 指標定義(indicators.py)
- SMA(n), 標本標準偏差 stdev(n, ddof=0), EMA(n, 初期値=最初のn本のSMA)
- TR = max(h−l, |h−prev_c|, |l−prev_c|), ATR(n) = Wilder平滑 (初期値=最初のn本TRの単純平均)
- BB(n=20, k=2.0): mid=SMA20, upper=mid+2σ, lower=mid−2σ
- Keltner(n=20, m): mid=EMA20, upper=mid+m×ATR20, lower=mid−m×ATR20
- BandWidth = (BBupper−BBlower)/BBmid
- bw_pctile(L): 直近Lバー(当バー含む)内でのBandWidthの百分位順位 [0,100]
- 全指標は「バーtの値=バーt確定時点で既知の値」。未確定バーの値を先読みしない。

## 戦略ロジック(strategy.py) — パラメータはdictで渡す
```
params = {
  "bb_n": 20, "bb_k": 2.0,
  "squeeze_mode": "pctile",   # "pctile" | "ttm"
  "bw_lookback": 120, "bw_q": 20.0,   # pctileモード: bw_pctile < bw_q でスクイーズ
  "kc_mult": 1.5,                      # ttmモード: BB両端がKC内側でスクイーズ
  "min_squeeze_bars": 6,      # 直近でスクイーズがSバー連続していた事
  "release_window": 3,        # スクイーズ解除からこのバー数以内のブレイクのみ有効
  "trend_ema": 200,           # 0なら無効。long: close>EMA, short: close<EMA
  "allow_long": true, "allow_short": true,
  "atr_n": 14, "stop_atr": 2.0, "trail_atr": 3.0,
  "flip_on_opposite": false,
}
```
- 「スクイーズ状態」: squeeze_mode に従い判定
- 「セットアップ有効」: バーtまでに min_squeeze_bars 連続でスクイーズが継続、その後
  スクイーズが解除された時点から release_window バー以内
- エントリーシグナル(バーt確定時):
  - long: セットアップ有効 かつ close[t] > BBupper[t] かつ (trend_ema=0 or close[t]>EMA[t])
  - short: 対称
- 同一スクイーズからは最初のシグナルのみ(発火したらそのセットアップは消費)

## 執行・ポジションモデル(engine.py)
- 資本初期値 100,000 USD。同時ポジションは最大1。
- シグナルバーtの翌バーt+1の始値で成行約定。買い: fill=open×(1+slip)、売り: fill=open×(1−slip)。
  手数料 fee_rate×約定金額を都度控除。既定: fee=0.0005, slip=0.0002。
- 初期ストップ: シグナルバーtのATRで entry∓stop_atr×ATR[t]。
- サイズ: risk_amount = equity×0.03(equity=約定直前の口座評価額)。
  qty = risk_amount / |entry_est − stop|(entry_estはシグナルバー終値で近似してよい)。
  名目上限: qty×entry ≤ equity×5(レバレッジ上限5x)。qty>0でなければ見送り。
- トレーリング(チャンデリア): ポジション保有中、long: stop=max(stop, 最高値(high)since entry − trail_atr×ATR[current])、shortは対称。ストップは不利方向に動かさない。
- ストップ執行: バー内で long: low ≤ stop → 約定価格 = min(open, stop)×(1−slip)(ギャップ考慮の保守的モデル)。shortは対称。エントリーと同一バーでのストップ判定も行う(t+1バー)。
- flip_on_opposite=trueなら反対シグナルで手仕舞い+新規(v1はfalse)。
- 各バーでmark-to-marketしたequity系列を記録。
- トレード記録: side, entry_ts/px, exit_ts/px, qty, bars_held, pnl, r_multiple
  (r_multiple = pnl / (初期リスク額) ; 初期リスク額 = qty×|entry_fill−initial_stop| … 手数料込みのpnlを使用)

## 評価指標(metrics.py)
- equity系列(1hバー)→ UTC日次終値でリサンプル → 日次リターン系列
- **Sharpe(主指標) = mean(daily_r)/stdev(daily_r, ddof=1)×sqrt(365)**(rf=0; stdev=0なら0)
- MaxDD(equity系列ベース, %)、総リターン、CAGR(365日換算)、勝率、平均R、
  profit factor、トレード数、平均保有バー、エクスポージャー(ポジション保有バー比率)

## 統計検定(stats.py) — 乱数は random.Random(seed) で再現可能に
1. `bootstrap_pvalue(r_multiples, n=10000, seed=42)`
   → 復元抽出で平均Rの分布を作り、p = (#{mean≤0}+1)/(n+1)
2. `random_entry_test(bars, eval_start, trades, params, n=1000, seed=42)`
   → 実戦略のトレードから (保有バー数リスト, 方向リスト) を抽出。各試行で:
     評価期間内のランダムなバーで順次エントリー(重複保有なし、同一サイジング3%、
     初期ストップ=同じstop_atr×ATR、トレーリングなし・保有バー数経過で翌バー始値手仕舞い
     ※ただしストップに掛かればストップ)、方向は実戦略の方向比率でランダム。
     手数料・スリッページ同一。日次Sharpeを計算。
   → p = (#{random_sharpe ≥ actual_sharpe}+1)/(n+1)
   ※ ストップあり版にするのは実戦略とのリスク構造を揃えるため。
3. 出力: 両p値、実Sharpe、ランダム分布の平均/95%点、トレード数

## run_prototype.py
- 1hデータで v1既定パラメータ(上記params初期値)を実行
- 標準出力: 主要メトリクス表、トレード数、検定p値
- `results/prototype_v1.json` に全結果を保存
- 追加で「B案(pctile)とA案(ttm)」両方、および long/short 許可の組合せ
  {both, long-only, short-only} × {pctile, ttm} = 6通りを一括実行して比較表を出す
  (これは最適化ではなく、プロトタイプの構造選択のための比較)

## 品質要件
- ルックアヘッドバイアス禁止(シグナルはバー確定情報のみ、約定は翌バー)
- ゼロ除算・空系列のガード
- `python3 -m py_compile` 通過、簡単なself-test(合成データでBB/ATR手計算一致)を
  `tests/test_indicators.py` に(unittest, 実行して緑であること)
```
