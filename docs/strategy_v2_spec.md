# 戦略 v2 仕様(構造再設計) — 事前登録

作成: 2026-07-22。v1の15mゲート不合格を受けた構造再設計。結果を見る前に本ゲートを固定する。

## v1からの診断結果(根拠)

- 1h ttm_short: Sharpe 1.87, rand_p 0.045 — ただし利益は2026年1月に集中、トレード21件で検定力不足
- 1h ttm_long: 24件中20件負け。同一足EMA200はベアラリー中にロングを誤許可(フィルターとして速すぎる)
- 15m: 全構成で失敗 — 短い時間軸ではスクイーズ解放後の値動きがノイズ+コストに埋没
- 教訓: ①より遅い時間軸が優位 ②トレンドフィルターは上位足で ③ブレイクの質の確認が必要

## v2 構造変更(v1 ttmベースからの差分)

1. **HTFトレンドフィルター(F1, 常時ON)**: 1h終値を日次(UTC)に集計→日足EMA50を計算→
   完結した直近日の値を1hバーへフォワードフィル(ルックアヘッド禁止)。
   long: close > dailyEMA50, short: close < dailyEMA50。同一足EMA200(trend_ema)は廃止。
2. **TTMモメンタム確認(F2, ON/OFF変数)**: midline = ((HH20+LL20)/2 + EMA20)/2、
   delta[t] = close[t] − midline[t]、osc[t] = 20バー線形回帰の終端値(=回帰直線のt時点の値)。
   long: osc > 0、short: osc < 0(シグナルバーで判定)。
3. **出来高確認(F3, ON/OFF変数)**: ブレイクバーの volume ≥ 1.3 × SMA(volume, 20)。
4. base変更: min_squeeze_bars 6→4(フィルター追加による標本減の補償、全変種一律)。
   その他は v1 ttm と同一(bb 20/2.0, kc_mult 1.5, release_window 3, atr 14, stop 2.0, trail 3.0,
   コスト同一, リスク3%, レバ上限5x)。

## 事前登録する v2 変種(1h主・2h確認の各8通り)

direction ∈ {short, both} × F2 ∈ {on, off} × F3 ∈ {on, off}(F1は全変種ON)

2h足は1hバーの集約(2本→1本、UTC偶数時起点)で生成。パラメータのバー数は不変。

## 事前登録ゲート(v2)

1hの8変種のうち、以下を全て満たす変種が1つ以上あれば最適化フェーズへ進む:
- trades ≥ 30
- bootstrap p < 0.05(トレードRの平均>0)
- random-entry p < 0.05
- **多重比較ガード**(v1から累計約20構成を検査しているため):
  rand_p < 0.02、または同一変種を2hに適用して Sharpe > 0.5(符号再現)

不合格の場合は構造仮説自体を棄却し、ユーザー要求(BBスクイーズ基盤)の範囲内で
別構造(例: リテスト型エントリー、レンジブレイク併用)を検討する。

## 実装メモ

- indicators.py 追加: rolling_max/min(n), linreg_endpoint(series, n), 日次集計EMAのFFill
- strategy.py: params拡張 {"htf_trend": true, "momentum_filter": bool, "volume_filter": bool,
  "vol_mult": 1.3, "trend_ema": 0(廃止)}
- scripts/run_v2.py: 8変種×{1h, 2h}を一括実行、有意性テスト付き、results/v2_{tf}.json
- 2h集約はローダー側 resample(bars, 2) で実装(端数バーは切捨て)
