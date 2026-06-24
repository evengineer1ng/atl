import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime, timedelta
from freqtrade.strategy import IStrategy
from pandas import DataFrame
from freqtrade.persistence import Trade

class Maverick(IStrategy):
    """
    Maverick: cross-sectional relative-outlier strategy for a Top 10 universe.
    Ranks all coins every 15m by composite momentum (15m, 1h, 4h).
    Enters long on the strongest positive outliers and short on the weakest
    negative outliers when divergence is accelerating.
    Revision: adds volume / volatility / adverse-swing confirmation, regime filter,
    wider stops, faster time-stops, and richer ROI/trailing tiers to improve payoff.
    """
    INTERFACE_VERSION = 3

    can_short: bool = True
    timeframe = '15m'
    process_only_new_candles = True
    startup_candle_count = 50

    # --- ENTRY GENOME PARAMETERS ---
    zscore_entry = 1.0
    min_dispersion = 0.003
    entry_rank_top_n = 3
    pair_cooldown_minutes = 45
    volume_factor = 1.3
    atr_multiplier = 0.7
    regime_bear_threshold = -0.002

    # --- EXIT GENOME PARAMETERS ---
    stoploss = -0.025
    trailing_stop = True
    trailing_stop_positive = 0.025
    trailing_stop_positive_offset = 0.060

    minimal_roi = {
        '0': 0.060,
        '20': 0.040,
        '40': 0.025,
        '60': 0.015,
        '90': 0.010,
        '120': 0.005,
        '180': 0.0,
    }

    use_exit_signal = False
    exit_profit_only = False

    exit_zscore_threshold = 0.2
    time_stop_minutes = 30
    time_stop_max_profit = 0.0

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._cache_date: Optional[datetime] = None
        self._cache_stats: Optional[dict] = None
        self._pair_cooldown_until: Dict[str, datetime] = {}

    # --- MANAGEMENT GENOME ---
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag: Optional[str], side: str, **kwargs) -> float:
        return 2.0

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        # No scaling; position size controlled by max_open_trades wallet split
        return None

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        self._pair_cooldown_until[pair] = current_time + timedelta(minutes=self.pair_cooldown_minutes)
        return True

    # --- ENTRY GENOME HELPERS ---
    def _composite_score(self, df: DataFrame) -> float:
        if df is None or len(df) < 16:
            return np.nan
        close = df['close']
        r15 = close.pct_change(1).iloc[-1]
        r1h = close.pct_change(4).iloc[-1]
        r4h = close.pct_change(16).iloc[-1]
        if pd.isna(r15) or pd.isna(r1h) or pd.isna(r4h):
            return np.nan
        return 0.5 * r15 + 0.3 * r1h + 0.2 * r4h

    def _universe_stats(self, dataframe: DataFrame, metadata: dict) -> dict:
        if self.dp is None or not hasattr(self.dp, 'current_whitelist'):
            return {
                'mean': 0.0, 'std': 1e-9, 'zscores': {},
                'regime': 'neutral', 'long_threshold': self.zscore_entry,
                'short_threshold': -self.zscore_entry
            }

        current_date = dataframe['date'].iloc[-1]
        if self._cache_date == current_date and self._cache_stats is not None:
            return self._cache_stats

        whitelist = self.dp.current_whitelist()
        scores = {}

        for pair in whitelist:
            if pair == metadata['pair']:
                df = dataframe
            else:
                df = self.dp.get_pair_dataframe(pair, self.timeframe)
            if df is None or df.empty:
                continue
            df = df[df['date'] <= current_date]
            if len(df) < 16:
                continue
            scores[pair] = self._composite_score(df)

        if len(scores) < 3:
            stats = {
                'mean': 0.0, 'std': 1e-9, 'zscores': scores,
                'regime': 'neutral', 'long_threshold': self.zscore_entry,
                'short_threshold': -self.zscore_entry
            }
            self._cache_date = current_date
            self._cache_stats = stats
            return stats

        series = pd.Series(scores)
        mean = series.mean()
        std = series.std()
        if pd.isna(std) or std == 0:
            std = 1e-9

        zseries = (series - mean) / std
        zscores = zseries.to_dict()

        pos_z = zseries[zseries > 0].sort_values(ascending=False)
        neg_z = zseries[zseries < 0].sort_values()
        long_thresh = pos_z.iloc[min(len(pos_z) - 1, self.entry_rank_top_n - 1)] if len(pos_z) > 0 else self.zscore_entry
        short_thresh = neg_z.iloc[min(len(neg_z) - 1, self.entry_rank_top_n - 1)] if len(neg_z) > 0 else -self.zscore_entry

        if mean > 0:
            regime = 'bullish'
        elif mean < self.regime_bear_threshold:
            regime = 'bearish'
        else:
            regime = 'neutral'

        stats = {
            'mean': float(mean),
            'std': float(std),
            'zscores': zscores,
            'regime': regime,
            'long_threshold': float(long_thresh),
            'short_threshold': float(short_thresh),
        }
        self._cache_date = current_date
        self._cache_stats = stats
        return stats

    def _in_cooldown(self, pair: str, current_time: datetime) -> bool:
        until = self._pair_cooldown_until.get(pair)
        if until is None:
            return False
        return current_time < until

    # --- ENTRY GENOME ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['return_15m'] = dataframe['close'].pct_change(1)
        dataframe['return_1h'] = dataframe['close'].pct_change(4)
        dataframe['return_4h'] = dataframe['close'].pct_change(16)
        dataframe['composite'] = (
            0.5 * dataframe['return_15m']
            + 0.3 * dataframe['return_1h']
            + 0.2 * dataframe['return_4h']
        )
        dataframe['composite_prev'] = dataframe['composite'].shift(1)

        # True Range / ATR for volatility confirmation
        prev_close = dataframe['close'].shift(1)
        tr1 = dataframe['high'] - dataframe['low']
        tr2 = (dataframe['high'] - prev_close).abs()
        tr3 = (dataframe['low'] - prev_close).abs()
        dataframe['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        dataframe['atr'] = dataframe['tr'].rolling(window=14).mean()

        # Volume confirmation baseline
        dataframe['volume_sma'] = dataframe['volume'].rolling(window=16).mean()

        # Adverse-swing flags: micro pullbacks within the outlier move
        dataframe['adverse_long'] = dataframe['close'] < dataframe['open']
        dataframe['adverse_short'] = dataframe['close'] > dataframe['open']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        stats = self._universe_stats(dataframe, metadata)
        pair = metadata['pair']
        z = stats['zscores'].get(pair, 0.0)

        if pd.isna(z):
            return dataframe

        if stats['std'] < self.min_dispersion:
            return dataframe

        current_time = dataframe['date'].iloc[-1]
        if self._in_cooldown(pair, current_time):
            return dataframe

        last_idx = dataframe.index[-1]

        cur_comp = dataframe['composite'].iloc[-1]
        prev_comp = dataframe['composite_prev'].iloc[-1]
        cur_vol = dataframe['volume'].iloc[-1]
        vol_sma = dataframe['volume_sma'].iloc[-1]
        cur_tr = dataframe['tr'].iloc[-1]
        cur_atr = dataframe['atr'].iloc[-1]
        adverse_long = dataframe['adverse_long'].iloc[-1]
        adverse_short = dataframe['adverse_short'].iloc[-1]

        if pd.isna(prev_comp) or pd.isna(cur_comp) or pd.isna(vol_sma) or pd.isna(cur_atr):
            return dataframe

        volume_ok = cur_vol > vol_sma * self.volume_factor
        volatility_ok = cur_tr > cur_atr * self.atr_multiplier

        z_prev = (prev_comp - stats['mean']) / stats['std'] if stats['std'] > 1e-9 else 0.0

        long_hurdle = max(self.zscore_entry, stats.get('long_threshold', self.zscore_entry))
        short_hurdle = min(-self.zscore_entry, stats.get('short_threshold', -self.zscore_entry))

        regime = stats.get('regime', 'neutral')

        # Long: top positive outlier, accelerating, volume/vol expansion, red-candle dip
        if (
            z > long_hurdle
            and z > z_prev
            and cur_comp > 0
            and volume_ok
            and volatility_ok
            and adverse_long
        ):
            dataframe.loc[last_idx, 'enter_long'] = 1

        # Short: top negative outlier, accelerating, volume/vol expansion, green-candle bounce
        # Regime-gated: only allowed in confirmed bearish regime
        elif (
            regime == 'bearish'
            and z < short_hurdle
            and z < z_prev
            and cur_comp < 0
            and volume_ok
            and volatility_ok
            and adverse_short
        ):
            dataframe.loc[last_idx, 'enter_short'] = 1

        return dataframe

    # --- EXIT GENOME ---
    def _latest_zscore(self, pair: str, current_time: datetime) -> Optional[float]:
        df = self.dp.get_pair_dataframe(pair, self.timeframe)
        if df is None or len(df) < 16:
            return None
        df = df[df['date'] <= current_time]
        if len(df) < 16:
            return None
        stats = self._universe_stats(df, {'pair': pair})
        return stats['zscores'].get(pair)

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        # 1. Fast time-stop: recycle capital quickly if the trade is not profitable
        if trade.open_date is not None:
            duration = current_time - trade.open_date
            if duration >= timedelta(minutes=self.time_stop_minutes):
                if current_profit <= self.time_stop_max_profit:
                    return "time_stop_loser"

        # 2. Profit-taking mean-reversion: only act when divergence has materially faded
        #    and we are already in the green. Never use zscore to cut a losing trade.
        z = self._latest_zscore(pair, current_time)
        if z is not None and not pd.isna(z) and current_profit > 0.01:
            if trade.is_short and z > -self.exit_zscore_threshold:
                return "zscore_mean_revert_short"
            if not trade.is_short and z < self.exit_zscore_threshold:
                return "zscore_mean_revert_long"

        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
