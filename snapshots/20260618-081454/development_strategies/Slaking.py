from pandas import DataFrame, merge
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, stoploss_from_open
from freqtrade.persistence import Trade
from datetime import datetime
import talib.abstract as ta
import numpy as np


class Slaking(IStrategy):
    """
    Slaking — patient, contrarian, exhaustion-driven mean reversion.

    Identity (UNCHANGED): wait for a statistically significant deviation from
    equilibrium, fade the overreaction, avoid strong trends. Few trades, long
    periods of inactivity are expected and desirable. Oversold longs / overbought
    shorts only.

    This revision keeps that personality but:

      1. ENTRY — loosens the heaviest multiplicative suppressors *modestly* so the
         strategy gets a few more shots at its edge, without becoming a momentum
         chaser. The entry is still "below the band, oversold, turning, in a
         non-trending regime" — just less extreme on each axis.

      2. EXIT — replaced the old "snap back to the mean and dump the whole
         position" logic (which exited fast after waiting forever) with a
         gradual distribution-to-zero. The position is milked out in tranches as
         profit matures, and a residual *runner* is left to ride extended trends,
         protected by a ratcheting trail and a mirror-image exhaustion exit.
    """

    INTERFACE_VERSION = 3
    can_short = True
    timeframe = '4h'
    stoploss = -0.08
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count = 150

    # Profit-taking is fully delegated to the scale-out ladder + trailing stop +
    # exhaustion exit below. A flat minimal_roi would close the WHOLE position at
    # a fixed gain, which is exactly the "exit too soon" behaviour we are removing.
    minimal_roi = {}

    # Patience reinforcement: never let a signal/trail dump a position at a loss —
    # the hard stoploss is the only thing allowed to realise red. Lets winners run.
    exit_profit_only = True

    # Required for the gradual scale-out (partial exits) and the runner trail.
    position_adjustment_enable = True
    use_custom_stoploss = True

    # ----------------------------------------------------------------------- #
    # Core mean-reversion parameters (identity preserved; thresholds softened)
    # ----------------------------------------------------------------------- #
    # bb_std 2.2 -> 2.0: still a 2-sigma capitulation break, just not as rare.
    bb_period = IntParameter(20, 40, default=24, space="buy", optimize=False)
    bb_std = DecimalParameter(2.0, 3.5, default=2.0, space="buy", optimize=False)
    rsi_period = IntParameter(10, 20, default=14, space="buy", optimize=False)
    # rsi gates relaxed one notch (38->42 / 62->58): oversold/overbought, not extreme.
    rsi_long_max = IntParameter(25, 45, default=42, space="buy", optimize=False)
    rsi_short_min = IntParameter(55, 75, default=58, space="buy", optimize=False)
    adx_period = IntParameter(10, 20, default=14, space="buy", optimize=False)
    # adx_max 28 -> 32: still demands a non-trending regime, just a wider window.
    adx_max = IntParameter(20, 40, default=32, space="buy", optimize=False)

    # Exhaustion confirmation & regime
    vol_ma_period = IntParameter(20, 40, default=24, space="buy", optimize=False)
    # vol_mult 1.3 -> 1.15: this AND-ed spike was the single heaviest suppressor —
    # capitulation candles do not always print a big volume bar.
    vol_mult = DecimalParameter(1.0, 2.5, default=1.15, space="buy", optimize=False)
    stoch_period = IntParameter(10, 20, default=14, space="buy", optimize=False)
    stoch_smooth = IntParameter(3, 5, default=3, space="buy", optimize=False)
    atr_period = IntParameter(10, 20, default=14, space="buy", optimize=False)
    # stoch extreme 20/80 -> 30/70: we no longer require the deepest possible
    # stochastic reading on top of an already-rare band break + RSI extreme.
    stoch_long_max = IntParameter(15, 35, default=30, space="buy", optimize=False)
    stoch_short_min = IntParameter(65, 85, default=70, space="buy", optimize=False)

    # ----------------------------------------------------------------------- #
    # Gradual distribution-to-zero (exit philosophy = entry philosophy)
    # ----------------------------------------------------------------------- #
    # Each rung sells a fraction of the *remaining* position once profit matures.
    # Geometric decay leaves a runner instead of a hard close:
    #   100% -> 75% -> ~50% -> ~30% -> ~15% (the runner) -> trail/exhaustion.
    # (profit_threshold, fraction_of_remaining_to_sell)
    scale_out_ladder = [
        (0.03, 0.25),   # +3%:  first reversion target reached -> bank a quarter
        (0.06, 0.33),   # +6%:  reversion confirmed
        (0.10, 0.40),   # +10%: move is extending
        (0.16, 0.50),   # +16%: distribute again, keep the runner
    ]
    # Don't fire a partial smaller than this notional (avoids dust exits).
    min_partial_notional = 12.0

    # Runner trail: once meaningfully in profit, ratchet a stop behind the move so
    # the residual can ride an extended trend but never gives the gains all back.
    trail_start = 0.05   # start trailing at +5% (locks ~breakeven first)
    trail_gap = 0.05     # leave the runner 5% of room to breathe

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(p, '1d') for p in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Primary 4h indicators
        middle = dataframe['close'].rolling(window=self.bb_period.value).mean()
        std = dataframe['close'].rolling(window=self.bb_period.value).std()
        dataframe['bb_middleband'] = middle
        dataframe['bb_lowerband'] = middle - self.bb_std.value * std
        dataframe['bb_upperband'] = middle + self.bb_std.value * std
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period.value)
        dataframe['volume_ma'] = dataframe['volume'].rolling(window=self.vol_ma_period.value).mean()
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        stoch = ta.STOCH(dataframe, fastk_period=self.stoch_period.value,
                         slowk_period=self.stoch_smooth.value, slowd_period=self.stoch_smooth.value)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']

        # Informative 1d for higher-timeframe regime
        if self.dp:
            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1d')
            if not informative.empty:
                informative['daily_rsi'] = ta.RSI(informative, timeperiod=14)
                informative['daily_adx'] = ta.ADX(informative, timeperiod=14)
                informative = informative[['date', 'daily_rsi', 'daily_adx']]
                dataframe = merge(dataframe, informative, on='date', how='left')
                dataframe.ffill(inplace=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0

        daily_rsi = dataframe.get('daily_rsi', 50)
        daily_adx = dataframe.get('daily_adx', 20)

        vol_ok = dataframe['volume'] > (dataframe['volume_ma'] * self.vol_mult.value)

        long_cond = (
            (dataframe['close'] < dataframe['bb_lowerband']) &
            (dataframe['rsi'] < self.rsi_long_max.value) &
            (dataframe['adx'] < self.adx_max.value) &
            (dataframe['stoch_k'] < self.stoch_long_max.value) &
            (dataframe['stoch_k'] > dataframe['stoch_d']) &   # momentum turning up (kept: cheap conviction)
            vol_ok &
            (daily_rsi < 70) &
            (daily_adx < 40)
        )
        dataframe.loc[long_cond, 'enter_long'] = 1

        short_cond = (
            (dataframe['close'] > dataframe['bb_upperband']) &
            (dataframe['rsi'] > self.rsi_short_min.value) &
            (dataframe['adx'] < self.adx_max.value) &
            (dataframe['stoch_k'] > self.stoch_short_min.value) &
            (dataframe['stoch_k'] < dataframe['stoch_d']) &   # momentum turning down
            vol_ok &
            (daily_rsi > 30) &
            (daily_adx < 40)
        )
        dataframe.loc[short_cond, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Thesis-based FULL exit for the residual runner only — the mirror image of
        # our entry. We entered on oversold exhaustion at the lower band; we close
        # the remainder on overbought exhaustion at the upper band (and vice-versa).
        # While momentum keeps pushing, the runner stays on; the gradual scale-out
        # and the trail have already de-risked everything ahead of this point.
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0

        long_exit = (
            (dataframe['close'] > dataframe['bb_upperband']) &
            (dataframe['rsi'] > self.rsi_short_min.value) &
            (dataframe['stoch_k'] > self.stoch_short_min.value) &
            (dataframe['stoch_k'] < dataframe['stoch_d'])     # up-move rolling over
        )
        dataframe.loc[long_exit, 'exit_long'] = 1

        short_exit = (
            (dataframe['close'] < dataframe['bb_lowerband']) &
            (dataframe['rsi'] < self.rsi_long_max.value) &
            (dataframe['stoch_k'] < self.stoch_long_max.value) &
            (dataframe['stoch_k'] > dataframe['stoch_d'])     # down-move rolling over
        )
        dataframe.loc[short_exit, 'exit_short'] = 1

        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              **kwargs):
        """
        Gradual distribution-to-zero. As profit matures the position is milked out
        one rung at a time, each rung selling a fraction of what remains. We never
        ADD to a position (Slaking commits once, with conviction); we only ever
        scale OUT, leaving a residual runner for the trail / exhaustion exit.
        """
        if current_profit <= 0:
            return None

        # nr_of_successful_exits is the rung counter and survives restarts: rung i
        # fires once we have already taken i partials and profit clears its level.
        rung = trade.nr_of_successful_exits
        if rung >= len(self.scale_out_ladder):
            return None  # only the runner is left — hand it to trail / exhaustion

        threshold, fraction = self.scale_out_ladder[rung]
        if current_profit < threshold:
            return None

        reduce_stake = float(trade.stake_amount) * fraction

        # Skip dust partials (and respect the exchange minimum) — the rung will
        # simply fire on a later candle once the slice is large enough.
        floor = self.min_partial_notional
        if min_stake is not None:
            floor = max(floor, float(min_stake))
        if reduce_stake < floor:
            return None

        return -reduce_stake, f"scale_out_{rung + 1}_{int(threshold * 100)}pct"

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        **kwargs) -> float | None:
        """
        Runner trail. Below trail_start we keep the original hard stop (return None
        = unchanged) so the trade has room to work. Once it is meaningfully green,
        ratchet a stop trail_gap behind the best profit so the residual position
        can ride an extended trend while protecting the milked gains.
        """
        if current_profit < self.trail_start:
            return None
        desired_profit = current_profit - self.trail_gap
        return stoploss_from_open(desired_profit, current_profit,
                                  is_short=trade.is_short, leverage=trade.leverage)

    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag: str | None, side: str, **kwargs) -> float:
        return 1.0

    def custom_stake_amount(self, pair: str, current_time, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: str | None, side: str, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) < 50:
            return proposed_stake
        last_atr = dataframe['atr'].iloc[-1]
        atr_mean = dataframe['atr'].rolling(window=50).mean().iloc[-1]
        if np.isfinite(last_atr) and np.isfinite(atr_mean) and last_atr > atr_mean * 1.5:
            return proposed_stake * 0.65
        return proposed_stake
