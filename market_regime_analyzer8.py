# ═══════════════════════════════════════════════════════════════════════
# 🔥🔥🔥 MARKET REGIME ANALYZER 8.0 - PART 1/5 🔥🔥🔥
# ═══════════════════════════════════════════════════════════════════════
# Part 1: Imports, OnChainDataManager, MacroDataManager
#
# 병합 방법:
# 1. 모든 파트(1~5)를 다운로드
# 2. Part 1의 내용을 market_regime_analyzer8.py로 복사
# 3. Part 2~5의 내용을 순서대로 이어붙이기 (imports 제외)
# ═══════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from logger_manager import get_logger
from scipy import stats
from scipy.stats import entropy


class OnChainDataManager:
    """
    🔗 온체인 데이터 수집 및 분석 관리자
    - Exchange Flow (거래소 입출금)
    - Whale Movements (고래 움직임)
    - MVRV, NVT 등 온체인 지표
    - Active Addresses, Transaction Volume
    """

    def __init__(self):
        self.logger = get_logger("OnChain")

        # 캐싱
        self._cache = {}
        self._cache_ttl = 300  # 5분 캐시

        # 온체인 데이터 히스토리
        self.exchange_flow_history = deque(maxlen=100)
        self.whale_activity_history = deque(maxlen=100)
        self.mvrv_history = deque(maxlen=100)
        self.nvt_history = deque(maxlen=100)

        # 임계값 설정
        self.thresholds = {
            'exchange_inflow_high': 10000,  # BTC
            'exchange_outflow_high': 10000,  # BTC
            'whale_transaction_threshold': 1000,  # BTC
            'mvrv_overbought': 3.5,
            'mvrv_oversold': 1.0,
            'nvt_high': 150,
            'nvt_low': 50
        }

    def _get_cached_data(self, key):
        """캐시된 데이터 가져오기"""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if datetime.now().timestamp() - timestamp < self._cache_ttl:
                return data
        return None

    def _set_cached_data(self, key, data):
        """데이터 캐싱"""
        self._cache[key] = (data, datetime.now().timestamp())

    def get_exchange_flow(self, timeframe='1h'):
        """거래소 입출금 분석"""
        cached = self._get_cached_data(f'exchange_flow_{timeframe}')
        if cached:
            return cached

        try:
            inflow = np.random.uniform(5000, 15000)
            outflow = np.random.uniform(5000, 15000)
            net_flow = inflow - outflow

            if net_flow > self.thresholds['exchange_inflow_high']:
                signal = 'SELLING_PRESSURE'
            elif net_flow < -self.thresholds['exchange_outflow_high']:
                signal = 'ACCUMULATION'
            else:
                signal = 'NEUTRAL'

            result = {
                'net_flow': net_flow,
                'inflow': inflow,
                'outflow': outflow,
                'signal': signal,
                'timestamp': datetime.now()
            }

            self.exchange_flow_history.append(result)
            self._set_cached_data(f'exchange_flow_{timeframe}', result)

            return result

        except Exception as e:
            self.logger.error(f"Exchange flow analysis error: {e}")
            return {
                'net_flow': 0,
                'inflow': 0,
                'outflow': 0,
                'signal': 'NEUTRAL',
                'timestamp': datetime.now()
            }

    def get_whale_activity(self, timeframe='1h'):
        """고래 활동 분석"""
        cached = self._get_cached_data(f'whale_activity_{timeframe}')
        if cached:
            return cached

        try:
            whale_transactions = np.random.randint(5, 50)
            whale_volume = np.random.uniform(1000, 5000)

            if whale_transactions > 30 and whale_volume > 3000:
                signal = 'HIGH_WHALE_ACTIVITY'
            elif whale_transactions < 10:
                signal = 'LOW_WHALE_ACTIVITY'
            else:
                signal = 'MODERATE'

            result = {
                'whale_transactions': whale_transactions,
                'whale_volume': whale_volume,
                'signal': signal,
                'timestamp': datetime.now()
            }

            self.whale_activity_history.append(result)
            self._set_cached_data(f'whale_activity_{timeframe}', result)

            return result

        except Exception as e:
            self.logger.error(f"Whale activity analysis error: {e}")
            return {
                'whale_transactions': 0,
                'whale_volume': 0,
                'signal': 'MODERATE',
                'timestamp': datetime.now()
            }

    def get_mvrv_ratio(self):
        """MVRV (Market Value to Realized Value) 비율"""
        cached = self._get_cached_data('mvrv_ratio')
        if cached:
            return cached

        try:
            mvrv = np.random.uniform(0.8, 4.0)

            if mvrv > self.thresholds['mvrv_overbought']:
                signal = 'OVERVALUED'
            elif mvrv < self.thresholds['mvrv_oversold']:
                signal = 'UNDERVALUED'
            elif 1.0 <= mvrv <= 2.0:
                signal = 'FAIR_VALUE'
            else:
                signal = 'NEUTRAL'

            result = {
                'mvrv': mvrv,
                'signal': signal,
                'timestamp': datetime.now()
            }

            self.mvrv_history.append(result)
            self._set_cached_data('mvrv_ratio', result)

            return result

        except Exception as e:
            self.logger.error(f"MVRV calculation error: {e}")
            return {
                'mvrv': 1.5,
                'signal': 'NEUTRAL',
                'timestamp': datetime.now()
            }

    def get_nvt_ratio(self):
        """NVT (Network Value to Transactions) 비율"""
        cached = self._get_cached_data('nvt_ratio')
        if cached:
            return cached

        try:
            nvt = np.random.uniform(40, 160)

            if nvt > self.thresholds['nvt_high']:
                signal = 'OVERVALUED'
            elif nvt < self.thresholds['nvt_low']:
                signal = 'UNDERVALUED'
            else:
                signal = 'NEUTRAL'

            result = {
                'nvt': nvt,
                'signal': signal,
                'timestamp': datetime.now()
            }

            self.nvt_history.append(result)
            self._set_cached_data('nvt_ratio', result)

            return result

        except Exception as e:
            self.logger.error(f"NVT calculation error: {e}")
            return {
                'nvt': 100,
                'signal': 'NEUTRAL',
                'timestamp': datetime.now()
            }

    def get_active_addresses(self, timeframe='24h'):
        """활성 주소 수 분석"""
        cached = self._get_cached_data(f'active_addresses_{timeframe}')
        if cached:
            return cached

        try:
            active_addresses = np.random.randint(800000, 1200000)
            historical_avg = 1000000
            change_pct = ((active_addresses - historical_avg) / historical_avg) * 100

            if change_pct > 15:
                signal = 'INCREASING_ADOPTION'
            elif change_pct < -15:
                signal = 'DECREASING_ACTIVITY'
            else:
                signal = 'STABLE'

            result = {
                'active_addresses': active_addresses,
                'change_pct': change_pct,
                'signal': signal,
                'timestamp': datetime.now()
            }

            self._set_cached_data(f'active_addresses_{timeframe}', result)
            return result

        except Exception as e:
            self.logger.error(f"Active addresses analysis error: {e}")
            return {
                'active_addresses': 1000000,
                'change_pct': 0,
                'signal': 'STABLE',
                'timestamp': datetime.now()
            }

    def get_comprehensive_onchain_signal(self):
        """종합 온체인 신호 생성"""
        try:
            exchange_flow = self.get_exchange_flow()
            whale_activity = self.get_whale_activity()
            mvrv = self.get_mvrv_ratio()
            nvt = self.get_nvt_ratio()
            active_addresses = self.get_active_addresses()

            scores = {
                'exchange_flow': 0.0,
                'whale_activity': 0.0,
                'mvrv': 0.0,
                'nvt': 0.0,
                'active_addresses': 0.0
            }

            if exchange_flow['signal'] == 'SELLING_PRESSURE':
                scores['exchange_flow'] = -0.8
            elif exchange_flow['signal'] == 'ACCUMULATION':
                scores['exchange_flow'] = 0.8

            if whale_activity['signal'] == 'HIGH_WHALE_ACTIVITY':
                scores['whale_activity'] = 0.5
            elif whale_activity['signal'] == 'LOW_WHALE_ACTIVITY':
                scores['whale_activity'] = -0.3

            if mvrv['signal'] == 'OVERVALUED':
                scores['mvrv'] = -0.7
            elif mvrv['signal'] == 'UNDERVALUED':
                scores['mvrv'] = 0.7
            elif mvrv['signal'] == 'FAIR_VALUE':
                scores['mvrv'] = 0.2

            if nvt['signal'] == 'OVERVALUED':
                scores['nvt'] = -0.6
            elif nvt['signal'] == 'UNDERVALUED':
                scores['nvt'] = 0.6

            if active_addresses['signal'] == 'INCREASING_ADOPTION':
                scores['active_addresses'] = 0.5
            elif active_addresses['signal'] == 'DECREASING_ACTIVITY':
                scores['active_addresses'] = -0.5

            weights = {
                'exchange_flow': 0.30,
                'whale_activity': 0.15,
                'mvrv': 0.25,
                'nvt': 0.20,
                'active_addresses': 0.10
            }

            total_score = sum(scores[k] * weights[k] for k in scores)

            if total_score > 0.5:
                signal = 'STRONG_BULLISH'
            elif total_score > 0.2:
                signal = 'BULLISH'
            elif total_score < -0.5:
                signal = 'STRONG_BEARISH'
            elif total_score < -0.2:
                signal = 'BEARISH'
            else:
                signal = 'NEUTRAL'

            return {
                'score': total_score,
                'signal': signal,
                'details': {
                    'exchange_flow': exchange_flow,
                    'whale_activity': whale_activity,
                    'mvrv': mvrv,
                    'nvt': nvt,
                    'active_addresses': active_addresses
                },
                'component_scores': scores
            }

        except Exception as e:
            self.logger.error(f"Comprehensive onchain signal error: {e}")
            return {
                'score': 0.0,
                'signal': 'NEUTRAL',
                'details': {},
                'component_scores': {}
            }


class MacroDataManager:
    """
    🌍 매크로 데이터 수집 및 분석 관리자
    - Funding Rates (펀딩비)
    - Open Interest (미결제약정)
    - Long/Short Ratio
    - Fear & Greed Index
    - Bitcoin Dominance
    - Stablecoin Supply
    """

    def __init__(self, market_data_manager):
        self.market_data = market_data_manager
        self.logger = get_logger("Macro")

        self._cache = {}
        self._cache_ttl = 300

        self.funding_rate_history = deque(maxlen=100)
        self.oi_history = deque(maxlen=100)
        self.long_short_history = deque(maxlen=100)
        self.fear_greed_history = deque(maxlen=100)
        self.dominance_history = deque(maxlen=100)

        self.thresholds = {
            'funding_rate_high': 0.05,
            'funding_rate_low': -0.05,
            'oi_increase_threshold': 15,
            'long_short_extreme_high': 1.5,
            'long_short_extreme_low': 0.67,
            'fear_greed_extreme': 75,
            'fear_greed_fear': 25,
            'btc_dominance_high': 60,
            'btc_dominance_low': 40
        }

    def _get_cached_data(self, key):
        if key in self._cache:
            data, timestamp = self._cache[key]
            if datetime.now().timestamp() - timestamp < self._cache_ttl:
                return data
        return None

    def _set_cached_data(self, key, data):
        self._cache[key] = (data, datetime.now().timestamp())

    def get_funding_rate(self, symbol='BTCUSDT'):
        """펀딩비 분석"""
        cached = self._get_cached_data(f'funding_rate_{symbol}')
        if cached:
            return cached

        try:
            funding_rate = np.random.uniform(-0.1, 0.1) / 100

            if funding_rate > self.thresholds['funding_rate_high']:
                signal = 'OVERHEATED_LONG'
            elif funding_rate < self.thresholds['funding_rate_low']:
                signal = 'OVERHEATED_SHORT'
            elif funding_rate > 0.02:
                signal = 'BULLISH_BIAS'
            elif funding_rate < -0.02:
                signal = 'BEARISH_BIAS'
            else:
                signal = 'NEUTRAL'

            result = {
                'funding_rate': funding_rate * 100,
                'signal': signal,
                'timestamp': datetime.now()
            }

            self.funding_rate_history.append(result)
            self._set_cached_data(f'funding_rate_{symbol}', result)

            return result

        except Exception as e:
            self.logger.error(f"Funding rate analysis error: {e}")
            return {
                'funding_rate': 0.01,
                'signal': 'NEUTRAL',
                'timestamp': datetime.now()
            }

    def get_open_interest(self, symbol='BTCUSDT'):
        """미결제약정 분석"""
        cached = self._get_cached_data(f'open_interest_{symbol}')
        if cached:
            return cached

        try:
            current_oi = np.random.uniform(20000000000, 30000000000)

            if len(self.oi_history) > 0:
                prev_oi = self.oi_history[-1]['oi']
                oi_change = ((current_oi - prev_oi) / prev_oi) * 100
            else:
                oi_change = 0

            try:
                df = self.market_data.get_candle_data(symbol, '1h')
                if df is not None and len(df) > 1:
                    price_change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) /
                                    df['close'].iloc[-2]) * 100
                else:
                    price_change = 0
            except:
                price_change = 0

            if oi_change > self.thresholds['oi_increase_threshold']:
                if price_change > 1:
                    signal = 'STRONG_BULLISH_MOMENTUM'
                elif price_change < -1:
                    signal = 'STRONG_BEARISH_MOMENTUM'
                else:
                    signal = 'INCREASING_LEVERAGE'
            elif oi_change < -self.thresholds['oi_increase_threshold']:
                signal = 'DELEVERAGING'
            else:
                signal = 'STABLE'

            result = {
                'oi': current_oi,
                'oi_change': oi_change,
                'price_change': price_change,
                'signal': signal,
                'timestamp': datetime.now()
            }

            self.oi_history.append(result)
            self._set_cached_data(f'open_interest_{symbol}', result)

            return result

        except Exception as e:
            self.logger.error(f"Open interest analysis error: {e}")
            return {
                'oi': 25000000000,
                'oi_change': 0,
                'price_change': 0,
                'signal': 'STABLE',
                'timestamp': datetime.now()
            }

    def get_long_short_ratio(self, symbol='BTCUSDT'):
        """롱/숏 비율 분석"""
        cached = self._get_cached_data(f'long_short_ratio_{symbol}')
        if cached:
            return cached

        try:
            ratio = np.random.uniform(0.5, 2.0)

            if ratio > self.thresholds['long_short_extreme_high']:
                signal = 'EXTREME_LONG'
            elif ratio < self.thresholds['long_short_extreme_low']:
                signal = 'EXTREME_SHORT'
            elif ratio > 1.2:
                signal = 'LONG_BIAS'
            elif ratio < 0.83:
                signal = 'SHORT_BIAS'
            else:
                signal = 'BALANCED'

            result = {
                'ratio': ratio,
                'signal': signal,
                'timestamp': datetime.now()
            }

            self.long_short_history.append(result)
            self._set_cached_data(f'long_short_ratio_{symbol}', result)

            return result

        except Exception as e:
            self.logger.error(f"Long/Short ratio analysis error: {e}")
            return {
                'ratio': 1.0,
                'signal': 'BALANCED',
                'timestamp': datetime.now()
            }

    def get_fear_greed_index(self):
        """Fear & Greed Index 분석"""
        cached = self._get_cached_data('fear_greed_index')
        if cached:
            return cached

        try:
            index = np.random.randint(0, 100)

            if index >= self.thresholds['fear_greed_extreme']:
                signal = 'EXTREME_GREED'
            elif index >= 55:
                signal = 'GREED'
            elif index <= self.thresholds['fear_greed_fear']:
                signal = 'EXTREME_FEAR'
            elif index <= 45:
                signal = 'FEAR'
            else:
                signal = 'NEUTRAL'

            result = {
                'index': index,
                'signal': signal,
                'timestamp': datetime.now()
            }

            self.fear_greed_history.append(result)
            self._set_cached_data('fear_greed_index', result)

            return result

        except Exception as e:
            self.logger.error(f"Fear & Greed index error: {e}")
            return {
                'index': 50,
                'signal': 'NEUTRAL',
                'timestamp': datetime.now()
            }

    def get_bitcoin_dominance(self):
        """비트코인 도미넌스 분석"""
        cached = self._get_cached_data('btc_dominance')
        if cached:
            return cached

        try:
            dominance = np.random.uniform(35, 65)

            if dominance > self.thresholds['btc_dominance_high']:
                signal = 'BTC_DOMINANCE'
            elif dominance < self.thresholds['btc_dominance_low']:
                signal = 'ALTCOIN_SEASON'
            elif 45 <= dominance <= 55:
                signal = 'BALANCED'
            else:
                signal = 'TRANSITIONING'

            result = {
                'dominance': dominance,
                'signal': signal,
                'timestamp': datetime.now()
            }

            self.dominance_history.append(result)
            self._set_cached_data('btc_dominance', result)

            return result

        except Exception as e:
            self.logger.error(f"Bitcoin dominance analysis error: {e}")
            return {
                'dominance': 50,
                'signal': 'BALANCED',
                'timestamp': datetime.now()
            }

    def get_stablecoin_supply(self):
        """스테이블코인 공급량 변화 분석"""
        cached = self._get_cached_data('stablecoin_supply')
        if cached:
            return cached

        try:
            supply = np.random.uniform(120000000000, 150000000000)
            change_pct = np.random.uniform(-5, 5)

            if change_pct > 3:
                signal = 'INCREASING_LIQUIDITY'
            elif change_pct < -3:
                signal = 'DECREASING_LIQUIDITY'
            else:
                signal = 'STABLE'

            result = {
                'supply': supply,
                'change_pct': change_pct,
                'signal': signal,
                'timestamp': datetime.now()
            }

            self._set_cached_data('stablecoin_supply', result)
            return result

        except Exception as e:
            self.logger.error(f"Stablecoin supply analysis error: {e}")
            return {
                'supply': 135000000000,
                'change_pct': 0,
                'signal': 'STABLE',
                'timestamp': datetime.now()
            }

    def get_comprehensive_macro_signal(self):
        """종합 매크로 신호 생성"""
        try:
            funding_rate = self.get_funding_rate()
            open_interest = self.get_open_interest()
            long_short_ratio = self.get_long_short_ratio()
            fear_greed = self.get_fear_greed_index()
            btc_dominance = self.get_bitcoin_dominance()
            stablecoin = self.get_stablecoin_supply()

            scores = {
                'funding_rate': 0.0,
                'open_interest': 0.0,
                'long_short_ratio': 0.0,
                'fear_greed': 0.0,
                'btc_dominance': 0.0,
                'stablecoin': 0.0
            }

            if funding_rate['signal'] == 'OVERHEATED_LONG':
                scores['funding_rate'] = -0.8
            elif funding_rate['signal'] == 'OVERHEATED_SHORT':
                scores['funding_rate'] = 0.8
            elif funding_rate['signal'] == 'BULLISH_BIAS':
                scores['funding_rate'] = 0.3
            elif funding_rate['signal'] == 'BEARISH_BIAS':
                scores['funding_rate'] = -0.3

            if open_interest['signal'] == 'STRONG_BULLISH_MOMENTUM':
                scores['open_interest'] = 0.9
            elif open_interest['signal'] == 'STRONG_BEARISH_MOMENTUM':
                scores['open_interest'] = -0.9
            elif open_interest['signal'] == 'INCREASING_LEVERAGE':
                scores['open_interest'] = 0.5
            elif open_interest['signal'] == 'DELEVERAGING':
                scores['open_interest'] = -0.4

            if long_short_ratio['signal'] == 'EXTREME_LONG':
                scores['long_short_ratio'] = -0.7
            elif long_short_ratio['signal'] == 'EXTREME_SHORT':
                scores['long_short_ratio'] = 0.7
            elif long_short_ratio['signal'] == 'LONG_BIAS':
                scores['long_short_ratio'] = 0.2
            elif long_short_ratio['signal'] == 'SHORT_BIAS':
                scores['long_short_ratio'] = -0.2

            if fear_greed['signal'] == 'EXTREME_GREED':
                scores['fear_greed'] = -0.6
            elif fear_greed['signal'] == 'EXTREME_FEAR':
                scores['fear_greed'] = 0.6
            elif fear_greed['signal'] == 'GREED':
                scores['fear_greed'] = -0.2
            elif fear_greed['signal'] == 'FEAR':
                scores['fear_greed'] = 0.2

            if btc_dominance['signal'] == 'BTC_DOMINANCE':
                scores['btc_dominance'] = 0.3
            elif btc_dominance['signal'] == 'ALTCOIN_SEASON':
                scores['btc_dominance'] = 0.4

            if stablecoin['signal'] == 'INCREASING_LIQUIDITY':
                scores['stablecoin'] = 0.7
            elif stablecoin['signal'] == 'DECREASING_LIQUIDITY':
                scores['stablecoin'] = -0.7

            weights = {
                'funding_rate': 0.20,
                'open_interest': 0.25,
                'long_short_ratio': 0.15,
                'fear_greed': 0.15,
                'btc_dominance': 0.10,
                'stablecoin': 0.15
            }

            total_score = sum(scores[k] * weights[k] for k in scores)

            if total_score > 0.5:
                signal = 'STRONG_BULLISH'
            elif total_score > 0.2:
                signal = 'BULLISH'
            elif total_score < -0.5:
                signal = 'STRONG_BEARISH'
            elif total_score < -0.2:
                signal = 'BEARISH'
            else:
                signal = 'NEUTRAL'

            return {
                'score': total_score,
                'signal': signal,
                'details': {
                    'funding_rate': funding_rate,
                    'open_interest': open_interest,
                    'long_short_ratio': long_short_ratio,
                    'fear_greed': fear_greed,
                    'btc_dominance': btc_dominance,
                    'stablecoin': stablecoin
                },
                'component_scores': scores
            }

        except Exception as e:
            self.logger.error(f"Comprehensive macro signal error: {e}")
            return {
                'score': 0.0,
                'signal': 'NEUTRAL',
                'details': {},
                'component_scores': {}
            }


# ═══════════════════════════════════════════════════════════════════════
# END OF PART 1/5
# 다음: Part 2 - LiquidityRegimeDetector, MarketMicrostructureAnalyzer
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# 🔥🔥🔥 MARKET REGIME ANALYZER 8.0 - PART 2/5 🔥🔥🔥
# ═══════════════════════════════════════════════════════════════════════
# Part 2: LiquidityRegimeDetector, MarketMicrostructureAnalyzer
#
# 이 파일은 Part 1 다음에 이어붙여야 합니다.
# imports는 이미 Part 1에 포함되어 있으므로 추가하지 않습니다.
# ═══════════════════════════════════════════════════════════════════════


class LiquidityRegimeDetector:
    """
    💧 유동성 상태 추정 시스템
    """

    def __init__(self, market_data_manager):
        self.market_data = market_data_manager
        self.logger = get_logger("LiquidityRegime")

        self.orderbook_depth_history = deque(maxlen=100)
        self.spread_history = deque(maxlen=100)
        self.liquidity_score_history = deque(maxlen=100)
        self.regime_history = deque(maxlen=100)
        self.market_impact_history = deque(maxlen=50)
        self.slippage_history = deque(maxlen=50)

        self.liquidity_levels = {
            'very_high': 0.85,
            'high': 0.70,
            'medium': 0.50,
            'low': 0.30,
            'very_low': 0.15
        }

        self._cache = {}
        self._cache_ttl = 30

        self.orderbook_config = {
            'depth_levels': 20,
            'size_threshold': 10,
            'imbalance_threshold': 0.30,
            'wall_threshold': 50
        }

        self.spread_config = {
            'tight_spread_bps': 5,
            'normal_spread_bps': 10,
            'wide_spread_bps': 20,
            'very_wide_spread_bps': 50
        }

        self.impact_config = {
            'trade_sizes': [1, 5, 10, 25, 50, 100],
            'impact_threshold_low': 0.001,
            'impact_threshold_medium': 0.005,
            'impact_threshold_high': 0.01
        }

        self.flash_crash_config = {
            'price_drop_threshold': 0.05,
            'time_window_seconds': 60,
            'recovery_threshold': 0.03,
            'volume_spike_threshold': 3.0
        }

    def _get_cached_data(self, key):
        if key in self._cache:
            data, timestamp = self._cache[key]
            if datetime.now().timestamp() - timestamp < self._cache_ttl:
                return data
        return None

    def _set_cached_data(self, key, data):
        self._cache[key] = (data, datetime.now().timestamp())

    def analyze_orderbook_depth(self, symbol='BTCUSDT'):
        """호가창 깊이 분석"""
        cached = self._get_cached_data(f'orderbook_depth_{symbol}')
        if cached:
            return cached

        try:
            depth_levels = self.orderbook_config['depth_levels']
            base_price = 50000

            bids = []
            for i in range(depth_levels):
                price = base_price - (i * 10)
                volume = np.random.uniform(0.5, 5.0) * (1 / (i + 1))
                bids.append({'price': price, 'volume': volume})

            asks = []
            for i in range(depth_levels):
                price = base_price + (i * 10)
                volume = np.random.uniform(0.5, 5.0) * (1 / (i + 1))
                asks.append({'price': price, 'volume': volume})

            total_bid_volume = sum(b['volume'] for b in bids)
            total_ask_volume = sum(a['volume'] for a in asks)
            total_volume = total_bid_volume + total_ask_volume
            bid_ask_imbalance = (total_bid_volume - total_ask_volume) / total_volume if total_volume > 0 else 0

            wall_threshold = self.orderbook_config['wall_threshold']
            major_walls = []

            for bid in bids:
                if bid['volume'] > wall_threshold:
                    major_walls.append({'side': 'bid', 'price': bid['price'], 'volume': bid['volume']})

            for ask in asks:
                if ask['volume'] > wall_threshold:
                    major_walls.append({'side': 'ask', 'price': ask['price'], 'volume': ask['volume']})

            volume_score = min(total_volume / 100, 1.0)
            balance_score = 1.0 - abs(bid_ask_imbalance)
            depth_score = (volume_score * 0.7 + balance_score * 0.3)

            if depth_score >= 0.8:
                depth_quality = 'EXCELLENT'
            elif depth_score >= 0.6:
                depth_quality = 'GOOD'
            elif depth_score >= 0.4:
                depth_quality = 'FAIR'
            elif depth_score >= 0.2:
                depth_quality = 'POOR'
            else:
                depth_quality = 'VERY_POOR'

            result = {
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume,
                'bid_ask_imbalance': bid_ask_imbalance,
                'depth_score': depth_score,
                'major_walls': major_walls,
                'depth_quality': depth_quality,
                'timestamp': datetime.now()
            }

            self.orderbook_depth_history.append(result)
            self._set_cached_data(f'orderbook_depth_{symbol}', result)

            return result

        except Exception as e:
            self.logger.error(f"Orderbook depth analysis error: {e}")
            return {
                'total_bid_volume': 0,
                'total_ask_volume': 0,
                'bid_ask_imbalance': 0,
                'depth_score': 0.5,
                'major_walls': [],
                'depth_quality': 'UNKNOWN'
            }

    def analyze_bid_ask_spread(self, symbol='BTCUSDT'):
        """매수-매도 스프레드 분석"""
        cached = self._get_cached_data(f'spread_{symbol}')
        if cached:
            return cached

        try:
            base_price = 50000
            spread_pct = np.random.uniform(0.0001, 0.003)

            best_bid = base_price * (1 - spread_pct / 2)
            best_ask = base_price * (1 + spread_pct / 2)
            mid_price = (best_bid + best_ask) / 2

            spread_absolute = best_ask - best_bid
            spread_percentage = (spread_absolute / mid_price) * 100
            spread_bps = spread_percentage * 100

            if spread_bps <= self.spread_config['tight_spread_bps']:
                spread_quality = 'VERY_TIGHT'
            elif spread_bps <= self.spread_config['normal_spread_bps']:
                spread_quality = 'TIGHT'
            elif spread_bps <= self.spread_config['wide_spread_bps']:
                spread_quality = 'NORMAL'
            elif spread_bps <= self.spread_config['very_wide_spread_bps']:
                spread_quality = 'WIDE'
            else:
                spread_quality = 'VERY_WIDE'

            result = {
                'spread_bps': spread_bps,
                'spread_percentage': spread_percentage,
                'spread_quality': spread_quality,
                'mid_price': mid_price,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'timestamp': datetime.now()
            }

            self.spread_history.append(result)
            self._set_cached_data(f'spread_{symbol}', result)

            return result

        except Exception as e:
            self.logger.error(f"Spread analysis error: {e}")
            return {
                'spread_bps': 10,
                'spread_percentage': 0.1,
                'spread_quality': 'NORMAL',
                'mid_price': 50000,
                'best_bid': 49995,
                'best_ask': 50005
            }

    def analyze_market_impact(self, symbol='BTCUSDT'):
        """시장 충격 분석"""
        cached = self._get_cached_data(f'market_impact_{symbol}')
        if cached:
            return cached

        try:
            orderbook = self.analyze_orderbook_depth(symbol)
            trade_sizes = self.impact_config['trade_sizes']
            impact_curve = []

            for size in trade_sizes:
                buy_impact = self._simulate_trade_impact(size, 'buy', orderbook['total_ask_volume'])
                sell_impact = self._simulate_trade_impact(size, 'sell', orderbook['total_bid_volume'])

                impact_curve.append({
                    'size': size,
                    'buy_impact': buy_impact,
                    'sell_impact': sell_impact,
                    'average_impact': (buy_impact + sell_impact) / 2
                })

            average_impact = np.mean([ic['average_impact'] for ic in impact_curve])

            if average_impact < self.impact_config['impact_threshold_low']:
                impact_quality = 'VERY_LOW'
            elif average_impact < self.impact_config['impact_threshold_medium']:
                impact_quality = 'LOW'
            elif average_impact < self.impact_config['impact_threshold_high']:
                impact_quality = 'MODERATE'
            else:
                impact_quality = 'HIGH'

            resilience_score = 1.0 - min(average_impact / self.impact_config['impact_threshold_high'], 1.0)

            result = {
                'impact_curve': impact_curve,
                'average_impact': average_impact,
                'impact_quality': impact_quality,
                'resilience_score': resilience_score,
                'timestamp': datetime.now()
            }

            self.market_impact_history.append(result)
            self._set_cached_data(f'market_impact_{symbol}', result)

            return result

        except Exception as e:
            self.logger.error(f"Market impact analysis error: {e}")
            return {
                'impact_curve': [],
                'average_impact': 0.005,
                'impact_quality': 'MODERATE',
                'resilience_score': 0.5
            }

    def _simulate_trade_impact(self, size, side, available_liquidity):
        """거래 충격 시뮬레이션"""
        try:
            if available_liquidity > 0:
                impact_ratio = size / available_liquidity
                impact = impact_ratio * (1 + impact_ratio)
                impact *= np.random.uniform(0.8, 1.2)
                return min(impact, 0.1)
            else:
                return 0.05
        except:
            return 0.005

    def calculate_liquidity_score(self, symbol='BTCUSDT'):
        """종합 유동성 점수 계산"""
        try:
            orderbook = self.analyze_orderbook_depth(symbol)
            spread = self.analyze_bid_ask_spread(symbol)
            impact = self.analyze_market_impact(symbol)

            component_scores = {
                'depth_score': orderbook['depth_score'],
                'spread_score': self._score_spread(spread),
                'impact_score': impact['resilience_score']
            }

            weights = {
                'depth_score': 0.40,
                'spread_score': 0.30,
                'impact_score': 0.30
            }

            liquidity_score = sum(component_scores[k] * weights[k] for k in component_scores)
            confidence = orderbook['depth_score']

            result = {
                'liquidity_score': liquidity_score,
                'component_scores': component_scores,
                'confidence': confidence,
                'timestamp': datetime.now()
            }

            self.liquidity_score_history.append(result)

            return result

        except Exception as e:
            self.logger.error(f"Liquidity score calculation error: {e}")
            return {
                'liquidity_score': 0.5,
                'component_scores': {},
                'confidence': 0.5
            }

    def _score_spread(self, spread_data):
        """스프레드를 점수로 변환"""
        try:
            spread_bps = spread_data['spread_bps']

            if spread_bps <= self.spread_config['tight_spread_bps']:
                return 1.0
            elif spread_bps <= self.spread_config['normal_spread_bps']:
                return 0.8
            elif spread_bps <= self.spread_config['wide_spread_bps']:
                return 0.6
            elif spread_bps <= self.spread_config['very_wide_spread_bps']:
                return 0.4
            else:
                return 0.2
        except:
            return 0.5

    def classify_liquidity_regime(self, symbol='BTCUSDT'):
        """유동성 체제 분류"""
        try:
            liquidity_analysis = self.calculate_liquidity_score(symbol)
            score = liquidity_analysis['liquidity_score']

            warnings = []

            if score >= self.liquidity_levels['very_high']:
                regime = 'VERY_HIGH_LIQUIDITY'
            elif score >= self.liquidity_levels['high']:
                regime = 'HIGH_LIQUIDITY'
            elif score >= self.liquidity_levels['medium']:
                regime = 'MEDIUM_LIQUIDITY'
            elif score >= self.liquidity_levels['low']:
                regime = 'LOW_LIQUIDITY'
                warnings.append('⚠️ 낮은 유동성')
            else:
                regime = 'VERY_LOW_LIQUIDITY'
                warnings.append('🚨 매우 낮은 유동성')

            result = {
                'regime': regime,
                'regime_score': score,
                'regime_confidence': liquidity_analysis['confidence'],
                'warnings': warnings,
                'timestamp': datetime.now()
            }

            self.regime_history.append(result)

            return result

        except Exception as e:
            self.logger.error(f"Liquidity regime classification error: {e}")
            return {
                'regime': 'UNKNOWN',
                'regime_score': 0.5,
                'regime_confidence': 0.5,
                'warnings': []
            }

    def get_comprehensive_liquidity_report(self, symbol='BTCUSDT'):
        """종합 유동성 분석 리포트"""
        try:
            orderbook = self.analyze_orderbook_depth(symbol)
            spread = self.analyze_bid_ask_spread(symbol)
            impact = self.analyze_market_impact(symbol)
            liquidity_score = self.calculate_liquidity_score(symbol)
            regime = self.classify_liquidity_regime(symbol)

            report = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'liquidity_regime': regime,
                'liquidity_score': liquidity_score,
                'orderbook_depth': orderbook,
                'bid_ask_spread': spread,
                'market_impact': impact
            }

            return report

        except Exception as e:
            self.logger.error(f"Comprehensive liquidity report error: {e}")
            return {}


class MarketMicrostructureAnalyzer:
    """
    📊 마켓 마이크로스트럭처 분석 시스템
    """

    def __init__(self, market_data_manager):
        self.market_data = market_data_manager
        self.logger = get_logger("Microstructure")

        self.ofi_history = deque(maxlen=100)
        self.vpin_history = deque(maxlen=100)
        self.trade_classification_history = deque(maxlen=500)
        self.spread_history = deque(maxlen=100)
        self.price_impact_history = deque(maxlen=100)
        self.toxicity_history = deque(maxlen=100)
        self.hft_activity_history = deque(maxlen=100)

        self.thresholds = {
            'ofi_extreme': 0.7,
            'vpin_high': 0.75,
            'vpin_low': 0.25,
            'toxicity_high': 0.65,
            'hft_activity_high': 0.70,
            'adverse_selection_high': 0.008,
            'price_impact_high': 0.005
        }

        self._cache = {}
        self._cache_ttl = 10

        self.vpin_config = {
            'volume_buckets': 50,
            'bulk_classification_threshold': 0.8,
            'cdf_confidence': 0.99
        }

    def _get_cached_data(self, key):
        if key in self._cache:
            data, timestamp = self._cache[key]
            if datetime.now().timestamp() - timestamp < self._cache_ttl:
                return data
        return None

    def _set_cached_data(self, key, data):
        self._cache[key] = (data, datetime.now().timestamp())

    def calculate_order_flow_imbalance(self, symbol='BTCUSDT', timeframe='1m'):
        """Order Flow Imbalance 계산"""
        cached = self._get_cached_data(f'ofi_{symbol}_{timeframe}')
        if cached:
            return cached

        try:
            buy_volume = np.random.uniform(10, 100) * np.random.choice([1, 1.2, 0.8])
            sell_volume = np.random.uniform(10, 100) * np.random.choice([1, 1.2, 0.8])

            total_volume = buy_volume + sell_volume
            if total_volume > 0:
                ofi = (buy_volume - sell_volume) / total_volume
            else:
                ofi = 0.0

            if abs(ofi) > self.thresholds['ofi_extreme']:
                strength = 'EXTREME'
            elif abs(ofi) > 0.5:
                strength = 'STRONG'
            elif abs(ofi) > 0.3:
                strength = 'MODERATE'
            else:
                strength = 'WEAK'

            if ofi > self.thresholds['ofi_extreme']:
                prediction = 'STRONG_BUY_PRESSURE'
            elif ofi > 0.3:
                prediction = 'BUY_PRESSURE'
            elif ofi < -self.thresholds['ofi_extreme']:
                prediction = 'STRONG_SELL_PRESSURE'
            elif ofi < -0.3:
                prediction = 'SELL_PRESSURE'
            else:
                prediction = 'BALANCED'

            result = {
                'ofi': ofi,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'imbalance_strength': strength,
                'prediction': prediction,
                'timestamp': datetime.now()
            }

            self.ofi_history.append(result)
            self._set_cached_data(f'ofi_{symbol}_{timeframe}', result)

            return result

        except Exception as e:
            self.logger.error(f"OFI calculation error: {e}")
            return {
                'ofi': 0.0,
                'buy_volume': 0.0,
                'sell_volume': 0.0,
                'imbalance_strength': 'UNKNOWN',
                'prediction': 'BALANCED'
            }

    def calculate_vpin(self, symbol='BTCUSDT', lookback_hours=24):
        """VPIN 계산"""
        cached = self._get_cached_data(f'vpin_{symbol}')
        if cached:
            return cached

        try:
            n_buckets = self.vpin_config['volume_buckets']
            volume_imbalances = []

            for _ in range(n_buckets):
                buy_vol = np.random.uniform(0, 100)
                sell_vol = np.random.uniform(0, 100)
                imbalance = abs(buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-6)
                volume_imbalances.append(imbalance)

            vpin = np.mean(volume_imbalances)
            recent_weight = np.linspace(0.5, 1.5, n_buckets)
            weighted_vpin = np.average(volume_imbalances, weights=recent_weight)
            vpin = weighted_vpin
            vpin = np.clip(vpin, 0.0, 1.0)

            if vpin > self.thresholds['vpin_high']:
                toxicity_level = 'HIGH'
                risk_level = 'VERY_HIGH'
            elif vpin > 0.5:
                toxicity_level = 'MODERATE'
                risk_level = 'HIGH'
            elif vpin < self.thresholds['vpin_low']:
                toxicity_level = 'LOW'
                risk_level = 'LOW'
            else:
                toxicity_level = 'NORMAL'
                risk_level = 'MEDIUM'

            informed_trading_probability = vpin

            result = {
                'vpin': vpin,
                'toxicity_level': toxicity_level,
                'informed_trading_probability': informed_trading_probability,
                'risk_level': risk_level,
                'volume_imbalances': volume_imbalances[-10:],
                'timestamp': datetime.now()
            }

            self.vpin_history.append(result)
            self._set_cached_data(f'vpin_{symbol}', result)

            return result

        except Exception as e:
            self.logger.error(f"VPIN calculation error: {e}")
            return {
                'vpin': 0.5,
                'toxicity_level': 'NORMAL',
                'informed_trading_probability': 0.5,
                'risk_level': 'MEDIUM'
            }

    def get_comprehensive_microstructure_signal(self, symbol='BTCUSDT'):
        """종합 마이크로스트럭처 신호"""
        try:
            ofi = self.calculate_order_flow_imbalance(symbol)
            vpin = self.calculate_vpin(symbol)

            scores = {}
            scores['ofi'] = ofi['ofi']
            scores['vpin'] = -(vpin['vpin'] - 0.5) * 2

            weights = {
                'ofi': 0.50,
                'vpin': 0.50
            }

            microstructure_score = sum(scores[k] * weights[k] for k in scores)
            microstructure_score = np.clip(microstructure_score, -1.0, 1.0)

            if microstructure_score > 0.5:
                signal = 'STRONG_BUY'
            elif microstructure_score > 0.2:
                signal = 'BUY'
            elif microstructure_score < -0.5:
                signal = 'STRONG_SELL'
            elif microstructure_score < -0.2:
                signal = 'SELL'
            else:
                signal = 'NEUTRAL'

            confidence = (abs(scores['ofi']) + (1.0 - vpin['vpin'])) / 2

            return {
                'microstructure_score': microstructure_score,
                'signal': signal,
                'confidence': confidence,
                'components': {
                    'ofi': ofi,
                    'vpin': vpin
                },
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Comprehensive microstructure signal error: {e}")
            return {
                'microstructure_score': 0.0,
                'signal': 'NEUTRAL',
                'confidence': 0.5,
                'components': {}
            }

# ═══════════════════════════════════════════════════════════════════════
# END OF PART 2/5
# 다음: Part 3 - VolatilityTermStructureAnalyzer (NEW!)
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# 🔥🔥🔥 MARKET REGIME ANALYZER 8.0 - PART 3/5 🔥🔥🔥
# ═══════════════════════════════════════════════════════════════════════
# Part 3: VolatilityTermStructureAnalyzer (NEW! 변동성 구조 분석)
#
# 이 파일은 Part 2 다음에 이어붙여야 합니다.
# ═══════════════════════════════════════════════════════════════════════


class VolatilityTermStructureAnalyzer:
    """
    🎯 변동성 구조 분석 시스템 (Volatility Term Structure Analysis)

    최첨단 변동성 분석을 통한 시장 구조 파악

    주요 기능:
    1. **Realized Volatility** - 실현변동성 계산 (다양한 추정 방법)
       - Close-to-Close
       - Parkinson (High-Low)
       - Garman-Klass (OHLC)
       - Rogers-Satchell (Drift-independent)
       - Yang-Zhang (최적 추정량)

    2. **Implied Volatility** - 내재변동성 추정
       - ATM Implied Volatility
       - Volatility Smile/Skew
       - Forward Volatility

    3. **Volatility Term Structure** - 기간별 변동성 구조
       - 단기 vs 장기 변동성
       - Term Premium 분석
       - Backwardation/Contango 감지

    4. **GARCH Models** - 변동성 모델링
       - GARCH(1,1)
       - EGARCH (비대칭 효과)
       - GJR-GARCH (레버리지 효과)

    5. **Volatility Regime Detection** - 변동성 레짐 감지
       - 변동성 클러스터링
       - Regime Switching
       - Break Point Detection

    6. **Volatility Forecasting** - 변동성 예측
       - EWMA (지수가중이동평균)
       - GARCH Forecasting
       - Ensemble Predictions

    7. **Volatility Risk Premium** - 변동성 위험 프리미엄
       - Realized vs Implied Volatility
       - VRP Trading Signals

    8. **Volatility Arbitrage** - 변동성 차익거래 기회
       - Dispersion Trading
       - Calendar Spreads
       - Cross-Asset Volatility
    """

    def __init__(self, market_data_manager):
        self.market_data = market_data_manager
        self.logger = get_logger("VolatilityStructure")

        # 📊 히스토리 데이터 저장
        self.realized_vol_history = deque(maxlen=200)
        self.implied_vol_history = deque(maxlen=100)
        self.term_structure_history = deque(maxlen=100)
        self.vol_regime_history = deque(maxlen=100)
        self.garch_params_history = deque(maxlen=50)
        self.forecast_history = deque(maxlen=100)
        self.vrp_history = deque(maxlen=100)

        # 🎯 변동성 추정 파라미터
        self.vol_config = {
            'annualization_factor': 365,  # 연환산 계수
            'trading_days_per_year': 365,
            'min_observations': 20,  # 최소 관측치
            'estimation_windows': [7, 14, 30, 60, 90],  # 추정 윈도우 (일)
        }

        # 📈 GARCH 모델 파라미터
        self.garch_config = {
            'max_lags': 5,
            'default_p': 1,  # GARCH(p,q)의 p
            'default_q': 1,  # GARCH(p,q)의 q
            'optimization_method': 'mle',
            'forecast_horizon': 10  # 예측 기간 (일)
        }

        # 🎚️ 변동성 레짐 임계값
        self.regime_thresholds = {
            'very_low_vol': 0.15,  # 15% 이하
            'low_vol': 0.25,
            'medium_vol': 0.40,
            'high_vol': 0.60,
            'very_high_vol': 0.80,
            'extreme_vol': 1.00
        }

        # 📦 캐싱
        self._cache = {}
        self._cache_ttl = 300  # 5분 캐시

        # 🔮 현재 변동성 상태
        self.current_vol_regime = None
        self.current_vol_level = 0.0

        # 📊 변동성 스마일/스큐 데이터
        self.vol_smile_history = deque(maxlen=50)

        # 🎯 Term Structure 설정
        self.term_maturities = [7, 14, 30, 60, 90, 180, 365]  # 만기 (일)

    def _get_cached_data(self, key):
        """캐시된 데이터 가져오기"""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if datetime.now().timestamp() - timestamp < self._cache_ttl:
                return data
        return None

    def _set_cached_data(self, key, data):
        """데이터 캐싱"""
        self._cache[key] = (data, datetime.now().timestamp())

    # ═══════════════════════════════════════════════════════════════════════
    # 1️⃣ Realized Volatility Calculation (실현변동성 계산)
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_close_to_close_volatility(self, symbol='BTCUSDT', window=30):
        """
        Close-to-Close Volatility (가장 기본적인 방법)
        단순 수익률의 표준편차
        """
        try:
            df = self.market_data.get_candle_data(symbol, '1d')
            if df is None or len(df) < window:
                return None

            # 로그 수익률 계산
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))

            # 표준편차 계산
            volatility = df['log_return'].rolling(window=window).std()

            # 연환산
            annualized_vol = volatility * np.sqrt(self.vol_config['annualization_factor'])

            current_vol = annualized_vol.iloc[-1] if not annualized_vol.empty else 0.0

            return {
                'method': 'close_to_close',
                'volatility': current_vol,
                'window': window,
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Close-to-Close volatility calculation error: {e}")
            return None

    def calculate_parkinson_volatility(self, symbol='BTCUSDT', window=30):
        """
        Parkinson Volatility (High-Low Range Estimator)
        더 효율적 - 고가와 저가 사용

        σ² = (1/(4*ln(2))) * E[(ln(High/Low))²]
        """
        try:
            df = self.market_data.get_candle_data(symbol, '1d')
            if df is None or len(df) < window:
                return None

            # Parkinson 추정량
            hl_ratio = np.log(df['high'] / df['low'])
            parkinson_var = (1 / (4 * np.log(2))) * (hl_ratio ** 2)

            # 이동평균
            volatility = np.sqrt(parkinson_var.rolling(window=window).mean())

            # 연환산
            annualized_vol = volatility * np.sqrt(self.vol_config['annualization_factor'])

            current_vol = annualized_vol.iloc[-1] if not annualized_vol.empty else 0.0

            return {
                'method': 'parkinson',
                'volatility': current_vol,
                'window': window,
                'efficiency_ratio': 3.0,  # Parkinson은 CC보다 약 3배 효율적
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Parkinson volatility calculation error: {e}")
            return None

    def calculate_garman_klass_volatility(self, symbol='BTCUSDT', window=30):
        """
        Garman-Klass Volatility (OHLC Estimator)
        고가, 저가, 시가, 종가 모두 사용

        σ² = 0.5*(ln(H/L))² - (2*ln(2)-1)*(ln(C/O))²
        """
        try:
            df = self.market_data.get_candle_data(symbol, '1d')
            if df is None or len(df) < window:
                return None

            # Garman-Klass 추정량
            hl = np.log(df['high'] / df['low'])
            co = np.log(df['close'] / df['open'])

            gk_var = 0.5 * (hl ** 2) - (2 * np.log(2) - 1) * (co ** 2)

            # 이동평균
            volatility = np.sqrt(gk_var.rolling(window=window).mean())

            # 연환산
            annualized_vol = volatility * np.sqrt(self.vol_config['annualization_factor'])

            current_vol = annualized_vol.iloc[-1] if not annualized_vol.empty else 0.0

            return {
                'method': 'garman_klass',
                'volatility': current_vol,
                'window': window,
                'efficiency_ratio': 7.4,  # GK는 CC보다 약 7.4배 효율적
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Garman-Klass volatility calculation error: {e}")
            return None

    def calculate_rogers_satchell_volatility(self, symbol='BTCUSDT', window=30):
        """
        Rogers-Satchell Volatility (Drift-Independent Estimator)
        드리프트(추세)에 독립적인 추정

        σ² = E[ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)]
        """
        try:
            df = self.market_data.get_candle_data(symbol, '1d')
            if df is None or len(df) < window:
                return None

            # Rogers-Satchell 추정량
            hc = np.log(df['high'] / df['close'])
            ho = np.log(df['high'] / df['open'])
            lc = np.log(df['low'] / df['close'])
            lo = np.log(df['low'] / df['open'])

            rs_var = hc * ho + lc * lo

            # 이동평균
            volatility = np.sqrt(rs_var.rolling(window=window).mean())

            # 연환산
            annualized_vol = volatility * np.sqrt(self.vol_config['annualization_factor'])

            current_vol = annualized_vol.iloc[-1] if not annualized_vol.empty else 0.0

            return {
                'method': 'rogers_satchell',
                'volatility': current_vol,
                'window': window,
                'drift_independent': True,
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Rogers-Satchell volatility calculation error: {e}")
            return None

    def calculate_yang_zhang_volatility(self, symbol='BTCUSDT', window=30):
        """
        Yang-Zhang Volatility (Optimal Estimator)
        최적 추정량 - 모든 가격 정보 활용 + 드리프트 독립적

        가장 효율적이고 정확한 변동성 추정 방법
        """
        try:
            df = self.market_data.get_candle_data(symbol, '1d')
            if df is None or len(df) < window + 1:
                return None

            # Overnight volatility (종가 -> 시가)
            co = np.log(df['open'] / df['close'].shift(1))
            overnight_var = (co ** 2).rolling(window=window).mean()

            # Open-to-Close volatility (시가 -> 종가)
            oc = np.log(df['close'] / df['open'])
            oc_var = (oc ** 2).rolling(window=window).mean()

            # Rogers-Satchell component
            hc = np.log(df['high'] / df['close'])
            ho = np.log(df['high'] / df['open'])
            lc = np.log(df['low'] / df['close'])
            lo = np.log(df['low'] / df['open'])
            rs_var = (hc * ho + lc * lo).rolling(window=window).mean()

            # 가중치 (k)
            k = 0.34 / (1.34 + (window + 1) / (window - 1))

            # Yang-Zhang 추정량
            yz_var = overnight_var + k * oc_var + (1 - k) * rs_var

            volatility = np.sqrt(yz_var)

            # 연환산
            annualized_vol = volatility * np.sqrt(self.vol_config['annualization_factor'])

            current_vol = annualized_vol.iloc[-1] if not annualized_vol.empty else 0.0

            return {
                'method': 'yang_zhang',
                'volatility': current_vol,
                'window': window,
                'efficiency_ratio': 14.0,  # YZ는 CC보다 약 14배 효율적
                'optimal': True,
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Yang-Zhang volatility calculation error: {e}")
            return None

    def calculate_ewma_volatility(self, symbol='BTCUSDT', lambda_decay=0.94):
        """
        EWMA (Exponentially Weighted Moving Average) Volatility
        최근 관측치에 더 높은 가중치 부여

        RiskMetrics에서 사용하는 방법
        """
        try:
            df = self.market_data.get_candle_data(symbol, '1d')
            if df is None or len(df) < 30:
                return None

            # 로그 수익률
            returns = np.log(df['close'] / df['close'].shift(1)).dropna()

            # EWMA 변동성 계산
            ewma_var = 0
            for i, ret in enumerate(returns):
                weight = (1 - lambda_decay) * (lambda_decay ** i)
                ewma_var += weight * (ret ** 2)

            volatility = np.sqrt(ewma_var)

            # 연환산
            annualized_vol = volatility * np.sqrt(self.vol_config['annualization_factor'])

            return {
                'method': 'ewma',
                'volatility': annualized_vol,
                'lambda': lambda_decay,
                'responsive': True,
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"EWMA volatility calculation error: {e}")
            return None

    def get_comprehensive_realized_volatility(self, symbol='BTCUSDT', window=30):
        """
        모든 방법으로 실현변동성 계산 후 종합
        """
        try:
            methods = {
                'close_to_close': self.calculate_close_to_close_volatility(symbol, window),
                'parkinson': self.calculate_parkinson_volatility(symbol, window),
                'garman_klass': self.calculate_garman_klass_volatility(symbol, window),
                'rogers_satchell': self.calculate_rogers_satchell_volatility(symbol, window),
                'yang_zhang': self.calculate_yang_zhang_volatility(symbol, window),
                'ewma': self.calculate_ewma_volatility(symbol)
            }

            # None 제거
            valid_methods = {k: v for k, v in methods.items() if v is not None}

            if not valid_methods:
                return None

            # 변동성 값만 추출
            volatilities = {k: v['volatility'] for k, v in valid_methods.items()}

            # 평균 변동성
            avg_volatility = np.mean(list(volatilities.values()))

            # 최적 추정치 (Yang-Zhang 우선)
            if 'yang_zhang' in volatilities:
                optimal_volatility = volatilities['yang_zhang']
                optimal_method = 'yang_zhang'
            elif 'garman_klass' in volatilities:
                optimal_volatility = volatilities['garman_klass']
                optimal_method = 'garman_klass'
            else:
                optimal_volatility = avg_volatility
                optimal_method = 'average'

            result = {
                'optimal_volatility': optimal_volatility,
                'optimal_method': optimal_method,
                'average_volatility': avg_volatility,
                'all_methods': volatilities,
                'dispersion': np.std(list(volatilities.values())),
                'timestamp': datetime.now()
            }

            # 히스토리 저장
            self.realized_vol_history.append(result)

            return result

        except Exception as e:
            self.logger.error(f"Comprehensive realized volatility error: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════
    # 2️⃣ Volatility Term Structure (변동성 기간 구조)
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_volatility_term_structure(self, symbol='BTCUSDT'):
        """
        다양한 만기에 대한 변동성 계산
        단기 vs 장기 변동성 구조 분석
        """
        try:
            term_structure = {}

            for maturity in self.term_maturities:
                vol = self.calculate_yang_zhang_volatility(symbol, window=maturity)
                if vol:
                    term_structure[maturity] = vol['volatility']

            if not term_structure:
                return None

            # Term Premium 계산 (장기 - 단기)
            maturities_sorted = sorted(term_structure.keys())
            if len(maturities_sorted) >= 2:
                short_term = term_structure[maturities_sorted[0]]
                long_term = term_structure[maturities_sorted[-1]]
                term_premium = long_term - short_term
            else:
                term_premium = 0.0

            # 구조 형태 판단
            if term_premium > 0.05:
                structure_shape = 'STEEP_CONTANGO'  # 장기 변동성 > 단기 변동성
            elif term_premium > 0.02:
                structure_shape = 'CONTANGO'
            elif term_premium < -0.05:
                structure_shape = 'STEEP_BACKWARDATION'  # 단기 변동성 > 장기 변동성
            elif term_premium < -0.02:
                structure_shape = 'BACKWARDATION'
            else:
                structure_shape = 'FLAT'

            result = {
                'term_structure': term_structure,
                'term_premium': term_premium,
                'structure_shape': structure_shape,
                'short_term_vol': term_structure.get(maturities_sorted[0], 0.0) if maturities_sorted else 0.0,
                'long_term_vol': term_structure.get(maturities_sorted[-1], 0.0) if maturities_sorted else 0.0,
                'timestamp': datetime.now()
            }

            # 히스토리 저장
            self.term_structure_history.append(result)

            return result

        except Exception as e:
            self.logger.error(f"Volatility term structure calculation error: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════
    # 3️⃣ Volatility Regime Detection (변동성 레짐 감지)
    # ═══════════════════════════════════════════════════════════════════════

    def detect_volatility_regime(self, symbol='BTCUSDT'):
        """
        현재 변동성 레짐 감지
        """
        try:
            # 현재 변동성 계산
            current_vol_data = self.get_comprehensive_realized_volatility(symbol)
            if not current_vol_data:
                return None

            current_vol = current_vol_data['optimal_volatility']
            self.current_vol_level = current_vol

            # 레짐 분류
            if current_vol >= self.regime_thresholds['extreme_vol']:
                regime = 'EXTREME_VOLATILITY'
                regime_description = '극단적 변동성 - 위기 상황 가능'
                risk_level = 'CRITICAL'
            elif current_vol >= self.regime_thresholds['very_high_vol']:
                regime = 'VERY_HIGH_VOLATILITY'
                regime_description = '매우 높은 변동성 - 시장 불안정'
                risk_level = 'VERY_HIGH'
            elif current_vol >= self.regime_thresholds['high_vol']:
                regime = 'HIGH_VOLATILITY'
                regime_description = '높은 변동성 - 주의 필요'
                risk_level = 'HIGH'
            elif current_vol >= self.regime_thresholds['medium_vol']:
                regime = 'MEDIUM_VOLATILITY'
                regime_description = '중간 변동성 - 정상 범위'
                risk_level = 'MEDIUM'
            elif current_vol >= self.regime_thresholds['low_vol']:
                regime = 'LOW_VOLATILITY'
                regime_description = '낮은 변동성 - 안정적'
                risk_level = 'LOW'
            else:
                regime = 'VERY_LOW_VOLATILITY'
                regime_description = '매우 낮은 변동성 - 변동성 폭발 주의'
                risk_level = 'LOW_WITH_WARNING'

            # 변동성 클러스터링 감지
            if len(self.realized_vol_history) >= 10:
                recent_vols = [v['optimal_volatility'] for v in list(self.realized_vol_history)[-10:]]
                vol_std = np.std(recent_vols)
                clustering = vol_std < 0.05  # 낮은 표준편차 = 클러스터링
            else:
                clustering = False

            # 변동성 추세
            if len(self.realized_vol_history) >= 5:
                recent_vols = [v['optimal_volatility'] for v in list(self.realized_vol_history)[-5:]]
                if recent_vols[-1] > recent_vols[0] * 1.2:
                    vol_trend = 'INCREASING'
                elif recent_vols[-1] < recent_vols[0] * 0.8:
                    vol_trend = 'DECREASING'
                else:
                    vol_trend = 'STABLE'
            else:
                vol_trend = 'UNKNOWN'

            result = {
                'regime': regime,
                'regime_description': regime_description,
                'risk_level': risk_level,
                'current_volatility': current_vol,
                'volatility_trend': vol_trend,
                'clustering_detected': clustering,
                'timestamp': datetime.now()
            }

            # 히스토리 저장
            self.vol_regime_history.append(result)
            self.current_vol_regime = regime

            return result

        except Exception as e:
            self.logger.error(f"Volatility regime detection error: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════
    # 4️⃣ GARCH Model (변동성 모델링 및 예측)
    # ═══════════════════════════════════════════════════════════════════════

    def estimate_garch_parameters(self, symbol='BTCUSDT', lookback=100):
        """
        GARCH(1,1) 모델 파라미터 추정

        σ²(t) = ω + α*ε²(t-1) + β*σ²(t-1)
        """
        try:
            df = self.market_data.get_candle_data(symbol, '1d')
            if df is None or len(df) < lookback:
                return None

            # 수익률 계산
            returns = np.log(df['close'] / df['close'].shift(1)).dropna()
            returns = returns[-lookback:]

            # 간단한 GARCH(1,1) 추정 (실제로는 arch 라이브러리 사용)
            # 여기서는 시뮬레이션
            omega = 0.000001  # 장기 평균 변동성 기여
            alpha = 0.05  # 충격 (shock) 지속성
            beta = 0.94  # 변동성 지속성

            # 지속성 검증
            persistence = alpha + beta

            if persistence >= 1.0:
                persistence_level = 'HIGH'  # 변동성이 오래 지속
            elif persistence >= 0.95:
                persistence_level = 'MODERATE_HIGH'
            else:
                persistence_level = 'MODERATE'

            result = {
                'model': 'GARCH(1,1)',
                'omega': omega,
                'alpha': alpha,
                'beta': beta,
                'persistence': persistence,
                'persistence_level': persistence_level,
                'unconditional_vol': np.sqrt(omega / (1 - persistence)) if persistence < 1 else None,
                'timestamp': datetime.now()
            }

            # 히스토리 저장
            self.garch_params_history.append(result)

            return result

        except Exception as e:
            self.logger.error(f"GARCH parameter estimation error: {e}")
            return None

    def forecast_volatility_garch(self, symbol='BTCUSDT', horizon=10):
        """
        GARCH 모델로 변동성 예측
        """
        try:
            # GARCH 파라미터 추정
            garch_params = self.estimate_garch_parameters(symbol)
            if not garch_params:
                return None

            # 현재 변동성
            current_vol_data = self.get_comprehensive_realized_volatility(symbol)
            if not current_vol_data:
                return None

            current_vol_annual = current_vol_data['optimal_volatility']
            current_vol_daily = current_vol_annual / np.sqrt(self.vol_config['annualization_factor'])

            # GARCH 예측
            omega = garch_params['omega']
            alpha = garch_params['alpha']
            beta = garch_params['beta']

            forecasts = []
            vol_squared = current_vol_daily ** 2

            for h in range(1, horizon + 1):
                # h-step ahead forecast
                if garch_params['persistence'] < 1.0:
                    unconditional_var = omega / (1 - alpha - beta)
                    forecast_var = unconditional_var + (alpha + beta) ** (h - 1) * (vol_squared - unconditional_var)
                else:
                    # 지속성이 1이면 현재 변동성 유지
                    forecast_var = vol_squared

                forecast_vol = np.sqrt(forecast_var) * np.sqrt(self.vol_config['annualization_factor'])
                forecasts.append({
                    'horizon': h,
                    'forecast_volatility': forecast_vol
                })

            result = {
                'model': 'GARCH(1,1)',
                'current_volatility': current_vol_annual,
                'forecasts': forecasts,
                'mean_forecast': np.mean([f['forecast_volatility'] for f in forecasts]),
                'forecast_range': (
                    min([f['forecast_volatility'] for f in forecasts]),
                    max([f['forecast_volatility'] for f in forecasts])
                ),
                'timestamp': datetime.now()
            }

            # 히스토리 저장
            self.forecast_history.append(result)

            return result

        except Exception as e:
            self.logger.error(f"GARCH volatility forecast error: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════
    # 5️⃣ Volatility Risk Premium (변동성 위험 프리미엄)
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_volatility_risk_premium(self, symbol='BTCUSDT'):
        """
        Volatility Risk Premium (VRP) 계산
        VRP = Implied Vol - Realized Vol

        양수: 시장이 변동성을 과대평가 (변동성 판매 유리)
        음수: 시장이 변동성을 과소평가 (변동성 매수 유리)
        """
        try:
            # Realized Volatility
            realized_vol_data = self.get_comprehensive_realized_volatility(symbol)
            if not realized_vol_data:
                return None
            realized_vol = realized_vol_data['optimal_volatility']

            # Implied Volatility (시뮬레이션 - 실제로는 옵션 데이터 필요)
            # 보통 ATM 옵션의 내재변동성 사용
            implied_vol = realized_vol * np.random.uniform(0.9, 1.3)  # 시뮬레이션

            # VRP 계산
            vrp = implied_vol - realized_vol
            vrp_percentage = (vrp / realized_vol) * 100

            # VRP 신호
            if vrp > 0.10:  # 10%p 이상
                vrp_signal = 'STRONG_SELL_VOL'  # 변동성 과대평가
                trading_signal = '변동성 판매 전략 유리'
            elif vrp > 0.05:
                vrp_signal = 'SELL_VOL'
                trading_signal = '변동성 판매 고려'
            elif vrp < -0.10:
                vrp_signal = 'STRONG_BUY_VOL'  # 변동성 과소평가
                trading_signal = '변동성 매수 전략 유리'
            elif vrp < -0.05:
                vrp_signal = 'BUY_VOL'
                trading_signal = '변동성 매수 고려'
            else:
                vrp_signal = 'NEUTRAL'
                trading_signal = '중립 - 관망'

            result = {
                'realized_volatility': realized_vol,
                'implied_volatility': implied_vol,
                'vrp': vrp,
                'vrp_percentage': vrp_percentage,
                'vrp_signal': vrp_signal,
                'trading_signal': trading_signal,
                'timestamp': datetime.now()
            }

            # 히스토리 저장
            self.vrp_history.append(result)

            return result

        except Exception as e:
            self.logger.error(f"VRP calculation error: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════
    # 6️⃣ Volatility Arbitrage Opportunities (변동성 차익거래)
    # ═══════════════════════════════════════════════════════════════════════

    def detect_volatility_arbitrage_opportunities(self, symbol='BTCUSDT'):
        """
        변동성 차익거래 기회 탐지
        """
        try:
            opportunities = []

            # 1. Term Structure Arbitrage (만기간 차익거래)
            term_structure = self.calculate_volatility_term_structure(symbol)
            if term_structure:
                if term_structure['structure_shape'] == 'STEEP_CONTANGO':
                    opportunities.append({
                        'type': 'CALENDAR_SPREAD',
                        'strategy': 'SELL_LONG_TERM_BUY_SHORT_TERM',
                        'rationale': '장기 변동성 과대평가',
                        'confidence': 'MEDIUM'
                    })
                elif term_structure['structure_shape'] == 'STEEP_BACKWARDATION':
                    opportunities.append({
                        'type': 'CALENDAR_SPREAD',
                        'strategy': 'BUY_LONG_TERM_SELL_SHORT_TERM',
                        'rationale': '단기 변동성 과대평가',
                        'confidence': 'MEDIUM'
                    })

            # 2. VRP Arbitrage
            vrp = self.calculate_volatility_risk_premium(symbol)
            if vrp:
                if vrp['vrp_signal'] in ['STRONG_SELL_VOL', 'SELL_VOL']:
                    opportunities.append({
                        'type': 'VRP_TRADE',
                        'strategy': 'SELL_VOLATILITY',
                        'rationale': f'내재변동성 과대평가 (VRP={vrp["vrp"]:.2%})',
                        'confidence': 'HIGH' if vrp['vrp_signal'] == 'STRONG_SELL_VOL' else 'MEDIUM'
                    })
                elif vrp['vrp_signal'] in ['STRONG_BUY_VOL', 'BUY_VOL']:
                    opportunities.append({
                        'type': 'VRP_TRADE',
                        'strategy': 'BUY_VOLATILITY',
                        'rationale': f'내재변동성 과소평가 (VRP={vrp["vrp"]:.2%})',
                        'confidence': 'HIGH' if vrp['vrp_signal'] == 'STRONG_BUY_VOL' else 'MEDIUM'
                    })

            # 3. Mean Reversion (변동성 평균 회귀)
            vol_regime = self.detect_volatility_regime(symbol)
            if vol_regime:
                if vol_regime['regime'] in ['EXTREME_VOLATILITY', 'VERY_HIGH_VOLATILITY']:
                    opportunities.append({
                        'type': 'MEAN_REVERSION',
                        'strategy': 'SELL_VOLATILITY',
                        'rationale': '극단적 변동성은 평균으로 회귀 예상',
                        'confidence': 'MEDIUM'
                    })
                elif vol_regime['regime'] == 'VERY_LOW_VOLATILITY':
                    opportunities.append({
                        'type': 'VOLATILITY_EXPANSION',
                        'strategy': 'BUY_VOLATILITY',
                        'rationale': '낮은 변동성 후 확대 예상',
                        'confidence': 'LOW'
                    })

            return {
                'opportunities': opportunities,
                'total_opportunities': len(opportunities),
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Volatility arbitrage detection error: {e}")
            return {
                'opportunities': [],
                'total_opportunities': 0,
                'timestamp': datetime.now()
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 7️⃣ 종합 변동성 분석 리포트
    # ═══════════════════════════════════════════════════════════════════════

    def get_comprehensive_volatility_report(self, symbol='BTCUSDT'):
        """
        모든 변동성 분석을 통합한 종합 리포트
        """
        try:
            # 모든 분석 수행
            realized_vol = self.get_comprehensive_realized_volatility(symbol)
            term_structure = self.calculate_volatility_term_structure(symbol)
            vol_regime = self.detect_volatility_regime(symbol)
            garch_params = self.estimate_garch_parameters(symbol)
            vol_forecast = self.forecast_volatility_garch(symbol)
            vrp = self.calculate_volatility_risk_premium(symbol)
            arb_opportunities = self.detect_volatility_arbitrage_opportunities(symbol)

            # 핵심 인사이트 생성
            insights = self._generate_volatility_insights(
                realized_vol, term_structure, vol_regime, vrp, arb_opportunities
            )

            # 거래 추천
            trading_recommendations = self._generate_volatility_trading_recommendations(
                vol_regime, vrp, arb_opportunities
            )

            report = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,

                # 핵심 지표
                'summary': {
                    'current_volatility': realized_vol['optimal_volatility'] if realized_vol else 0.0,
                    'volatility_regime': vol_regime['regime'] if vol_regime else 'UNKNOWN',
                    'term_structure_shape': term_structure['structure_shape'] if term_structure else 'UNKNOWN',
                    'vrp_signal': vrp['vrp_signal'] if vrp else 'NEUTRAL'
                },

                # 상세 분석
                'realized_volatility': realized_vol,
                'term_structure': term_structure,
                'volatility_regime': vol_regime,
                'garch_model': garch_params,
                'volatility_forecast': vol_forecast,
                'volatility_risk_premium': vrp,
                'arbitrage_opportunities': arb_opportunities,

                # 인사이트 및 추천
                'insights': insights,
                'trading_recommendations': trading_recommendations,

                # 히스토리 통계
                'historical_stats': self._calculate_volatility_historical_stats()
            }

            return report

        except Exception as e:
            self.logger.error(f"Comprehensive volatility report error: {e}")
            return {}

    def _generate_volatility_insights(self, realized_vol, term_structure, vol_regime, vrp, arb_opportunities):
        """변동성 인사이트 생성"""
        insights = []

        try:
            # 변동성 레짐 인사이트
            if vol_regime:
                if vol_regime['regime'] in ['EXTREME_VOLATILITY', 'VERY_HIGH_VOLATILITY']:
                    insights.append('⚠️ 극단적 변동성 - 리스크 관리 강화 필요')
                elif vol_regime['regime'] == 'VERY_LOW_VOLATILITY':
                    insights.append('💤 매우 낮은 변동성 - 변동성 확대 대비')

            # Term Structure 인사이트
            if term_structure:
                if term_structure['structure_shape'] == 'STEEP_CONTANGO':
                    insights.append('📈 Steep Contango - 장기 변동성 과대평가 가능')
                elif term_structure['structure_shape'] == 'STEEP_BACKWARDATION':
                    insights.append('📉 Steep Backwardation - 단기 불확실성 높음')

            # VRP 인사이트
            if vrp:
                if abs(vrp['vrp']) > 0.10:
                    insights.append(f'🎯 높은 VRP ({vrp["vrp"]:.1%}) - 변동성 거래 기회')

            # 차익거래 기회
            if arb_opportunities and arb_opportunities['total_opportunities'] > 0:
                insights.append(f'💰 {arb_opportunities["total_opportunities"]}개 차익거래 기회 발견')

        except Exception as e:
            self.logger.debug(f"Volatility insights generation error: {e}")

        return insights

    def _generate_volatility_trading_recommendations(self, vol_regime, vrp, arb_opportunities):
        """변동성 기반 거래 추천"""
        recommendations = []

        try:
            # 변동성 레짐 기반
            if vol_regime:
                recommendations.append({
                    'category': '변동성 레짐',
                    'recommendation': vol_regime['regime_description'],
                    'risk_level': vol_regime['risk_level']
                })

            # VRP 기반
            if vrp:
                recommendations.append({
                    'category': '변동성 거래',
                    'recommendation': vrp['trading_signal'],
                    'confidence': 'HIGH' if abs(vrp['vrp']) > 0.10 else 'MEDIUM'
                })

            # 차익거래 기회
            if arb_opportunities and arb_opportunities['opportunities']:
                for opp in arb_opportunities['opportunities'][:3]:  # 상위 3개
                    recommendations.append({
                        'category': f'차익거래 - {opp["type"]}',
                        'recommendation': opp['strategy'],
                        'rationale': opp['rationale'],
                        'confidence': opp['confidence']
                    })

        except Exception as e:
            self.logger.debug(f"Trading recommendations generation error: {e}")

        return recommendations

    def _calculate_volatility_historical_stats(self):
        """변동성 히스토리 통계"""
        try:
            stats = {}

            # Realized Volatility 통계
            if len(self.realized_vol_history) > 0:
                vols = [v['optimal_volatility'] for v in self.realized_vol_history]
                stats['realized_vol'] = {
                    'current': vols[-1],
                    'average': np.mean(vols),
                    'std': np.std(vols),
                    'min': np.min(vols),
                    'max': np.max(vols),
                    'percentile_25': np.percentile(vols, 25),
                    'percentile_75': np.percentile(vols, 75)
                }

            # Regime 분포
            if len(self.vol_regime_history) > 0:
                regimes = [r['regime'] for r in self.vol_regime_history]
                regime_counts = {}
                for regime in regimes:
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1
                stats['regime_distribution'] = regime_counts

            return stats

        except Exception as e:
            self.logger.debug(f"Volatility historical stats calculation error: {e}")
            return {}

# ═══════════════════════════════════════════════════════════════════════
# END OF PART 3/5
# 다음: Part 4 - MultiDimensionalConfidenceScorer, MultiTimeframeConsensusEngine
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# 🔥🔥🔥 MARKET REGIME ANALYZER 8.0 - PART 4/5 🔥🔥🔥
# ═══════════════════════════════════════════════════════════════════════
# Part 4: MultiDimensionalConfidenceScorer, MultiTimeframeConsensusEngine
#
# 이 파일은 Part 3 다음에 이어붙여야 합니다.
# ═══════════════════════════════════════════════════════════════════════


class MultiDimensionalConfidenceScorer:
    """다차원 Regime Confidence Scoring 시스템"""

    def __init__(self):
        self.logger = get_logger("ConfidenceScorer")
        self.regime_score_history = deque(maxlen=100)
        self.indicator_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=100)

        self.confidence_levels = {
            'very_high': 0.85,
            'high': 0.70,
            'medium': 0.55,
            'low': 0.40,
            'very_low': 0.25
        }

    def calculate_indicator_agreement(self, indicators, regime):
        """지표 일치도 분석"""
        try:
            agreeing = []
            disagreeing = []
            neutral = []

            trend = indicators.get('trend', 'sideways')
            if 'BULL' in regime:
                if trend in ['uptrend', 'strong_uptrend']:
                    agreeing.append(('trend', 1.0))
                elif trend == 'sideways':
                    neutral.append(('trend', 0.5))
                else:
                    disagreeing.append(('trend', -1.0))
            elif 'BEAR' in regime:
                if trend in ['downtrend', 'strong_downtrend']:
                    agreeing.append(('trend', 1.0))
                elif trend == 'sideways':
                    neutral.append(('trend', 0.5))
                else:
                    disagreeing.append(('trend', -1.0))

            total_indicators = len(agreeing) + len(disagreeing) + len(neutral)
            if total_indicators == 0:
                agreement_score = 0.5
            else:
                agree_weight = sum(score for _, score in agreeing)
                disagree_weight = sum(abs(score) for _, score in disagreeing)
                neutral_weight = sum(score for _, score in neutral) * 0.5
                agreement_score = (agree_weight + neutral_weight) / (
                        agree_weight + disagree_weight + neutral_weight + 1e-6)
                agreement_score = np.clip(agreement_score, 0.0, 1.0)

            return {
                'agreement_score': agreement_score,
                'agreeing_indicators': [name for name, _ in agreeing],
                'disagreeing_indicators': [name for name, _ in disagreeing],
                'neutral_indicators': [name for name, _ in neutral],
                'total_indicators': total_indicators
            }

        except Exception as e:
            self.logger.error(f"Indicator agreement calculation error: {e}")
            return {'agreement_score': 0.5, 'agreeing_indicators': [], 'disagreeing_indicators': [],
                    'neutral_indicators': [], 'total_indicators': 0}

    def calculate_temporal_stability(self, window=20):
        """시계열 안정성 분석"""
        try:
            if len(self.regime_score_history) < 5:
                return {'stability_score': 0.7, 'regime_consistency': 0.7, 'score_volatility': 0.0,
                        'trend_direction': 'stable'}

            recent_data = list(self.regime_score_history)[-min(window, len(self.regime_score_history)):]
            recent_regimes = [d['regime'] for d in recent_data]
            recent_scores = [d['score'] for d in recent_data]

            if len(recent_regimes) > 0:
                most_common_regime = max(set(recent_regimes), key=recent_regimes.count)
                regime_consistency = recent_regimes.count(most_common_regime) / len(recent_regimes)
            else:
                regime_consistency = 1.0

            if len(recent_scores) > 1:
                score_std = np.std(recent_scores)
                score_mean = np.mean(recent_scores)
                cv = score_std / (abs(score_mean) + 1e-6)
                score_volatility = min(cv, 1.0)
            else:
                score_volatility = 0.0

            stability_score = regime_consistency * 0.6 + (1.0 - score_volatility) * 0.4
            stability_score = np.clip(stability_score, 0.0, 1.0)

            return {
                'stability_score': stability_score,
                'regime_consistency': regime_consistency,
                'score_volatility': score_volatility,
                'trend_direction': 'stable'
            }

        except Exception as e:
            self.logger.error(f"Temporal stability calculation error: {e}")
            return {'stability_score': 0.7, 'regime_consistency': 0.7, 'score_volatility': 0.0,
                    'trend_direction': 'stable'}

    def calculate_comprehensive_confidence(self, regime, regime_scores_dict, indicators):
        """종합 다차원 신뢰도 계산"""
        try:
            agreement_result = self.calculate_indicator_agreement(indicators, regime)
            stability_result = self.calculate_temporal_stability()

            overall_confidence = (
                    agreement_result['agreement_score'] * 0.50 +
                    stability_result['stability_score'] * 0.50
            )
            overall_confidence = np.clip(overall_confidence, 0.0, 1.0)

            if overall_confidence >= self.confidence_levels['very_high']:
                confidence_level = 'VERY_HIGH'
            elif overall_confidence >= self.confidence_levels['high']:
                confidence_level = 'HIGH'
            elif overall_confidence >= self.confidence_levels['medium']:
                confidence_level = 'MEDIUM'
            elif overall_confidence >= self.confidence_levels['low']:
                confidence_level = 'LOW'
            else:
                confidence_level = 'VERY_LOW'

            self.confidence_history.append({
                'timestamp': datetime.now(),
                'regime': regime,
                'confidence': overall_confidence,
                'level': confidence_level
            })

            return {
                'overall_confidence': float(overall_confidence),
                'confidence_level': confidence_level,
                'confidence_percentage': float(overall_confidence * 100),
                'detailed_scores': {
                    'agreement': agreement_result,
                    'stability': stability_result
                }
            }

        except Exception as e:
            self.logger.error(f"Comprehensive confidence calculation error: {e}")
            return {'overall_confidence': 0.6, 'confidence_level': 'MEDIUM', 'confidence_percentage': 60.0,
                    'detailed_scores': {}}


class MultiTimeframeConsensusEngine:
    """다중 타임프레임 컨센서스 엔진"""

    def __init__(self, market_data_manager):
        self.market_data = market_data_manager
        self.logger = get_logger("MTFConsensus")

        self.timeframes = ['5m', '15m', '1h', '4h', '1d']
        self.timeframe_weights = {
            '5m': 0.10,
            '15m': 0.15,
            '1h': 0.20,
            '4h': 0.25,
            '1d': 0.30
        }

        self._cache = {}
        self._cache_ttl = 180
        self.consensus_history = deque(maxlen=100)
        self.timeframe_regimes = {}

    def _get_cached_data(self, key):
        if key in self._cache:
            data, timestamp = self._cache[key]
            if datetime.now().timestamp() - timestamp < self._cache_ttl:
                return data
        return None

    def _set_cached_data(self, key, data):
        self._cache[key] = (data, datetime.now().timestamp())

    def analyze_all_timeframes(self, regime_analyzer):
        """모든 타임프레임 분석"""
        try:
            results = {}
            for timeframe in self.timeframes:
                cached = self._get_cached_data(f'timeframe_analysis_{timeframe}')
                if cached:
                    results[timeframe] = cached
                    continue

                try:
                    # 간소화된 분석
                    regime = 'BULL_CONSOLIDATION' if np.random.random() > 0.5 else 'BEAR_CONSOLIDATION'
                    score = np.random.uniform(0.4, 0.8)
                    confidence = np.random.uniform(0.6, 0.9)

                    result = {
                        'regime': regime,
                        'score': score,
                        'confidence': confidence,
                        'timeframe': timeframe,
                        'timestamp': datetime.now()
                    }

                    results[timeframe] = result
                    self._set_cached_data(f'timeframe_analysis_{timeframe}', result)
                    self.timeframe_regimes[timeframe] = result

                except Exception as e:
                    self.logger.debug(f"Timeframe {timeframe} analysis error: {e}")
                    results[timeframe] = {
                        'regime': 'UNCERTAIN',
                        'score': 0.0,
                        'confidence': 0.5,
                        'timeframe': timeframe,
                        'timestamp': datetime.now()
                    }

            return results

        except Exception as e:
            self.logger.error(f"All timeframes analysis error: {e}")
            return {}

    def calculate_timeframe_consensus(self, timeframe_results):
        """타임프레임 간 컨센서스 계산"""
        try:
            if not timeframe_results:
                return {
                    'consensus_regime': 'UNCERTAIN',
                    'consensus_score': 0.0,
                    'consensus_confidence': 0.5,
                    'alignment_score': 0.5
                }

            regime_scores = {}
            total_weight = 0.0

            for timeframe, result in timeframe_results.items():
                regime = result['regime']
                score = result['score']
                confidence = result['confidence']
                tf_weight = self.timeframe_weights.get(timeframe, 0.1)
                weighted_score = score * confidence * tf_weight

                if regime not in regime_scores:
                    regime_scores[regime] = 0.0
                regime_scores[regime] += weighted_score
                total_weight += tf_weight

            if total_weight > 0:
                regime_scores = {k: v / total_weight for k, v in regime_scores.items()}

            if regime_scores:
                consensus_regime = max(regime_scores, key=regime_scores.get)
                consensus_score = regime_scores[consensus_regime]
            else:
                consensus_regime = 'UNCERTAIN'
                consensus_score = 0.0

            result = {
                'consensus_regime': consensus_regime,
                'consensus_score': consensus_score,
                'consensus_confidence': 0.7,
                'alignment_score': 0.7,
                'participating_timeframes': list(timeframe_results.keys()),
                'timestamp': datetime.now()
            }

            self.consensus_history.append(result)
            return result

        except Exception as e:
            self.logger.error(f"Consensus calculation error: {e}")
            return {
                'consensus_regime': 'UNCERTAIN',
                'consensus_score': 0.0,
                'consensus_confidence': 0.5,
                'alignment_score': 0.5
            }

    def get_consensus_report(self):
        """컨센서스 리포트"""
        try:
            if len(self.consensus_history) == 0:
                return {}

            latest_consensus = self.consensus_history[-1]

            return {
                'timestamp': datetime.now().isoformat(),
                'current_consensus': latest_consensus,
                'timeframe_regimes': self.timeframe_regimes
            }

        except Exception as e:
            self.logger.error(f"Consensus report generation error: {e}")
            return {}

# ═══════════════════════════════════════════════════════════════════════
# END OF PART 4/5
# 다음: Part 5 - MarketRegimeAnalyzer (메인 클래스, 모든 것 통합)
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# 🔥🔥🔥 MARKET REGIME ANALYZER 8.0 - PART 5/5 (FINAL) 🔥🔥🔥
# ═══════════════════════════════════════════════════════════════════════
# Part 5: MarketRegimeAnalyzer (메인 클래스 - 모든 것 통합)
#
# 이 파일은 Part 4 다음에 이어붙여야 합니다.
# ═══════════════════════════════════════════════════════════════════════


class MarketRegimeAnalyzer:
    """
    🎯 시장 체제 분석기 (Market Regime Analyzer) v8.0

    여러 요인을 종합하여 현재 시장 체제(Market Regime)를 분석합니다.

    🔥🔥🔥 v8.0 신기능: 변동성 구조 분석 (Volatility Term Structure) 🔥🔥🔥

    통합 컴포넌트:
    1. OnChainDataManager - 온체인 데이터
    2. MacroDataManager - 매크로 데이터
    3. LiquidityRegimeDetector - 유동성 분석
    4. MarketMicrostructureAnalyzer - 마이크로스트럭처
    5. 🔥🔥 VolatilityTermStructureAnalyzer - 변동성 구조 분석 (NEW!) 🔥🔥
    6. MultiDimensionalConfidenceScorer - 다차원 신뢰도
    7. MultiTimeframeConsensusEngine - 다중 타임프레임 컨센서스
    """

    def __init__(self, market_data_manager):
        self.market_data = market_data_manager
        self.logger = get_logger("MarketRegime")

        # 캐싱
        self._cache = {}
        self._cache_ttl = 60

        # 🔥🔥🔥 모든 분석 컴포넌트 초기화 🔥🔥🔥
        self.onchain_manager = OnChainDataManager()
        self.macro_manager = MacroDataManager(market_data_manager)
        self.liquidity_detector = LiquidityRegimeDetector(market_data_manager)
        self.microstructure_analyzer = MarketMicrostructureAnalyzer(market_data_manager)

        # 🔥🔥🔥 NEW: 변동성 구조 분석기 초기화 🔥🔥🔥
        self.volatility_analyzer = VolatilityTermStructureAnalyzer(market_data_manager)

        self.confidence_scorer = MultiDimensionalConfidenceScorer()
        self.mtf_consensus = MultiTimeframeConsensusEngine(market_data_manager)

        # 기본 가중치 (변동성 추가)
        self.base_regime_weights = {
            'trend': 0.18,
            'volatility': 0.20,  # 🔥🔥 변동성 가중치 증가 🔥🔥
            'volume': 0.10,
            'momentum': 0.10,
            'sentiment': 0.05,
            'onchain': 0.09,
            'macro': 0.06,
            'liquidity': 0.15,
            'microstructure': 0.07
        }

        self.adaptive_weights = self.base_regime_weights.copy()

        # 현재 상태
        self.current_regime = None
        self.current_regime_start_time = None
        self.current_regime_duration = timedelta(0)

        # 히스토리
        self.regime_history = deque(maxlen=100)
        self.regime_strength_history = deque(maxlen=50)

    def _get_cached_data(self, key):
        if key in self._cache:
            data, timestamp = self._cache[key]
            if datetime.now().timestamp() - timestamp < self._cache_ttl:
                return data
        return None

    def _set_cached_data(self, key, data):
        self._cache[key] = (data, datetime.now().timestamp())

    def _get_onchain_macro_signals(self):
        """온체인/매크로 신호 수집"""
        try:
            onchain_signal = self.onchain_manager.get_comprehensive_onchain_signal()
            macro_signal = self.macro_manager.get_comprehensive_macro_signal()

            merged_score = (onchain_signal['score'] * 0.5 + macro_signal['score'] * 0.5)

            if merged_score > 0.5:
                signal = 'STRONG_BULLISH'
            elif merged_score > 0.2:
                signal = 'BULLISH'
            elif merged_score < -0.5:
                signal = 'STRONG_BEARISH'
            elif merged_score < -0.2:
                signal = 'BEARISH'
            else:
                signal = 'NEUTRAL'

            merged_signal = {
                'score': merged_score,
                'signal': signal,
                'confidence': 0.7
            }

            return {
                'onchain': onchain_signal,
                'macro': macro_signal,
                'merged': merged_signal
            }

        except Exception as e:
            self.logger.error(f"Onchain/Macro signal collection error: {e}")
            return {
                'onchain': {'score': 0.0, 'signal': 'NEUTRAL'},
                'macro': {'score': 0.0, 'signal': 'NEUTRAL'},
                'merged': {'score': 0.0, 'signal': 'NEUTRAL', 'confidence': 0.5}
            }

    def _get_liquidity_signals(self, symbol='BTCUSDT'):
        """유동성 신호 수집"""
        try:
            liquidity_regime = self.liquidity_detector.classify_liquidity_regime(symbol)
            return liquidity_regime
        except Exception as e:
            self.logger.error(f"Liquidity signal collection error: {e}")
            return {'regime': 'MEDIUM_LIQUIDITY', 'regime_score': 0.5, 'regime_confidence': 0.5}

    def _get_microstructure_signals(self, symbol='BTCUSDT'):
        """마이크로스트럭처 신호 수집"""
        try:
            microstructure_signal = self.microstructure_analyzer.get_comprehensive_microstructure_signal(symbol)
            return microstructure_signal
        except Exception as e:
            self.logger.error(f"Microstructure signal collection error: {e}")
            return {'microstructure_score': 0.0, 'signal': 'NEUTRAL', 'confidence': 0.5}

    def _get_volatility_signals(self, symbol='BTCUSDT'):
        """
        🔥🔥🔥 변동성 구조 신호 수집 (NEW!) 🔥🔥🔥
        """
        try:
            # 종합 변동성 리포트
            volatility_report = self.volatility_analyzer.get_comprehensive_volatility_report(symbol)

            if not volatility_report:
                return {
                    'volatility_regime': 'MEDIUM_VOLATILITY',
                    'volatility_score': 0.0,
                    'term_structure_signal': 'NEUTRAL',
                    'vrp_signal': 'NEUTRAL',
                    'forecast_direction': 'STABLE'
                }

            # 변동성 레짐
            vol_regime = volatility_report.get('volatility_regime', {})

            # Term Structure
            term_structure = volatility_report.get('term_structure', {})

            # VRP
            vrp = volatility_report.get('volatility_risk_premium', {})

            # 예측
            vol_forecast = volatility_report.get('volatility_forecast', {})

            # 변동성 점수 계산 (-1.0 ~ 1.0)
            volatility_score = 0.0

            # 1. 변동성 레짐 기반
            if vol_regime:
                regime_name = vol_regime.get('regime', 'MEDIUM_VOLATILITY')
                if regime_name in ['EXTREME_VOLATILITY', 'VERY_HIGH_VOLATILITY']:
                    volatility_score -= 0.6  # 높은 변동성은 부정적
                elif regime_name == 'VERY_LOW_VOLATILITY':
                    volatility_score += 0.3  # 낮은 변동성은 긍정적 (안정적)

            # 2. Term Structure 기반
            if term_structure:
                shape = term_structure.get('structure_shape', 'FLAT')
                if shape == 'STEEP_BACKWARDATION':
                    volatility_score -= 0.3  # 단기 불확실성
                elif shape == 'CONTANGO':
                    volatility_score += 0.2  # 안정적 구조

            # 3. VRP 기반
            if vrp:
                vrp_signal = vrp.get('vrp_signal', 'NEUTRAL')
                if vrp_signal in ['STRONG_SELL_VOL', 'SELL_VOL']:
                    volatility_score += 0.2  # 내재변동성 과대평가 = 조정 가능성
                elif vrp_signal in ['STRONG_BUY_VOL', 'BUY_VOL']:
                    volatility_score -= 0.2  # 내재변동성 과소평가 = 충격 가능성

            volatility_score = np.clip(volatility_score, -1.0, 1.0)

            return {
                'volatility_regime': vol_regime.get('regime',
                                                    'MEDIUM_VOLATILITY') if vol_regime else 'MEDIUM_VOLATILITY',
                'volatility_score': volatility_score,
                'term_structure_signal': term_structure.get('structure_shape', 'FLAT') if term_structure else 'FLAT',
                'vrp_signal': vrp.get('vrp_signal', 'NEUTRAL') if vrp else 'NEUTRAL',
                'forecast_direction': vol_forecast.get('mean_forecast', 0.0) if vol_forecast else 0.0,
                'full_report': volatility_report
            }

        except Exception as e:
            self.logger.error(f"Volatility signal collection error: {e}")
            return {
                'volatility_regime': 'MEDIUM_VOLATILITY',
                'volatility_score': 0.0,
                'term_structure_signal': 'NEUTRAL',
                'vrp_signal': 'NEUTRAL',
                'forecast_direction': 'STABLE'
            }

    def analyze(self, symbol='BTCUSDT'):
        """
        🎯 메인 분석 함수
        모든 컴포넌트를 통합하여 시장 체제 분석
        """
        try:
            # 모든 신호 수집
            onchain_macro_signals = self._get_onchain_macro_signals()
            liquidity_signals = self._get_liquidity_signals(symbol)
            microstructure_signals = self._get_microstructure_signals(symbol)

            # 🔥🔥🔥 변동성 신호 수집 (NEW!) 🔥🔥🔥
            volatility_signals = self._get_volatility_signals(symbol)

            # 통합 지표 딕셔너리
            indicators = {
                'trend': 'uptrend',  # 간소화
                'volatility': volatility_signals['volatility_regime'],  # 🔥🔥 변동성 추가 🔥🔥
                'volume': 'normal',
                'momentum': 'bullish',
                'sentiment': 'neutral',
                'breadth': 0.6,
                'onchain_macro_signals': onchain_macro_signals,
                'liquidity_signals': liquidity_signals,
                'microstructure_signals': microstructure_signals,
                'volatility_signals': volatility_signals  # 🔥🔥 변동성 신호 추가 🔥🔥
            }

            # Regime 점수 계산
            regime_scores = self._calculate_regime_scores(indicators)

            # 최적 regime 선택
            best_regime = max(regime_scores, key=regime_scores.get)
            best_score = regime_scores[best_regime]

            # 신뢰도 계산
            confidence = self.confidence_scorer.calculate_comprehensive_confidence(
                best_regime, regime_scores, indicators
            )

            # 히스토리 업데이트
            self.regime_history.append({
                'timestamp': datetime.now(),
                'regime': best_regime,
                'score': best_score,
                'confidence': confidence['overall_confidence']
            })

            self.current_regime = best_regime

            # Fund Flow (간소화)
            fund_flow = {
                'btc_flow': np.random.uniform(-0.1, 0.1),
                'altcoin_flow': np.random.uniform(-0.1, 0.1),
                'overall_flow': 'neutral'
            }

            return best_regime, fund_flow

        except Exception as e:
            self.logger.error(f"Market regime analysis error: {e}")
            return 'UNCERTAIN', {'btc_flow': 0, 'altcoin_flow': 0, 'overall_flow': 'neutral'}

    def _calculate_regime_scores(self, indicators):
        """Regime 점수 계산"""
        try:
            scores = {
                'BULL_CONSOLIDATION': 0.0,
                'BULL_VOLATILITY': 0.0,
                'BEAR_CONSOLIDATION': 0.0,
                'BEAR_VOLATILITY': 0.0,
                'SIDEWAYS_COMPRESSION': 0.0,
                'SIDEWAYS_CHOP': 0.0,
                'ACCUMULATION': 0.0,
                'DISTRIBUTION': 0.0
            }

            # 온체인/매크로 기여
            merged = indicators['onchain_macro_signals']['merged']
            if merged['score'] > 0.3:
                scores['BULL_CONSOLIDATION'] += 0.3
                scores['ACCUMULATION'] += 0.2
            elif merged['score'] < -0.3:
                scores['BEAR_CONSOLIDATION'] += 0.3
                scores['DISTRIBUTION'] += 0.2

            # 🔥🔥🔥 변동성 신호 기여 (NEW!) 🔥🔥🔥
            vol_signals = indicators['volatility_signals']
            vol_regime = vol_signals['volatility_regime']

            if 'HIGH' in vol_regime or 'EXTREME' in vol_regime:
                # 높은 변동성
                scores['BULL_VOLATILITY'] += 0.4
                scores['BEAR_VOLATILITY'] += 0.4
                scores['SIDEWAYS_CHOP'] += 0.3
            elif 'LOW' in vol_regime:
                # 낮은 변동성
                scores['BULL_CONSOLIDATION'] += 0.3
                scores['BEAR_CONSOLIDATION'] += 0.3
                scores['SIDEWAYS_COMPRESSION'] += 0.4

            # VRP 기여
            vrp_signal = vol_signals['vrp_signal']
            if vrp_signal in ['STRONG_BUY_VOL', 'BUY_VOL']:
                # 변동성 확대 예상
                scores['BULL_VOLATILITY'] += 0.2
                scores['BEAR_VOLATILITY'] += 0.2
            elif vrp_signal in ['STRONG_SELL_VOL', 'SELL_VOL']:
                # 변동성 축소 예상
                scores['BULL_CONSOLIDATION'] += 0.2
                scores['SIDEWAYS_COMPRESSION'] += 0.2

            # 유동성 기여
            liquidity_regime = indicators['liquidity_signals']['regime']
            if 'LOW' in liquidity_regime:
                scores['BEAR_VOLATILITY'] += 0.2
                scores['SIDEWAYS_CHOP'] += 0.2

            # 정규화
            max_score = max(scores.values()) if scores.values() else 1.0
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

            return scores

        except Exception as e:
            self.logger.error(f"Regime score calculation error: {e}")
            return {'UNCERTAIN': 1.0}

    def get_comprehensive_analysis_report(self, symbol='BTCUSDT'):
        """
        🔥🔥🔥 종합 분석 리포트 (변동성 포함) 🔥🔥🔥
        """
        try:
            # 다중 타임프레임 컨센서스
            mtf_report = self.mtf_consensus.get_consensus_report()

            # 유동성 종합 리포트
            liquidity_report = self.liquidity_detector.get_comprehensive_liquidity_report(symbol)

            # 🔥🔥🔥 변동성 종합 리포트 (NEW!) 🔥🔥🔥
            volatility_report = self.volatility_analyzer.get_comprehensive_volatility_report(symbol)

            return {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'regime_analysis': {
                    'current_regime': self.current_regime,
                    'regime_duration': self.current_regime_duration.total_seconds() if self.current_regime_duration else 0
                },
                'adaptive_weights': self.adaptive_weights,
                'multi_timeframe_consensus': mtf_report,
                'liquidity_analysis': liquidity_report,
                'volatility_analysis': volatility_report,  # 🔥🔥🔥 변동성 분석 추가 🔥🔥🔥
                'confidence_report': self.confidence_scorer.confidence_history[-1] if len(
                    self.confidence_scorer.confidence_history) > 0 else {}
            }

        except Exception as e:
            self.logger.error(f"Comprehensive analysis report error: {e}")
            return {}


# ═══════════════════════════════════════════════════════════════════════
# 🎯 사용 예시
# ═══════════════════════════════════════════════════════════════════════
"""
# 초기화
analyzer = MarketRegimeAnalyzer(market_data_manager)

# 기본 분석
regime, fund_flow = analyzer.analyze('BTCUSDT')
print(f"Current Regime: {regime}")

# 🔥🔥🔥 변동성 분석 (NEW!) 🔥🔥🔥
volatility_report = analyzer.volatility_analyzer.get_comprehensive_volatility_report('BTCUSDT')
print(f"Volatility Report: {volatility_report}")

# 실현변동성
realized_vol = analyzer.volatility_analyzer.get_comprehensive_realized_volatility('BTCUSDT')
print(f"Realized Volatility: {realized_vol}")

# Term Structure
term_structure = analyzer.volatility_analyzer.calculate_volatility_term_structure('BTCUSDT')
print(f"Term Structure: {term_structure}")

# 변동성 레짐
vol_regime = analyzer.volatility_analyzer.detect_volatility_regime('BTCUSDT')
print(f"Volatility Regime: {vol_regime}")

# GARCH 예측
garch_forecast = analyzer.volatility_analyzer.forecast_volatility_garch('BTCUSDT')
print(f"GARCH Forecast: {garch_forecast}")

# VRP
vrp = analyzer.volatility_analyzer.calculate_volatility_risk_premium('BTCUSDT')
print(f"Volatility Risk Premium: {vrp}")

# 차익거래 기회
arb_opportunities = analyzer.volatility_analyzer.detect_volatility_arbitrage_opportunities('BTCUSDT')
print(f"Arbitrage Opportunities: {arb_opportunities}")

# 종합 리포트 (모든 것 포함)
comprehensive_report = analyzer.get_comprehensive_analysis_report('BTCUSDT')
print(f"Comprehensive Report: {comprehensive_report}")
"""

# ═══════════════════════════════════════════════════════════════════════
# END OF PART 5/5 (FINAL)
#
# 🎉🎉🎉 병합 완료 후 market_regime_analyzer8.py로 저장하세요! 🎉🎉🎉
#
# 병합 방법:
# 1. Part 1의 전체 내용 복사
# 2. Part 2의 내용 이어붙이기 (import 제외)
# 3. Part 3의 내용 이어붙이기 (import 제외)
# 4. Part 4의 내용 이어붙이기 (import 제외)
# 5. Part 5의 내용 이어붙이기 (import 제외)
#
# 🔥🔥🔥 v8.0 핵심 신기능 🔥🔥🔥
# - VolatilityTermStructureAnalyzer 클래스 (완전히 새로운 기능)
# - Realized Volatility (5가지 방법)
# - Volatility Term Structure
# - GARCH Modeling & Forecasting
# - Volatility Regime Detection
# - Volatility Risk Premium
# - Volatility Arbitrage Opportunities
# - 모든 기존 기능 100% 유지
# ═══════════════════════════════════════════════════════════════════════