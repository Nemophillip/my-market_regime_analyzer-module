# market_regime_analyzer6.py

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
        """
        거래소 입출금 분석
        Returns: {'net_flow': float, 'inflow': float, 'outflow': float, 'signal': str}
        """
        cached = self._get_cached_data(f'exchange_flow_{timeframe}')
        if cached:
            return cached

        try:
            # 🔥 실제 구현 시 Glassnode, CryptoQuant API 사용
            # 여기서는 시뮬레이션 데이터
            inflow = np.random.uniform(5000, 15000)
            outflow = np.random.uniform(5000, 15000)
            net_flow = inflow - outflow

            # 신호 생성
            if net_flow > self.thresholds['exchange_inflow_high']:
                signal = 'SELLING_PRESSURE'  # 거래소로 대량 유입 = 매도 압력
            elif net_flow < -self.thresholds['exchange_outflow_high']:
                signal = 'ACCUMULATION'  # 거래소에서 대량 유출 = 축적
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
        """
        고래 활동 분석
        Returns: {'whale_transactions': int, 'whale_volume': float, 'signal': str}
        """
        cached = self._get_cached_data(f'whale_activity_{timeframe}')
        if cached:
            return cached

        try:
            # 🔥 실제 구현 시 온체인 API 사용
            whale_transactions = np.random.randint(5, 50)
            whale_volume = np.random.uniform(1000, 5000)

            # 신호 생성
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
        """
        MVRV (Market Value to Realized Value) 비율
        - > 3.5: 과매수 (역사적 고점 근처)
        - < 1.0: 과매도 (역사적 저점 근처)
        Returns: {'mvrv': float, 'signal': str}
        """
        cached = self._get_cached_data('mvrv_ratio')
        if cached:
            return cached

        try:
            # 🔥 실제 구현 시 Glassnode API 사용
            mvrv = np.random.uniform(0.8, 4.0)

            # 신호 생성
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
        """
        NVT (Network Value to Transactions) 비율
        - 높을수록 과대평가
        - 낮을수록 저평가
        Returns: {'nvt': float, 'signal': str}
        """
        cached = self._get_cached_data('nvt_ratio')
        if cached:
            return cached

        try:
            # 🔥 실제 구현 시 Glassnode API 사용
            nvt = np.random.uniform(40, 160)

            # 신호 생성
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
        """
        활성 주소 수 분석
        Returns: {'active_addresses': int, 'change_pct': float, 'signal': str}
        """
        cached = self._get_cached_data(f'active_addresses_{timeframe}')
        if cached:
            return cached

        try:
            # 🔥 실제 구현 시 온체인 API 사용
            active_addresses = np.random.randint(800000, 1200000)
            historical_avg = 1000000
            change_pct = ((active_addresses - historical_avg) / historical_avg) * 100

            # 신호 생성
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
        """
        종합 온체인 신호 생성
        Returns: {'score': float, 'signal': str, 'details': dict}
        """
        try:
            exchange_flow = self.get_exchange_flow()
            whale_activity = self.get_whale_activity()
            mvrv = self.get_mvrv_ratio()
            nvt = self.get_nvt_ratio()
            active_addresses = self.get_active_addresses()

            # 신호별 점수화 (-1.0 ~ 1.0)
            scores = {
                'exchange_flow': 0.0,
                'whale_activity': 0.0,
                'mvrv': 0.0,
                'nvt': 0.0,
                'active_addresses': 0.0
            }

            # Exchange Flow 점수
            if exchange_flow['signal'] == 'SELLING_PRESSURE':
                scores['exchange_flow'] = -0.8
            elif exchange_flow['signal'] == 'ACCUMULATION':
                scores['exchange_flow'] = 0.8

            # Whale Activity 점수
            if whale_activity['signal'] == 'HIGH_WHALE_ACTIVITY':
                scores['whale_activity'] = 0.5  # 중립적 (방향성 불확실)
            elif whale_activity['signal'] == 'LOW_WHALE_ACTIVITY':
                scores['whale_activity'] = -0.3

            # MVRV 점수
            if mvrv['signal'] == 'OVERVALUED':
                scores['mvrv'] = -0.7
            elif mvrv['signal'] == 'UNDERVALUED':
                scores['mvrv'] = 0.7
            elif mvrv['signal'] == 'FAIR_VALUE':
                scores['mvrv'] = 0.2

            # NVT 점수
            if nvt['signal'] == 'OVERVALUED':
                scores['nvt'] = -0.6
            elif nvt['signal'] == 'UNDERVALUED':
                scores['nvt'] = 0.6

            # Active Addresses 점수
            if active_addresses['signal'] == 'INCREASING_ADOPTION':
                scores['active_addresses'] = 0.5
            elif active_addresses['signal'] == 'DECREASING_ACTIVITY':
                scores['active_addresses'] = -0.5

            # 가중 평균 계산
            weights = {
                'exchange_flow': 0.30,
                'whale_activity': 0.15,
                'mvrv': 0.25,
                'nvt': 0.20,
                'active_addresses': 0.10
            }

            total_score = sum(scores[k] * weights[k] for k in scores)

            # 종합 신호 생성
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

        # 캐싱
        self._cache = {}
        self._cache_ttl = 300  # 5분 캐시

        # 매크로 데이터 히스토리
        self.funding_rate_history = deque(maxlen=100)
        self.oi_history = deque(maxlen=100)
        self.long_short_history = deque(maxlen=100)
        self.fear_greed_history = deque(maxlen=100)
        self.dominance_history = deque(maxlen=100)

        # 임계값 설정
        self.thresholds = {
            'funding_rate_high': 0.05,  # 5%
            'funding_rate_low': -0.05,
            'oi_increase_threshold': 15,  # %
            'long_short_extreme_high': 1.5,
            'long_short_extreme_low': 0.67,
            'fear_greed_extreme': 75,
            'fear_greed_fear': 25,
            'btc_dominance_high': 60,
            'btc_dominance_low': 40
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

    def get_funding_rate(self, symbol='BTCUSDT'):
        """
        펀딩비 분석
        - 양수: 롱 포지션이 많음 (과열 가능)
        - 음수: 숏 포지션이 많음 (공포 가능)
        Returns: {'funding_rate': float, 'signal': str}
        """
        cached = self._get_cached_data(f'funding_rate_{symbol}')
        if cached:
            return cached

        try:
            # 🔥 실제 구현 시 Binance Futures API 사용
            funding_rate = np.random.uniform(-0.1, 0.1) / 100  # -0.1% ~ 0.1%

            # 신호 생성
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
                'funding_rate': funding_rate * 100,  # % 단위로 변환
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
        """
        미결제약정 (Open Interest) 분석
        - 증가 + 가격 상승: 강한 상승 추세
        - 증가 + 가격 하락: 강한 하락 추세
        - 감소: 추세 약화
        Returns: {'oi': float, 'oi_change': float, 'signal': str}
        """
        cached = self._get_cached_data(f'open_interest_{symbol}')
        if cached:
            return cached

        try:
            # 🔥 실제 구현 시 거래소 API 사용
            current_oi = np.random.uniform(20000000000, 30000000000)  # 200억 ~ 300억

            # 이전 데이터와 비교
            if len(self.oi_history) > 0:
                prev_oi = self.oi_history[-1]['oi']
                oi_change = ((current_oi - prev_oi) / prev_oi) * 100
            else:
                oi_change = 0

            # 가격 추세 확인
            try:
                df = self.market_data.get_candle_data(symbol, '1h')
                if df is not None and len(df) > 1:
                    price_change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) /
                                    df['close'].iloc[-2]) * 100
                else:
                    price_change = 0
            except:
                price_change = 0

            # 신호 생성
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
        """
        롱/숏 비율 분석
        - > 1.5: 극도의 롱 포지션 (조정 가능)
        - < 0.67: 극도의 숏 포지션 (반등 가능)
        Returns: {'ratio': float, 'signal': str}
        """
        cached = self._get_cached_data(f'long_short_ratio_{symbol}')
        if cached:
            return cached

        try:
            # 🔥 실제 구현 시 거래소 API 사용
            ratio = np.random.uniform(0.5, 2.0)

            # 신호 생성
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
        """
        Fear & Greed Index 분석
        - 0-25: Extreme Fear
        - 25-45: Fear
        - 45-55: Neutral
        - 55-75: Greed
        - 75-100: Extreme Greed
        Returns: {'index': int, 'signal': str}
        """
        cached = self._get_cached_data('fear_greed_index')
        if cached:
            return cached

        try:
            # 🔥 실제 구현 시 Alternative.me API 사용
            index = np.random.randint(0, 100)

            # 신호 생성
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
        """
        비트코인 도미넌스 분석
        - 높음 (>60%): BTC 강세, 알트 약세
        - 낮음 (<40%): 알트 강세
        Returns: {'dominance': float, 'signal': str}
        """
        cached = self._get_cached_data('btc_dominance')
        if cached:
            return cached

        try:
            # 🔥 실제 구현 시 CoinGecko API 사용
            dominance = np.random.uniform(35, 65)

            # 신호 생성
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
        """
        스테이블코인 공급량 변화 분석
        - 증가: 잠재적 매수 압력
        - 감소: 자금 유출
        Returns: {'supply': float, 'change_pct': float, 'signal': str}
        """
        cached = self._get_cached_data('stablecoin_supply')
        if cached:
            return cached

        try:
            # 🔥 실제 구현 시 온체인 API 사용
            supply = np.random.uniform(120000000000, 150000000000)  # 1200억 ~ 1500억
            change_pct = np.random.uniform(-5, 5)

            # 신호 생성
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
        """
        종합 매크로 신호 생성
        Returns: {'score': float, 'signal': str, 'details': dict}
        """
        try:
            funding_rate = self.get_funding_rate()
            open_interest = self.get_open_interest()
            long_short_ratio = self.get_long_short_ratio()
            fear_greed = self.get_fear_greed_index()
            btc_dominance = self.get_bitcoin_dominance()
            stablecoin = self.get_stablecoin_supply()

            # 신호별 점수화 (-1.0 ~ 1.0)
            scores = {
                'funding_rate': 0.0,
                'open_interest': 0.0,
                'long_short_ratio': 0.0,
                'fear_greed': 0.0,
                'btc_dominance': 0.0,
                'stablecoin': 0.0
            }

            # Funding Rate 점수
            if funding_rate['signal'] == 'OVERHEATED_LONG':
                scores['funding_rate'] = -0.8
            elif funding_rate['signal'] == 'OVERHEATED_SHORT':
                scores['funding_rate'] = 0.8
            elif funding_rate['signal'] == 'BULLISH_BIAS':
                scores['funding_rate'] = 0.3
            elif funding_rate['signal'] == 'BEARISH_BIAS':
                scores['funding_rate'] = -0.3

            # Open Interest 점수
            if open_interest['signal'] == 'STRONG_BULLISH_MOMENTUM':
                scores['open_interest'] = 0.9
            elif open_interest['signal'] == 'STRONG_BEARISH_MOMENTUM':
                scores['open_interest'] = -0.9
            elif open_interest['signal'] == 'INCREASING_LEVERAGE':
                scores['open_interest'] = 0.5
            elif open_interest['signal'] == 'DELEVERAGING':
                scores['open_interest'] = -0.4

            # Long/Short Ratio 점수
            if long_short_ratio['signal'] == 'EXTREME_LONG':
                scores['long_short_ratio'] = -0.7  # 역발상
            elif long_short_ratio['signal'] == 'EXTREME_SHORT':
                scores['long_short_ratio'] = 0.7  # 역발상
            elif long_short_ratio['signal'] == 'LONG_BIAS':
                scores['long_short_ratio'] = 0.2
            elif long_short_ratio['signal'] == 'SHORT_BIAS':
                scores['long_short_ratio'] = -0.2

            # Fear & Greed 점수
            if fear_greed['signal'] == 'EXTREME_GREED':
                scores['fear_greed'] = -0.6  # 역발상
            elif fear_greed['signal'] == 'EXTREME_FEAR':
                scores['fear_greed'] = 0.6  # 역발상
            elif fear_greed['signal'] == 'GREED':
                scores['fear_greed'] = -0.2
            elif fear_greed['signal'] == 'FEAR':
                scores['fear_greed'] = 0.2

            # Bitcoin Dominance 점수
            if btc_dominance['signal'] == 'BTC_DOMINANCE':
                scores['btc_dominance'] = 0.3  # BTC 강세
            elif btc_dominance['signal'] == 'ALTCOIN_SEASON':
                scores['btc_dominance'] = 0.4  # 시장 활성화

            # Stablecoin Supply 점수
            if stablecoin['signal'] == 'INCREASING_LIQUIDITY':
                scores['stablecoin'] = 0.7
            elif stablecoin['signal'] == 'DECREASING_LIQUIDITY':
                scores['stablecoin'] = -0.7

            # 가중 평균 계산
            weights = {
                'funding_rate': 0.20,
                'open_interest': 0.25,
                'long_short_ratio': 0.15,
                'fear_greed': 0.15,
                'btc_dominance': 0.10,
                'stablecoin': 0.15
            }

            total_score = sum(scores[k] * weights[k] for k in scores)

            # 종합 신호 생성
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
# 🔥🔥🔥 6️⃣ 유동성 상태 추정 (Liquidity Regime Detection) 🔥🔥🔥
# ═══════════════════════════════════════════════════════════════════════

class LiquidityRegimeDetector:
    """
    💧 유동성 상태 추정 시스템 (Liquidity Regime Detection)
    - Order Book Depth Analysis (호가창 깊이 분석)
    - Bid-Ask Spread Analysis (매수-매도 스프레드 분석)
    - Market Impact Analysis (시장 충격 분석)
    - Slippage Estimation (슬리피지 추정)
    - Liquidity Score Calculation (유동성 점수 계산)
    - Liquidity Regime Classification (유동성 체제 분류)
    - Volume Profile Analysis (거래량 프로필 분석)
    - Liquidity Heatmap (유동성 히트맵)
    - Flash Crash Detection (급락 감지)
    - Liquidity Provider Behavior (유동성 공급자 행동 분석)
    """

    def __init__(self, market_data_manager):
        self.market_data = market_data_manager
        self.logger = get_logger("LiquidityRegime")

        # 📊 히스토리 데이터 저장
        self.orderbook_depth_history = deque(maxlen=100)
        self.spread_history = deque(maxlen=100)
        self.liquidity_score_history = deque(maxlen=100)
        self.regime_history = deque(maxlen=100)
        self.market_impact_history = deque(maxlen=50)
        self.slippage_history = deque(maxlen=50)

        # 🎯 유동성 레벨 임계값
        self.liquidity_levels = {
            'very_high': 0.85,
            'high': 0.70,
            'medium': 0.50,
            'low': 0.30,
            'very_low': 0.15
        }

        # 📦 캐싱
        self._cache = {}
        self._cache_ttl = 30  # 30초 캐시 (유동성은 빠르게 변함)

        # 🎚️ 호가창 분석 파라미터
        self.orderbook_config = {
            'depth_levels': 20,  # 분석할 호가 레벨 수
            'size_threshold': 10,  # BTC, 대량 주문 임계값
            'imbalance_threshold': 0.30,  # 매수/매도 불균형 임계값
            'wall_threshold': 50  # 벽(wall) 감지 임계값 (BTC)
        }

        # 📏 스프레드 분석 파라미터
        self.spread_config = {
            'tight_spread_bps': 5,  # 타이트 스프레드 (5 bps)
            'normal_spread_bps': 10,
            'wide_spread_bps': 20,
            'very_wide_spread_bps': 50
        }

        # 💥 시장 충격 분석 파라미터
        self.impact_config = {
            'trade_sizes': [1, 5, 10, 25, 50, 100],  # BTC
            'impact_threshold_low': 0.001,  # 0.1%
            'impact_threshold_medium': 0.005,  # 0.5%
            'impact_threshold_high': 0.01  # 1.0%
        }

        # 🔥 플래시 크래시 감지 파라미터
        self.flash_crash_config = {
            'price_drop_threshold': 0.05,  # 5% 급락
            'time_window_seconds': 60,  # 1분 이내
            'recovery_threshold': 0.03,  # 3% 회복
            'volume_spike_threshold': 3.0  # 거래량 3배 증가
        }

        # 🌡️ 유동성 히트맵 파라미터
        self.heatmap_config = {
            'price_levels': 50,  # 가격 레벨 수
            'time_buckets': 24,  # 시간 버킷 (1시간씩)
            'min_liquidity_threshold': 1.0  # 최소 유동성 (BTC)
        }

        # 📊 유동성 공급자 행동 추적
        self.lp_behavior = {
            'maker_taker_ratio': deque(maxlen=50),
            'order_cancellation_rate': deque(maxlen=50),
            'order_update_frequency': deque(maxlen=50),
            'aggressive_quotes': deque(maxlen=50)
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

    # ═══════════════════════════════════════════════════════════════════════
    # 1️⃣ Order Book Depth Analysis (호가창 깊이 분석)
    # ═══════════════════════════════════════════════════════════════════════

    def analyze_orderbook_depth(self, symbol='BTCUSDT'):
        """
        호가창 깊이 분석
        Returns: {
            'total_bid_volume': float,
            'total_ask_volume': float,
            'bid_ask_imbalance': float,
            'depth_score': float,
            'major_walls': list,
            'depth_quality': str
        }
        """
        cached = self._get_cached_data(f'orderbook_depth_{symbol}')
        if cached:
            return cached

        try:
            # 🔥 실제 구현 시 거래소 API에서 호가창 데이터 가져오기
            # 여기서는 시뮬레이션 데이터
            depth_levels = self.orderbook_config['depth_levels']

            # 매수 호가 (bid) 시뮬레이션
            bids = []
            base_price = 50000  # BTC 기준 가격
            for i in range(depth_levels):
                price = base_price - (i * 10)
                volume = np.random.uniform(0.5, 5.0) * (1 / (i + 1))  # 가격에서 멀수록 작아짐
                bids.append({'price': price, 'volume': volume})

            # 매도 호가 (ask) 시뮬레이션
            asks = []
            for i in range(depth_levels):
                price = base_price + (i * 10)
                volume = np.random.uniform(0.5, 5.0) * (1 / (i + 1))
                asks.append({'price': price, 'volume': volume})

            # 총 거래량 계산
            total_bid_volume = sum(b['volume'] for b in bids)
            total_ask_volume = sum(a['volume'] for a in asks)

            # 매수/매도 불균형 계산
            total_volume = total_bid_volume + total_ask_volume
            bid_ask_imbalance = (total_bid_volume - total_ask_volume) / total_volume if total_volume > 0 else 0

            # 주요 벽(wall) 감지
            wall_threshold = self.orderbook_config['wall_threshold']
            major_walls = []

            for bid in bids:
                if bid['volume'] > wall_threshold:
                    major_walls.append({
                        'side': 'bid',
                        'price': bid['price'],
                        'volume': bid['volume']
                    })

            for ask in asks:
                if ask['volume'] > wall_threshold:
                    major_walls.append({
                        'side': 'ask',
                        'price': ask['price'],
                        'volume': ask['volume']
                    })

            # 깊이 점수 계산 (0.0 ~ 1.0)
            # 높은 거래량 + 균형잡힌 호가 = 높은 점수
            volume_score = min(total_volume / 100, 1.0)  # 100 BTC를 만점으로
            balance_score = 1.0 - abs(bid_ask_imbalance)
            depth_score = (volume_score * 0.7 + balance_score * 0.3)

            # 깊이 품질 분류
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

            # 히스토리 저장
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

    # ═══════════════════════════════════════════════════════════════════════
    # 2️⃣ Bid-Ask Spread Analysis (매수-매도 스프레드 분석)
    # ═══════════════════════════════════════════════════════════════════════

    def analyze_bid_ask_spread(self, symbol='BTCUSDT'):
        """
        매수-매도 스프레드 분석
        Returns: {
            'spread_bps': float,  # Basis Points (0.01%)
            'spread_percentage': float,
            'spread_quality': str,
            'mid_price': float,
            'best_bid': float,
            'best_ask': float
        }
        """
        cached = self._get_cached_data(f'spread_{symbol}')
        if cached:
            return cached

        try:
            # 🔥 실제 구현 시 거래소 API에서 티커 데이터 가져오기
            # 시뮬레이션 데이터
            base_price = 50000
            spread_pct = np.random.uniform(0.0001, 0.003)  # 0.01% ~ 0.3%

            best_bid = base_price * (1 - spread_pct / 2)
            best_ask = base_price * (1 + spread_pct / 2)
            mid_price = (best_bid + best_ask) / 2

            # 스프레드 계산
            spread_absolute = best_ask - best_bid
            spread_percentage = (spread_absolute / mid_price) * 100
            spread_bps = spread_percentage * 100  # Basis Points

            # 스프레드 품질 평가
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

            # 히스토리 저장
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

    # ═══════════════════════════════════════════════════════════════════════
    # 3️⃣ Market Impact Analysis (시장 충격 분석)
    # ═══════════════════════════════════════════════════════════════════════

    def analyze_market_impact(self, symbol='BTCUSDT'):
        """
        다양한 거래 규모에 대한 시장 충격 분석
        Returns: {
            'impact_curve': list,  # [{size, buy_impact, sell_impact}]
            'average_impact': float,
            'impact_quality': str,
            'resilience_score': float
        }
        """
        cached = self._get_cached_data(f'market_impact_{symbol}')
        if cached:
            return cached

        try:
            # 호가창 데이터 가져오기
            orderbook = self.analyze_orderbook_depth(symbol)

            # 다양한 거래 규모에 대한 충격 계산
            trade_sizes = self.impact_config['trade_sizes']
            impact_curve = []

            for size in trade_sizes:
                # 매수 충격 시뮬레이션
                buy_impact = self._simulate_trade_impact(
                    size, 
                    'buy', 
                    orderbook['total_ask_volume']
                )

                # 매도 충격 시뮬레이션
                sell_impact = self._simulate_trade_impact(
                    size, 
                    'sell', 
                    orderbook['total_bid_volume']
                )

                impact_curve.append({
                    'size': size,
                    'buy_impact': buy_impact,
                    'sell_impact': sell_impact,
                    'average_impact': (buy_impact + sell_impact) / 2
                })

            # 평균 충격 계산
            average_impact = np.mean([ic['average_impact'] for ic in impact_curve])

            # 충격 품질 평가
            if average_impact < self.impact_config['impact_threshold_low']:
                impact_quality = 'VERY_LOW'
            elif average_impact < self.impact_config['impact_threshold_medium']:
                impact_quality = 'LOW'
            elif average_impact < self.impact_config['impact_threshold_high']:
                impact_quality = 'MODERATE'
            else:
                impact_quality = 'HIGH'

            # 복원력 점수 (낮은 충격 = 높은 복원력)
            resilience_score = 1.0 - min(average_impact / self.impact_config['impact_threshold_high'], 1.0)

            result = {
                'impact_curve': impact_curve,
                'average_impact': average_impact,
                'impact_quality': impact_quality,
                'resilience_score': resilience_score,
                'timestamp': datetime.now()
            }

            # 히스토리 저장
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
            # 간단한 충격 모델: 거래량 / 가용 유동성
            if available_liquidity > 0:
                impact_ratio = size / available_liquidity
                
                # 비선형 충격 (큰 거래일수록 충격이 더 크게 증가)
                impact = impact_ratio * (1 + impact_ratio)
                
                # 무작위 변동 추가
                impact *= np.random.uniform(0.8, 1.2)
                
                return min(impact, 0.1)  # 최대 10% 충격
            else:
                return 0.05  # 기본값
        except:
            return 0.005

    # ═══════════════════════════════════════════════════════════════════════
    # 4️⃣ Slippage Estimation (슬리피지 추정)
    # ═══════════════════════════════════════════════════════════════════════

    def estimate_slippage(self, symbol='BTCUSDT', trade_size=10):
        """
        특정 거래 규모에 대한 슬리피지 추정
        Returns: {
            'expected_slippage_bps': float,
            'expected_slippage_pct': float,
            'worst_case_slippage_bps': float,
            'confidence': float,
            'execution_quality': str
        }
        """
        cached = self._get_cached_data(f'slippage_{symbol}_{trade_size}')
        if cached:
            return cached

        try:
            # 시장 충격 분석 데이터 활용
            impact_analysis = self.analyze_market_impact(symbol)
            
            # 스프레드 분석 데이터 활용
            spread_analysis = self.analyze_bid_ask_spread(symbol)

            # 슬리피지 = 스프레드 + 시장 충격
            spread_slippage = spread_analysis['spread_percentage']
            
            # 해당 거래 규모의 충격 찾기
            impact = 0
            for ic in impact_analysis['impact_curve']:
                if ic['size'] >= trade_size:
                    impact = ic['average_impact'] * 100  # 퍼센트로 변환
                    break
            
            if impact == 0:  # 거래 규모가 커브를 초과하면 외삽
                impact = impact_analysis['average_impact'] * 100 * (trade_size / 50)

            # 예상 슬리피지
            expected_slippage_pct = spread_slippage + impact
            expected_slippage_bps = expected_slippage_pct * 100

            # 최악의 경우 슬리피지 (변동성 고려)
            volatility_factor = 1.5  # 변동성 고려 계수
            worst_case_slippage_bps = expected_slippage_bps * volatility_factor

            # 신뢰도 계산 (호가창 깊이와 스프레드 안정성 기반)
            orderbook = self.analyze_orderbook_depth(symbol)
            confidence = orderbook['depth_score'] * 0.7 + (1.0 - min(spread_slippage / 0.5, 1.0)) * 0.3

            # 실행 품질 평가
            if expected_slippage_bps < 10:
                execution_quality = 'EXCELLENT'
            elif expected_slippage_bps < 25:
                execution_quality = 'GOOD'
            elif expected_slippage_bps < 50:
                execution_quality = 'FAIR'
            elif expected_slippage_bps < 100:
                execution_quality = 'POOR'
            else:
                execution_quality = 'VERY_POOR'

            result = {
                'expected_slippage_bps': expected_slippage_bps,
                'expected_slippage_pct': expected_slippage_pct,
                'worst_case_slippage_bps': worst_case_slippage_bps,
                'confidence': confidence,
                'execution_quality': execution_quality,
                'trade_size': trade_size,
                'timestamp': datetime.now()
            }

            # 히스토리 저장
            self.slippage_history.append(result)
            self._set_cached_data(f'slippage_{symbol}_{trade_size}', result)

            return result

        except Exception as e:
            self.logger.error(f"Slippage estimation error: {e}")
            return {
                'expected_slippage_bps': 25,
                'expected_slippage_pct': 0.25,
                'worst_case_slippage_bps': 50,
                'confidence': 0.5,
                'execution_quality': 'FAIR',
                'trade_size': trade_size
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 5️⃣ Liquidity Score Calculation (유동성 점수 계산)
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_liquidity_score(self, symbol='BTCUSDT'):
        """
        종합 유동성 점수 계산 (0.0 ~ 1.0)
        Returns: {
            'liquidity_score': float,
            'component_scores': dict,
            'score_breakdown': dict,
            'confidence': float
        }
        """
        try:
            # 모든 유동성 지표 수집
            orderbook = self.analyze_orderbook_depth(symbol)
            spread = self.analyze_bid_ask_spread(symbol)
            impact = self.analyze_market_impact(symbol)
            slippage = self.estimate_slippage(symbol, trade_size=10)

            # 각 컴포넌트별 점수 (0.0 ~ 1.0)
            component_scores = {
                'depth_score': orderbook['depth_score'],
                'spread_score': self._score_spread(spread),
                'impact_score': impact['resilience_score'],
                'slippage_score': self._score_slippage(slippage)
            }

            # 가중 평균 계산
            weights = {
                'depth_score': 0.35,
                'spread_score': 0.25,
                'impact_score': 0.25,
                'slippage_score': 0.15
            }

            liquidity_score = sum(
                component_scores[k] * weights[k] 
                for k in component_scores
            )

            # 신뢰도 계산 (데이터 품질 기반)
            confidence = np.mean([
                orderbook['depth_score'],
                slippage['confidence']
            ])

            # 점수 세부 내역
            score_breakdown = {
                'orderbook_depth': {
                    'score': component_scores['depth_score'],
                    'weight': weights['depth_score'],
                    'contribution': component_scores['depth_score'] * weights['depth_score']
                },
                'bid_ask_spread': {
                    'score': component_scores['spread_score'],
                    'weight': weights['spread_score'],
                    'contribution': component_scores['spread_score'] * weights['spread_score']
                },
                'market_impact': {
                    'score': component_scores['impact_score'],
                    'weight': weights['impact_score'],
                    'contribution': component_scores['impact_score'] * weights['impact_score']
                },
                'slippage': {
                    'score': component_scores['slippage_score'],
                    'weight': weights['slippage_score'],
                    'contribution': component_scores['slippage_score'] * weights['slippage_score']
                }
            }

            result = {
                'liquidity_score': liquidity_score,
                'component_scores': component_scores,
                'score_breakdown': score_breakdown,
                'confidence': confidence,
                'timestamp': datetime.now()
            }

            # 히스토리 저장
            self.liquidity_score_history.append(result)

            return result

        except Exception as e:
            self.logger.error(f"Liquidity score calculation error: {e}")
            return {
                'liquidity_score': 0.5,
                'component_scores': {},
                'score_breakdown': {},
                'confidence': 0.5
            }

    def _score_spread(self, spread_data):
        """스프레드를 점수로 변환 (낮을수록 좋음)"""
        try:
            spread_bps = spread_data['spread_bps']
            
            # 스프레드가 낮을수록 높은 점수
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

    def _score_slippage(self, slippage_data):
        """슬리피지를 점수로 변환 (낮을수록 좋음)"""
        try:
            slippage_bps = slippage_data['expected_slippage_bps']
            
            # 슬리피지가 낮을수록 높은 점수
            if slippage_bps <= 10:
                return 1.0
            elif slippage_bps <= 25:
                return 0.8
            elif slippage_bps <= 50:
                return 0.6
            elif slippage_bps <= 100:
                return 0.4
            else:
                return 0.2
        except:
            return 0.5

    # ═══════════════════════════════════════════════════════════════════════
    # 6️⃣ Liquidity Regime Classification (유동성 체제 분류)
    # ═══════════════════════════════════════════════════════════════════════

    def classify_liquidity_regime(self, symbol='BTCUSDT'):
        """
        현재 유동성 상태를 체제로 분류
        Returns: {
            'regime': str,
            'regime_score': float,
            'regime_confidence': float,
            'characteristics': dict,
            'warnings': list
        }
        """
        try:
            # 유동성 점수 계산
            liquidity_analysis = self.calculate_liquidity_score(symbol)
            score = liquidity_analysis['liquidity_score']

            # 추가 분석
            orderbook = self.analyze_orderbook_depth(symbol)
            spread = self.analyze_bid_ask_spread(symbol)
            impact = self.analyze_market_impact(symbol)

            # 체제 분류
            regime = None
            characteristics = {}
            warnings = []

            if score >= self.liquidity_levels['very_high']:
                regime = 'VERY_HIGH_LIQUIDITY'
                characteristics = {
                    'description': '매우 높은 유동성 - 대량 거래 가능',
                    'trade_recommendation': '대량 주문 실행에 최적',
                    'risk_level': 'VERY_LOW'
                }

            elif score >= self.liquidity_levels['high']:
                regime = 'HIGH_LIQUIDITY'
                characteristics = {
                    'description': '높은 유동성 - 원활한 거래 환경',
                    'trade_recommendation': '일반 거래에 적합',
                    'risk_level': 'LOW'
                }

            elif score >= self.liquidity_levels['medium']:
                regime = 'MEDIUM_LIQUIDITY'
                characteristics = {
                    'description': '중간 유동성 - 주의 필요',
                    'trade_recommendation': '중소형 주문 권장',
                    'risk_level': 'MEDIUM'
                }
                
                if spread['spread_bps'] > self.spread_config['wide_spread_bps']:
                    warnings.append('⚠️ 넓은 스프레드 감지')

            elif score >= self.liquidity_levels['low']:
                regime = 'LOW_LIQUIDITY'
                characteristics = {
                    'description': '낮은 유동성 - 신중한 거래 필요',
                    'trade_recommendation': '소량 주문만 권장',
                    'risk_level': 'HIGH'
                }
                
                warnings.append('⚠️ 낮은 유동성 - 슬리피지 주의')
                
                if orderbook['bid_ask_imbalance'] > 0.3:
                    warnings.append('⚠️ 호가 불균형 감지')

            else:
                regime = 'VERY_LOW_LIQUIDITY'
                characteristics = {
                    'description': '매우 낮은 유동성 - 거래 위험 높음',
                    'trade_recommendation': '거래 자제 권장',
                    'risk_level': 'VERY_HIGH'
                }
                
                warnings.append('🚨 매우 낮은 유동성 - 거래 위험')
                warnings.append('🚨 높은 슬리피지 예상')

            # 플래시 크래시 위험 체크
            flash_crash_risk = self._check_flash_crash_risk(orderbook, spread, impact)
            if flash_crash_risk['risk_detected']:
                warnings.append(f'🚨 플래시 크래시 위험: {flash_crash_risk["risk_level"]}')

            result = {
                'regime': regime,
                'regime_score': score,
                'regime_confidence': liquidity_analysis['confidence'],
                'characteristics': characteristics,
                'warnings': warnings,
                'flash_crash_risk': flash_crash_risk,
                'timestamp': datetime.now()
            }

            # 히스토리 저장
            self.regime_history.append(result)

            return result

        except Exception as e:
            self.logger.error(f"Liquidity regime classification error: {e}")
            return {
                'regime': 'UNKNOWN',
                'regime_score': 0.5,
                'regime_confidence': 0.5,
                'characteristics': {},
                'warnings': []
            }

    def _check_flash_crash_risk(self, orderbook, spread, impact):
        """플래시 크래시 위험 체크"""
        try:
            risk_detected = False
            risk_level = 'LOW'
            risk_factors = []

            # 1️⃣ 호가창 깊이 부족
            if orderbook['depth_score'] < 0.3:
                risk_factors.append('얕은 호가창')
                risk_detected = True

            # 2️⃣ 넓은 스프레드
            if spread['spread_bps'] > self.spread_config['very_wide_spread_bps']:
                risk_factors.append('매우 넓은 스프레드')
                risk_detected = True

            # 3️⃣ 높은 시장 충격
            if impact['average_impact'] > self.impact_config['impact_threshold_high']:
                risk_factors.append('높은 시장 충격')
                risk_detected = True

            # 4️⃣ 호가 불균형
            if abs(orderbook['bid_ask_imbalance']) > 0.5:
                risk_factors.append('심각한 호가 불균형')
                risk_detected = True

            # 위험 레벨 결정
            if len(risk_factors) >= 3:
                risk_level = 'VERY_HIGH'
            elif len(risk_factors) >= 2:
                risk_level = 'HIGH'
            elif len(risk_factors) >= 1:
                risk_level = 'MEDIUM'

            return {
                'risk_detected': risk_detected,
                'risk_level': risk_level,
                'risk_factors': risk_factors
            }

        except Exception as e:
            self.logger.debug(f"Flash crash risk check error: {e}")
            return {
                'risk_detected': False,
                'risk_level': 'UNKNOWN',
                'risk_factors': []
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 7️⃣ Volume Profile Analysis (거래량 프로필 분석)
    # ═══════════════════════════════════════════════════════════════════════

    def analyze_volume_profile(self, symbol='BTCUSDT', timeframe='1h', period=24):
        """
        거래량 프로필 분석 - 가격대별 거래량 분포
        Returns: {
            'volume_profile': list,  # [{price, volume, percentage}]
            'value_area': dict,  # 가장 많이 거래된 가격대
            'poc': float,  # Point of Control (최대 거래량 가격)
            'profile_shape': str
        }
        """
        try:
            # 🔥 실제 구현 시 캔들 데이터에서 거래량 프로필 생성
            # 시뮬레이션 데이터
            price_levels = 50
            base_price = 50000

            volume_profile = []
            total_volume = 0

            for i in range(price_levels):
                price = base_price - 500 + (i * 20)
                # 정규분포 형태의 거래량 (중간 가격대에서 많음)
                volume = np.random.normal(100, 30) * np.exp(-((i - 25) ** 2) / 200)
                volume = max(volume, 0)
                total_volume += volume

                volume_profile.append({
                    'price': price,
                    'volume': volume
                })

            # 퍼센티지 계산
            for vp in volume_profile:
                vp['percentage'] = (vp['volume'] / total_volume * 100) if total_volume > 0 else 0

            # POC (Point of Control) - 최대 거래량 가격
            poc_data = max(volume_profile, key=lambda x: x['volume'])
            poc = poc_data['price']

            # Value Area (상위 70% 거래량이 발생한 가격대)
            sorted_profile = sorted(volume_profile, key=lambda x: x['volume'], reverse=True)
            cumulative_volume = 0
            value_area_levels = []

            for vp in sorted_profile:
                cumulative_volume += vp['volume']
                value_area_levels.append(vp)
                
                if cumulative_volume / total_volume >= 0.70:
                    break

            value_area = {
                'high': max(vp['price'] for vp in value_area_levels),
                'low': min(vp['price'] for vp in value_area_levels),
                'poc': poc
            }

            # 프로필 형태 분류
            profile_shape = self._classify_profile_shape(volume_profile)

            return {
                'volume_profile': volume_profile,
                'value_area': value_area,
                'poc': poc,
                'profile_shape': profile_shape,
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Volume profile analysis error: {e}")
            return {
                'volume_profile': [],
                'value_area': {},
                'poc': 0,
                'profile_shape': 'UNKNOWN'
            }

    def _classify_profile_shape(self, volume_profile):
        """거래량 프로필 형태 분류"""
        try:
            if not volume_profile:
                return 'UNKNOWN'

            volumes = [vp['volume'] for vp in volume_profile]
            max_volume = max(volumes)
            max_idx = volumes.index(max_volume)

            # P-shaped: 최대 거래량이 상단에
            if max_idx < len(volumes) * 0.3:
                return 'P_SHAPED'
            # b-shaped: 최대 거래량이 하단에
            elif max_idx > len(volumes) * 0.7:
                return 'B_SHAPED'
            # D-shaped (left): 최대 거래량이 왼쪽(낮은 가격)에
            elif max_idx < len(volumes) * 0.2:
                return 'D_SHAPED_LEFT'
            # D-shaped (right): 최대 거래량이 오른쪽(높은 가격)에
            elif max_idx > len(volumes) * 0.8:
                return 'D_SHAPED_RIGHT'
            # Normal: 중앙에 최대 거래량
            else:
                return 'NORMAL'

        except Exception as e:
            self.logger.debug(f"Profile shape classification error: {e}")
            return 'UNKNOWN'

    # ═══════════════════════════════════════════════════════════════════════
    # 8️⃣ Liquidity Heatmap (유동성 히트맵)
    # ═══════════════════════════════════════════════════════════════════════

    def generate_liquidity_heatmap(self, symbol='BTCUSDT'):
        """
        유동성 히트맵 생성 - 시간대별, 가격대별 유동성 분포
        Returns: {
            'heatmap': numpy.ndarray,  # 2D array [price_level, time]
            'price_levels': list,
            'time_labels': list,
            'hot_zones': list,  # 고유동성 구역
            'cold_zones': list  # 저유동성 구역
        }
        """
        try:
            price_levels = self.heatmap_config['price_levels']
            time_buckets = self.heatmap_config['time_buckets']

            # 시뮬레이션 히트맵 데이터
            # 실제로는 과거 데이터에서 시간대별/가격대별 거래량 집계
            heatmap = np.random.rand(price_levels, time_buckets)
            
            # 시장 시간대 패턴 반영 (예: 특정 시간대에 유동성 높음)
            for t in range(time_buckets):
                # 거래 활발 시간대 (예: 8-10시, 20-22시)
                if t in [8, 9, 10, 20, 21, 22]:
                    heatmap[:, t] *= 1.5

            # 가격 레벨
            base_price = 50000
            price_range = 1000
            price_levels_list = [
                base_price - price_range/2 + (i * price_range / price_levels)
                for i in range(price_levels)
            ]

            # 시간 레이블
            time_labels = [f'{i:02d}:00' for i in range(time_buckets)]

            # 고유동성 구역 (hot zones) 찾기
            hot_threshold = np.percentile(heatmap, 80)
            hot_zones = []
            for i in range(price_levels):
                for j in range(time_buckets):
                    if heatmap[i, j] > hot_threshold:
                        hot_zones.append({
                            'price': price_levels_list[i],
                            'time': time_labels[j],
                            'liquidity': float(heatmap[i, j])
                        })

            # 저유동성 구역 (cold zones) 찾기
            cold_threshold = np.percentile(heatmap, 20)
            cold_zones = []
            for i in range(price_levels):
                for j in range(time_buckets):
                    if heatmap[i, j] < cold_threshold:
                        cold_zones.append({
                            'price': price_levels_list[i],
                            'time': time_labels[j],
                            'liquidity': float(heatmap[i, j])
                        })

            return {
                'heatmap': heatmap.tolist(),  # JSON 직렬화를 위해 list로 변환
                'price_levels': price_levels_list,
                'time_labels': time_labels,
                'hot_zones': hot_zones[:10],  # 상위 10개
                'cold_zones': cold_zones[:10],  # 하위 10개
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Liquidity heatmap generation error: {e}")
            return {
                'heatmap': [],
                'price_levels': [],
                'time_labels': [],
                'hot_zones': [],
                'cold_zones': []
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 9️⃣ Flash Crash Detection (급락 감지)
    # ═══════════════════════════════════════════════════════════════════════

    def detect_flash_crash(self, symbol='BTCUSDT'):
        """
        플래시 크래시 감지 및 분석
        Returns: {
            'flash_crash_detected': bool,
            'severity': str,
            'price_drop_pct': float,
            'recovery_status': str,
            'volume_spike': float,
            'liquidity_drain': bool
        }
        """
        try:
            # 🔥 실제 구현 시 최근 캔들 데이터 분석
            # 시뮬레이션
            
            # 최근 가격 변동 체크
            price_drop_pct = np.random.uniform(-0.02, 0.02)  # -2% ~ 2%
            is_significant_drop = price_drop_pct < -self.flash_crash_config['price_drop_threshold']

            # 거래량 급증 체크
            volume_spike = np.random.uniform(0.5, 4.0)  # 0.5배 ~ 4배
            is_volume_spike = volume_spike > self.flash_crash_config['volume_spike_threshold']

            # 유동성 분석
            liquidity_analysis = self.calculate_liquidity_score(symbol)
            liquidity_drain = liquidity_analysis['liquidity_score'] < 0.3

            # 플래시 크래시 판정
            flash_crash_detected = (
                is_significant_drop and 
                (is_volume_spike or liquidity_drain)
            )

            # 심각도 평가
            if flash_crash_detected:
                if price_drop_pct < -0.10:  # -10% 이상
                    severity = 'EXTREME'
                elif price_drop_pct < -0.07:  # -7% 이상
                    severity = 'SEVERE'
                elif price_drop_pct < -0.05:  # -5% 이상
                    severity = 'MODERATE'
                else:
                    severity = 'MILD'
            else:
                severity = 'NONE'

            # 회복 상태 체크
            recovery_pct = np.random.uniform(-0.01, 0.04)  # 회복률
            if recovery_pct > self.flash_crash_config['recovery_threshold']:
                recovery_status = 'RECOVERED'
            elif recovery_pct > 0.01:
                recovery_status = 'RECOVERING'
            elif recovery_pct > 0:
                recovery_status = 'SLOW_RECOVERY'
            else:
                recovery_status = 'NO_RECOVERY'

            result = {
                'flash_crash_detected': flash_crash_detected,
                'severity': severity,
                'price_drop_pct': price_drop_pct * 100,
                'recovery_status': recovery_status,
                'recovery_pct': recovery_pct * 100,
                'volume_spike': volume_spike,
                'liquidity_drain': liquidity_drain,
                'timestamp': datetime.now()
            }

            return result

        except Exception as e:
            self.logger.error(f"Flash crash detection error: {e}")
            return {
                'flash_crash_detected': False,
                'severity': 'NONE',
                'price_drop_pct': 0,
                'recovery_status': 'NORMAL',
                'volume_spike': 1.0,
                'liquidity_drain': False
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 🔟 Liquidity Provider Behavior (유동성 공급자 행동 분석)
    # ═══════════════════════════════════════════════════════════════════════

    def analyze_lp_behavior(self, symbol='BTCUSDT'):
        """
        유동성 공급자(LP) 행동 패턴 분석
        Returns: {
            'maker_taker_ratio': float,
            'order_cancellation_rate': float,
            'order_update_frequency': float,
            'aggressive_quoting': bool,
            'lp_confidence': float,
            'behavior_pattern': str
        }
        """
        try:
            # 🔥 실제 구현 시 거래소 API의 상세 주문 데이터 활용
            # 시뮬레이션 데이터

            # Maker/Taker 비율
            maker_taker_ratio = np.random.uniform(0.3, 1.5)
            self.lp_behavior['maker_taker_ratio'].append(maker_taker_ratio)

            # 주문 취소율
            order_cancellation_rate = np.random.uniform(0.2, 0.8)
            self.lp_behavior['order_cancellation_rate'].append(order_cancellation_rate)

            # 주문 업데이트 빈도 (초당)
            order_update_frequency = np.random.uniform(0.5, 5.0)
            self.lp_behavior['order_update_frequency'].append(order_update_frequency)

            # 공격적 호가 (타이트한 스프레드)
            spread_analysis = self.analyze_bid_ask_spread(symbol)
            aggressive_quoting = spread_analysis['spread_bps'] < self.spread_config['tight_spread_bps']
            self.lp_behavior['aggressive_quotes'].append(1 if aggressive_quoting else 0)

            # LP 신뢰도 계산
            # 높은 maker 비율 + 낮은 취소율 + 공격적 호가 = 높은 신뢰도
            lp_confidence = (
                min(maker_taker_ratio / 1.0, 1.0) * 0.4 +  # Maker 비율
                (1.0 - order_cancellation_rate) * 0.3 +  # 낮은 취소율
                (1 if aggressive_quoting else 0) * 0.3  # 공격적 호가
            )

            # 행동 패턴 분류
            if lp_confidence > 0.7 and maker_taker_ratio > 0.8:
                behavior_pattern = 'STABLE_PROVISION'
            elif order_cancellation_rate > 0.6 or order_update_frequency > 3.0:
                behavior_pattern = 'VOLATILE_PROVISION'
            elif maker_taker_ratio < 0.5:
                behavior_pattern = 'TAKER_DOMINATED'
            elif aggressive_quoting:
                behavior_pattern = 'AGGRESSIVE_COMPETITION'
            else:
                behavior_pattern = 'NORMAL_PROVISION'

            result = {
                'maker_taker_ratio': maker_taker_ratio,
                'order_cancellation_rate': order_cancellation_rate,
                'order_update_frequency': order_update_frequency,
                'aggressive_quoting': aggressive_quoting,
                'lp_confidence': lp_confidence,
                'behavior_pattern': behavior_pattern,
                'timestamp': datetime.now()
            }

            return result

        except Exception as e:
            self.logger.error(f"LP behavior analysis error: {e}")
            return {
                'maker_taker_ratio': 1.0,
                'order_cancellation_rate': 0.5,
                'order_update_frequency': 1.0,
                'aggressive_quoting': False,
                'lp_confidence': 0.5,
                'behavior_pattern': 'NORMAL_PROVISION'
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 종합 유동성 리포트
    # ═══════════════════════════════════════════════════════════════════════

    def get_comprehensive_liquidity_report(self, symbol='BTCUSDT'):
        """
        종합 유동성 분석 리포트
        Returns: 모든 유동성 지표를 통합한 상세 리포트
        """
        try:
            # 모든 분석 수행
            orderbook = self.analyze_orderbook_depth(symbol)
            spread = self.analyze_bid_ask_spread(symbol)
            impact = self.analyze_market_impact(symbol)
            slippage = self.estimate_slippage(symbol, trade_size=10)
            liquidity_score = self.calculate_liquidity_score(symbol)
            regime = self.classify_liquidity_regime(symbol)
            volume_profile = self.analyze_volume_profile(symbol)
            heatmap = self.generate_liquidity_heatmap(symbol)
            flash_crash = self.detect_flash_crash(symbol)
            lp_behavior = self.analyze_lp_behavior(symbol)

            # 종합 인사이트 생성
            insights = self._generate_liquidity_insights(
                regime, liquidity_score, spread, impact, flash_crash
            )

            # 거래 추천 생성
            trade_recommendations = self._generate_trade_recommendations(
                regime, liquidity_score, slippage
            )

            report = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                
                # 핵심 지표
                'liquidity_regime': regime,
                'liquidity_score': liquidity_score,
                
                # 상세 분석
                'orderbook_depth': orderbook,
                'bid_ask_spread': spread,
                'market_impact': impact,
                'slippage_estimate': slippage,
                'volume_profile': volume_profile,
                'liquidity_heatmap': heatmap,
                'flash_crash_status': flash_crash,
                'lp_behavior': lp_behavior,
                
                # 인사이트 및 추천
                'insights': insights,
                'trade_recommendations': trade_recommendations,
                
                # 히스토리 통계
                'historical_stats': self._calculate_historical_stats()
            }

            return report

        except Exception as e:
            self.logger.error(f"Comprehensive liquidity report error: {e}")
            return {}

    def _generate_liquidity_insights(self, regime, liquidity_score, spread, impact, flash_crash):
        """유동성 인사이트 생성"""
        insights = []

        try:
            # 체제 기반 인사이트
            if regime['regime'] == 'VERY_HIGH_LIQUIDITY':
                insights.append('✅ 최적의 거래 환경 - 대량 주문 실행 가능')
            elif regime['regime'] == 'VERY_LOW_LIQUIDITY':
                insights.append('⚠️ 위험: 매우 낮은 유동성 - 거래 자제 권장')

            # 스프레드 기반 인사이트
            if spread['spread_quality'] == 'VERY_WIDE':
                insights.append('💸 매우 넓은 스프레드 - 거래 비용 높음')
            elif spread['spread_quality'] == 'VERY_TIGHT':
                insights.append('💰 매우 타이트한 스프레드 - 거래 비용 최소화')

            # 시장 충격 기반 인사이트
            if impact['impact_quality'] == 'HIGH':
                insights.append('⚡ 높은 시장 충격 - 주문 분할 실행 권장')
            elif impact['impact_quality'] == 'VERY_LOW':
                insights.append('🎯 낮은 시장 충격 - 원활한 거래 가능')

            # 플래시 크래시 경고
            if flash_crash['flash_crash_detected']:
                insights.append(f'🚨 플래시 크래시 감지 - 심각도: {flash_crash["severity"]}')

            # 점수 기반 인사이트
            score = liquidity_score['liquidity_score']
            if score > 0.8:
                insights.append('🌟 우수한 유동성 환경')
            elif score < 0.3:
                insights.append('⚠️ 불량한 유동성 환경')

        except Exception as e:
            self.logger.debug(f"Insights generation error: {e}")

        return insights

    def _generate_trade_recommendations(self, regime, liquidity_score, slippage):
        """거래 추천 생성"""
        recommendations = []

        try:
            score = liquidity_score['liquidity_score']

            # 주문 규모 추천
            if score >= 0.8:
                recommendations.append({
                    'category': '주문 규모',
                    'recommendation': '대량 주문 가능 (50+ BTC)',
                    'confidence': 'HIGH'
                })
            elif score >= 0.6:
                recommendations.append({
                    'category': '주문 규모',
                    'recommendation': '중형 주문 권장 (10-50 BTC)',
                    'confidence': 'MEDIUM'
                })
            else:
                recommendations.append({
                    'category': '주문 규모',
                    'recommendation': '소형 주문만 권장 (< 10 BTC)',
                    'confidence': 'MEDIUM'
                })

            # 실행 전략 추천
            if slippage['execution_quality'] in ['EXCELLENT', 'GOOD']:
                recommendations.append({
                    'category': '실행 전략',
                    'recommendation': '시장가 주문 가능',
                    'confidence': 'HIGH'
                })
            elif slippage['execution_quality'] == 'FAIR':
                recommendations.append({
                    'category': '실행 전략',
                    'recommendation': '지정가 주문 권장',
                    'confidence': 'MEDIUM'
                })
            else:
                recommendations.append({
                    'category': '실행 전략',
                    'recommendation': 'TWAP/VWAP 전략 사용 권장',
                    'confidence': 'HIGH'
                })

            # 시간대 추천
            recommendations.append({
                'category': '거래 시간',
                'recommendation': '거래량이 많은 시간대 선택 (08:00-10:00, 20:00-22:00)',
                'confidence': 'MEDIUM'
            })

            # 위험 관리
            if regime['regime'] in ['LOW_LIQUIDITY', 'VERY_LOW_LIQUIDITY']:
                recommendations.append({
                    'category': '위험 관리',
                    'recommendation': 'Stop-loss 주문 사용 주의 (슬리피지 위험)',
                    'confidence': 'HIGH'
                })

        except Exception as e:
            self.logger.debug(f"Recommendations generation error: {e}")

        return recommendations

    def _calculate_historical_stats(self):
        """히스토리 통계 계산"""
        try:
            stats = {}

            # 유동성 점수 통계
            if len(self.liquidity_score_history) > 0:
                scores = [ls['liquidity_score'] for ls in self.liquidity_score_history]
                stats['liquidity_score'] = {
                    'current': scores[-1],
                    'average': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }

            # 스프레드 통계
            if len(self.spread_history) > 0:
                spreads = [s['spread_bps'] for s in self.spread_history]
                stats['spread'] = {
                    'current': spreads[-1],
                    'average': np.mean(spreads),
                    'min': np.min(spreads),
                    'max': np.max(spreads)
                }

            # 체제 분포
            if len(self.regime_history) > 0:
                regimes = [r['regime'] for r in self.regime_history]
                regime_counts = {}
                for regime in regimes:
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1
                stats['regime_distribution'] = regime_counts

            return stats

        except Exception as e:
            self.logger.debug(f"Historical stats calculation error: {e}")
            return {}


# ═══════════════════════════════════════════════════════════════════════
# 🔥🔥🔥 7️⃣ 마켓 마이크로스트럭처 분석 (Market Microstructure Analysis) 🔥🔥🔥
# ═══════════════════════════════════════════════════════════════════════

class MarketMicrostructureAnalyzer:
    """
    📊 마켓 마이크로스트럭처 분석 시스템 (Market Microstructure Analysis)
    
    최첨단 시장 미세구조 분석을 통한 초단기 트레이딩 신호 생성
    
    주요 기능:
    1. **Order Flow Imbalance (OFI)** - 주문 흐름 불균형 분석
    2. **VPIN** - Volume-Synchronized Probability of Informed Trading
    3. **Trade Classification** - Lee-Ready, Tick Test, Quote Rule
    4. **Effective Spread & Realized Spread** - 실효 스프레드 분석
    5. **Price Impact** - 거래의 가격 영향력 측정
    6. **Adverse Selection Cost** - 역선택 비용 추정
    7. **Quote Toxicity** - 호가 독성 분석
    8. **HFT Activity Detection** - 고빈도 거래 활동 탐지
    9. **Market Depth Resilience** - 시장 깊이 회복력
    10. **Price Discovery Contribution** - 가격 발견 기여도
    11. **Order Book Pressure** - 호가창 압력 분석
    12. **Trade Aggression Index** - 거래 공격성 지수
    """

    def __init__(self, market_data_manager):
        self.market_data = market_data_manager
        self.logger = get_logger("Microstructure")

        # 📊 히스토리 데이터 저장
        self.ofi_history = deque(maxlen=100)  # Order Flow Imbalance
        self.vpin_history = deque(maxlen=100)  # VPIN
        self.trade_classification_history = deque(maxlen=500)  # Trade Classification
        self.spread_history = deque(maxlen=100)  # Spread Analysis
        self.price_impact_history = deque(maxlen=100)  # Price Impact
        self.toxicity_history = deque(maxlen=100)  # Quote Toxicity
        self.hft_activity_history = deque(maxlen=100)  # HFT Activity
        self.depth_resilience_history = deque(maxlen=50)  # Depth Resilience
        
        # 🎯 임계값 설정
        self.thresholds = {
            'ofi_extreme': 0.7,  # OFI 극단값
            'vpin_high': 0.75,  # VPIN 높음 (정보거래 가능성 높음)
            'vpin_low': 0.25,  # VPIN 낮음
            'toxicity_high': 0.65,  # 높은 독성
            'hft_activity_high': 0.70,  # 높은 HFT 활동
            'adverse_selection_high': 0.008,  # 0.8% 이상
            'price_impact_high': 0.005,  # 0.5% 이상
            'depth_resilience_low': 0.3,  # 낮은 회복력
        }

        # 📦 캐싱
        self._cache = {}
        self._cache_ttl = 10  # 10초 캐시 (마이크로스트럭처는 빠르게 변함)

        # 🎨 VPIN 계산 파라미터
        self.vpin_config = {
            'volume_buckets': 50,  # Volume bucket 수
            'bulk_classification_threshold': 0.8,  # Bulk volume 분류 임계값
            'cdf_confidence': 0.99  # CDF 신뢰 수준
        }

        # 📈 Trade Classification 파라미터
        self.trade_classification_config = {
            'quote_range_seconds': 5,  # Quote 범위 (초)
            'tick_test_lookback': 1,  # Tick test lookback
            'min_price_change': 0.0001  # 최소 가격 변화
        }

        # 🔥 HFT Detection 파라미터
        self.hft_config = {
            'message_rate_threshold': 100,  # 초당 메시지 수
            'cancellation_ratio_threshold': 0.85,  # 취소율
            'quote_update_frequency_threshold': 50,  # 초당 호가 업데이트
            'small_order_ratio_threshold': 0.75  # 소규모 주문 비율
        }

        # 📊 Order Book 분석 파라미터
        self.orderbook_config = {
            'levels': 10,  # 분석할 호가 레벨 수
            'time_window_seconds': 60,  # 분석 시간 윈도우
            'pressure_lookback': 20  # 압력 계산 lookback
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

    # ═══════════════════════════════════════════════════════════════════════
    # 1️⃣ Order Flow Imbalance (OFI) 분석
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_order_flow_imbalance(self, symbol='BTCUSDT', timeframe='1m'):
        """
        Order Flow Imbalance (OFI) 계산
        
        OFI는 매수/매도 주문 흐름의 불균형을 측정하여 단기 가격 방향을 예측
        
        Returns:
            dict: {
                'ofi': float (-1.0 ~ 1.0),
                'buy_volume': float,
                'sell_volume': float,
                'imbalance_strength': str,
                'prediction': str
            }
        """
        cached = self._get_cached_data(f'ofi_{symbol}_{timeframe}')
        if cached:
            return cached

        try:
            # 🔥 실제 구현 시 거래소 API에서 Trade 데이터 수집
            # 시뮬레이션: 최근 거래 데이터 생성
            
            # Buy volume (taker buy)
            buy_volume = np.random.uniform(10, 100) * np.random.choice([1, 1.2, 0.8])
            
            # Sell volume (taker sell)
            sell_volume = np.random.uniform(10, 100) * np.random.choice([1, 1.2, 0.8])
            
            # OFI 계산
            total_volume = buy_volume + sell_volume
            if total_volume > 0:
                ofi = (buy_volume - sell_volume) / total_volume
            else:
                ofi = 0.0
            
            # OFI 강도 분류
            if abs(ofi) > self.thresholds['ofi_extreme']:
                strength = 'EXTREME'
            elif abs(ofi) > 0.5:
                strength = 'STRONG'
            elif abs(ofi) > 0.3:
                strength = 'MODERATE'
            else:
                strength = 'WEAK'
            
            # 예측 신호
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

    # ═══════════════════════════════════════════════════════════════════════
    # 2️⃣ VPIN (Volume-Synchronized Probability of Informed Trading)
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_vpin(self, symbol='BTCUSDT', lookback_hours=24):
        """
        VPIN 계산 - 정보거래 확률 추정
        
        VPIN이 높을수록:
        - 정보 비대칭성이 크다
        - 역선택 위험이 높다
        - 유동성 공급자가 스프레드를 넓힌다
        - 가격 변동성이 증가할 가능성
        
        Returns:
            dict: {
                'vpin': float (0.0 ~ 1.0),
                'toxicity_level': str,
                'informed_trading_probability': float,
                'risk_level': str
            }
        """
        cached = self._get_cached_data(f'vpin_{symbol}')
        if cached:
            return cached

        try:
            # 🔥 실제 구현 시 Volume bucket 기반 VPIN 계산
            # 시뮬레이션: VPIN 추정
            
            # Volume bucket별 buy/sell 불균형 계산
            n_buckets = self.vpin_config['volume_buckets']
            volume_imbalances = []
            
            for _ in range(n_buckets):
                # 각 bucket의 매수/매도 불균형
                buy_vol = np.random.uniform(0, 100)
                sell_vol = np.random.uniform(0, 100)
                imbalance = abs(buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-6)
                volume_imbalances.append(imbalance)
            
            # VPIN = 평균 불균형
            vpin = np.mean(volume_imbalances)
            
            # 추세 고려 (최근이 더 중요)
            recent_weight = np.linspace(0.5, 1.5, n_buckets)
            weighted_vpin = np.average(volume_imbalances, weights=recent_weight)
            vpin = weighted_vpin
            
            # VPIN 정규화 (0 ~ 1)
            vpin = np.clip(vpin, 0.0, 1.0)
            
            # 독성 수준 분류
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
            
            # 정보거래 확률
            informed_trading_probability = vpin
            
            result = {
                'vpin': vpin,
                'toxicity_level': toxicity_level,
                'informed_trading_probability': informed_trading_probability,
                'risk_level': risk_level,
                'volume_imbalances': volume_imbalances[-10:],  # 최근 10개
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

    # ═══════════════════════════════════════════════════════════════════════
    # 3️⃣ Trade Classification (Lee-Ready Algorithm)
    # ═══════════════════════════════════════════════════════════════════════

    def classify_trades(self, symbol='BTCUSDT', n_trades=100):
        """
        거래를 매수자 주도 vs 매도자 주도로 분류
        
        Lee-Ready Algorithm:
        1. Quote Rule: 거래가 매수호가에 가까우면 매수자 주도
        2. Tick Test: 가격이 상승하면 매수자 주도
        
        Returns:
            dict: {
                'buy_initiated_ratio': float,
                'sell_initiated_ratio': float,
                'trade_aggression_index': float,
                'market_direction': str
            }
        """
        try:
            # 🔥 실제 구현 시 실시간 거래 데이터 분류
            # 시뮬레이션
            
            buy_initiated = 0
            sell_initiated = 0
            
            for _ in range(n_trades):
                # Quote rule + Tick test 시뮬레이션
                classification = np.random.choice(
                    ['buy', 'sell'], 
                    p=[0.52, 0.48]  # 약간의 매수 편향
                )
                
                if classification == 'buy':
                    buy_initiated += 1
                else:
                    sell_initiated += 1
            
            # 비율 계산
            buy_ratio = buy_initiated / n_trades
            sell_ratio = sell_initiated / n_trades
            
            # Trade Aggression Index (TAI)
            # -1 (모두 매도자 주도) ~ +1 (모두 매수자 주도)
            tai = (buy_initiated - sell_initiated) / n_trades
            
            # 시장 방향
            if tai > 0.3:
                market_direction = 'STRONG_BUYING'
            elif tai > 0.1:
                market_direction = 'MODERATE_BUYING'
            elif tai < -0.3:
                market_direction = 'STRONG_SELLING'
            elif tai < -0.1:
                market_direction = 'MODERATE_SELLING'
            else:
                market_direction = 'NEUTRAL'
            
            result = {
                'buy_initiated_ratio': buy_ratio,
                'sell_initiated_ratio': sell_ratio,
                'trade_aggression_index': tai,
                'market_direction': market_direction,
                'n_trades_analyzed': n_trades,
                'timestamp': datetime.now()
            }
            
            self.trade_classification_history.append(result)
            
            return result

        except Exception as e:
            self.logger.error(f"Trade classification error: {e}")
            return {
                'buy_initiated_ratio': 0.5,
                'sell_initiated_ratio': 0.5,
                'trade_aggression_index': 0.0,
                'market_direction': 'NEUTRAL',
                'n_trades_analyzed': 0
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 4️⃣ Effective Spread & Realized Spread
    # ═══════════════════════════════════════════════════════════════════════

    def analyze_spreads(self, symbol='BTCUSDT'):
        """
        Effective Spread & Realized Spread 분석
        
        - Quoted Spread: 매수/매도 호가 차이
        - Effective Spread: 실제 거래 비용
        - Realized Spread: 유동성 공급자의 수익
        - Price Impact: 거래의 영구적 가격 영향
        
        Returns:
            dict: {
                'quoted_spread_bps': float,
                'effective_spread_bps': float,
                'realized_spread_bps': float,
                'price_impact_bps': float,
                'adverse_selection_component': float
            }
        """
        try:
            # 🔥 실제 구현 시 호가창 및 거래 데이터 사용
            # 시뮬레이션
            
            mid_price = 50000  # BTC 기준 가격
            
            # Quoted Spread
            best_bid = mid_price * 0.9999
            best_ask = mid_price * 1.0001
            quoted_spread = best_ask - best_bid
            quoted_spread_bps = (quoted_spread / mid_price) * 10000
            
            # Effective Spread (실제 거래 가격 고려)
            trade_price = np.random.uniform(best_bid, best_ask)
            effective_spread = 2 * abs(trade_price - mid_price)
            effective_spread_bps = (effective_spread / mid_price) * 10000
            
            # Realized Spread (5초 후 mid price와 비교)
            future_mid_price = mid_price * (1 + np.random.uniform(-0.0002, 0.0002))
            realized_spread = 2 * (trade_price - future_mid_price) * np.sign(trade_price - mid_price)
            realized_spread_bps = (realized_spread / mid_price) * 10000
            
            # Price Impact (영구적 가격 변화)
            price_impact = effective_spread - realized_spread
            price_impact_bps = (price_impact / mid_price) * 10000
            
            # Adverse Selection Component
            # = Price Impact / Effective Spread
            if effective_spread > 0:
                adverse_selection = abs(price_impact / effective_spread)
            else:
                adverse_selection = 0.0
            
            result = {
                'quoted_spread_bps': quoted_spread_bps,
                'effective_spread_bps': effective_spread_bps,
                'realized_spread_bps': realized_spread_bps,
                'price_impact_bps': abs(price_impact_bps),
                'adverse_selection_component': adverse_selection,
                'timestamp': datetime.now()
            }
            
            self.spread_history.append(result)
            
            return result

        except Exception as e:
            self.logger.error(f"Spread analysis error: {e}")
            return {
                'quoted_spread_bps': 10.0,
                'effective_spread_bps': 12.0,
                'realized_spread_bps': 8.0,
                'price_impact_bps': 4.0,
                'adverse_selection_component': 0.33
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 5️⃣ Quote Toxicity Analysis
    # ═══════════════════════════════════════════════════════════════════════

    def analyze_quote_toxicity(self, symbol='BTCUSDT'):
        """
        호가 독성 분석
        
        높은 독성 = 정보거래자가 많음 = 마켓메이커가 위험을 느낌
        
        Returns:
            dict: {
                'toxicity_score': float (0.0 ~ 1.0),
                'quote_update_frequency': float,
                'cancellation_ratio': float,
                'toxicity_level': str
            }
        """
        try:
            # 🔥 실제 구현 시 호가 업데이트 빈도 및 취소율 분석
            
            # 호가 업데이트 빈도 (초당)
            quote_updates_per_second = np.random.uniform(5, 100)
            
            # 취소율 (취소된 주문 / 전체 주문)
            cancellation_ratio = np.random.uniform(0.3, 0.95)
            
            # 작은 호가 비율 (1 BTC 미만)
            small_quote_ratio = np.random.uniform(0.5, 0.9)
            
            # Toxicity Score 계산
            toxicity_components = [
                min(quote_updates_per_second / 100, 1.0),  # 빈번한 업데이트
                cancellation_ratio,  # 높은 취소율
                small_quote_ratio  # 작은 주문 크기
            ]
            
            toxicity_score = np.mean(toxicity_components)
            
            # 독성 수준
            if toxicity_score > self.thresholds['toxicity_high']:
                toxicity_level = 'HIGH'
            elif toxicity_score > 0.45:
                toxicity_level = 'MODERATE'
            else:
                toxicity_level = 'LOW'
            
            result = {
                'toxicity_score': toxicity_score,
                'quote_update_frequency': quote_updates_per_second,
                'cancellation_ratio': cancellation_ratio,
                'small_quote_ratio': small_quote_ratio,
                'toxicity_level': toxicity_level,
                'timestamp': datetime.now()
            }
            
            self.toxicity_history.append(result)
            
            return result

        except Exception as e:
            self.logger.error(f"Quote toxicity analysis error: {e}")
            return {
                'toxicity_score': 0.5,
                'quote_update_frequency': 20.0,
                'cancellation_ratio': 0.6,
                'toxicity_level': 'MODERATE'
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 6️⃣ HFT Activity Detection
    # ═══════════════════════════════════════════════════════════════════════

    def detect_hft_activity(self, symbol='BTCUSDT'):
        """
        고빈도 거래(HFT) 활동 탐지
        
        HFT 특징:
        - 매우 높은 메시지 속도
        - 높은 주문 취소율
        - 작은 주문 크기
        - 짧은 보유 시간
        
        Returns:
            dict: {
                'hft_activity_score': float (0.0 ~ 1.0),
                'hft_detected': bool,
                'message_rate': float,
                'activity_level': str
            }
        """
        try:
            # 🔥 실제 구현 시 거래소 메시지 스트림 분석
            
            # 메시지 속도 (초당)
            message_rate = np.random.uniform(10, 200)
            
            # 주문 취소율
            cancellation_ratio = np.random.uniform(0.5, 0.95)
            
            # 소규모 주문 비율
            small_order_ratio = np.random.uniform(0.6, 0.95)
            
            # 호가 업데이트 빈도
            quote_update_freq = np.random.uniform(20, 150)
            
            # HFT Activity Score 계산
            hft_components = [
                min(message_rate / self.hft_config['message_rate_threshold'], 1.0),
                cancellation_ratio,
                small_order_ratio,
                min(quote_update_freq / self.hft_config['quote_update_frequency_threshold'], 1.0)
            ]
            
            hft_activity_score = np.mean(hft_components)
            
            # HFT 탐지
            hft_detected = hft_activity_score > self.thresholds['hft_activity_high']
            
            # 활동 수준
            if hft_activity_score > 0.75:
                activity_level = 'VERY_HIGH'
            elif hft_activity_score > 0.55:
                activity_level = 'HIGH'
            elif hft_activity_score > 0.35:
                activity_level = 'MODERATE'
            else:
                activity_level = 'LOW'
            
            result = {
                'hft_activity_score': hft_activity_score,
                'hft_detected': hft_detected,
                'message_rate': message_rate,
                'cancellation_ratio': cancellation_ratio,
                'small_order_ratio': small_order_ratio,
                'activity_level': activity_level,
                'timestamp': datetime.now()
            }
            
            self.hft_activity_history.append(result)
            
            return result

        except Exception as e:
            self.logger.error(f"HFT detection error: {e}")
            return {
                'hft_activity_score': 0.5,
                'hft_detected': False,
                'message_rate': 50.0,
                'activity_level': 'MODERATE'
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 7️⃣ Order Book Pressure Analysis
    # ═══════════════════════════════════════════════════════════════════════

    def analyze_orderbook_pressure(self, symbol='BTCUSDT'):
        """
        호가창 압력 분석
        
        매수/매도 호가의 크기 및 분포를 분석하여 단기 가격 방향 예측
        
        Returns:
            dict: {
                'buy_pressure': float,
                'sell_pressure': float,
                'net_pressure': float,
                'pressure_signal': str,
                'depth_imbalance': float
            }
        """
        try:
            # 🔥 실제 구현 시 호가창 데이터 사용
            
            levels = self.orderbook_config['levels']
            
            # 매수 호가 (bid) 압력
            buy_volumes = [np.random.uniform(0.5, 10.0) * (1 / (i + 1)) for i in range(levels)]
            total_buy_volume = sum(buy_volumes)
            
            # 매도 호가 (ask) 압력
            sell_volumes = [np.random.uniform(0.5, 10.0) * (1 / (i + 1)) for i in range(levels)]
            total_sell_volume = sum(sell_volumes)
            
            # 가중 압력 (가까운 호가에 더 높은 가중치)
            weights = [1 / (i + 1) for i in range(levels)]
            weighted_buy_pressure = np.average(buy_volumes, weights=weights)
            weighted_sell_pressure = np.average(sell_volumes, weights=weights)
            
            # 순 압력
            net_pressure = (total_buy_volume - total_sell_volume) / (total_buy_volume + total_sell_volume)
            
            # Depth Imbalance
            depth_imbalance = net_pressure
            
            # 압력 신호
            if net_pressure > 0.3:
                pressure_signal = 'STRONG_BUY_PRESSURE'
            elif net_pressure > 0.1:
                pressure_signal = 'BUY_PRESSURE'
            elif net_pressure < -0.3:
                pressure_signal = 'STRONG_SELL_PRESSURE'
            elif net_pressure < -0.1:
                pressure_signal = 'SELL_PRESSURE'
            else:
                pressure_signal = 'BALANCED'
            
            result = {
                'buy_pressure': total_buy_volume,
                'sell_pressure': total_sell_volume,
                'net_pressure': net_pressure,
                'pressure_signal': pressure_signal,
                'depth_imbalance': depth_imbalance,
                'weighted_buy_pressure': weighted_buy_pressure,
                'weighted_sell_pressure': weighted_sell_pressure,
                'timestamp': datetime.now()
            }
            
            return result

        except Exception as e:
            self.logger.error(f"Orderbook pressure analysis error: {e}")
            return {
                'buy_pressure': 0.0,
                'sell_pressure': 0.0,
                'net_pressure': 0.0,
                'pressure_signal': 'BALANCED',
                'depth_imbalance': 0.0
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 8️⃣ Market Depth Resilience
    # ═══════════════════════════════════════════════════════════════════════

    def analyze_depth_resilience(self, symbol='BTCUSDT'):
        """
        시장 깊이 회복력 분석
        
        대량 거래 후 호가창이 얼마나 빠르게 회복되는지 측정
        
        Returns:
            dict: {
                'resilience_score': float (0.0 ~ 1.0),
                'recovery_time_seconds': float,
                'resilience_level': str
            }
        """
        try:
            # 🔥 실제 구현 시 호가창 변화 추적
            
            # 회복 시간 시뮬레이션 (초)
            recovery_time = np.random.uniform(1, 30)
            
            # Resilience Score
            # 빠른 회복 = 높은 점수
            max_recovery_time = 30  # 30초
            resilience_score = 1.0 - min(recovery_time / max_recovery_time, 1.0)
            
            # 최근 거래량 변동성 고려
            volume_volatility = np.random.uniform(0.1, 0.5)
            resilience_score *= (1.0 - volume_volatility * 0.5)
            
            resilience_score = np.clip(resilience_score, 0.0, 1.0)
            
            # 회복력 수준
            if resilience_score > 0.7:
                resilience_level = 'HIGH'
            elif resilience_score > 0.4:
                resilience_level = 'MODERATE'
            else:
                resilience_level = 'LOW'
            
            result = {
                'resilience_score': resilience_score,
                'recovery_time_seconds': recovery_time,
                'resilience_level': resilience_level,
                'timestamp': datetime.now()
            }
            
            self.depth_resilience_history.append(result)
            
            return result

        except Exception as e:
            self.logger.error(f"Depth resilience analysis error: {e}")
            return {
                'resilience_score': 0.5,
                'recovery_time_seconds': 15.0,
                'resilience_level': 'MODERATE'
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 9️⃣ 종합 마이크로스트럭처 신호
    # ═══════════════════════════════════════════════════════════════════════

    def get_comprehensive_microstructure_signal(self, symbol='BTCUSDT'):
        """
        모든 마이크로스트럭처 지표를 종합한 신호
        
        Returns:
            dict: {
                'microstructure_score': float (-1.0 ~ 1.0),
                'signal': str,
                'confidence': float,
                'components': dict,
                'trading_recommendation': str
            }
        """
        try:
            # 모든 지표 수집
            ofi = self.calculate_order_flow_imbalance(symbol)
            vpin = self.calculate_vpin(symbol)
            trade_class = self.classify_trades(symbol)
            spreads = self.analyze_spreads(symbol)
            toxicity = self.analyze_quote_toxicity(symbol)
            hft = self.detect_hft_activity(symbol)
            ob_pressure = self.analyze_orderbook_pressure(symbol)
            resilience = self.analyze_depth_resilience(symbol)
            
            # 각 지표의 점수화 (-1.0 ~ 1.0)
            scores = {}
            
            # 1. OFI 점수
            scores['ofi'] = ofi['ofi']
            
            # 2. VPIN 점수 (높을수록 부정적 - 역선택 위험)
            scores['vpin'] = -(vpin['vpin'] - 0.5) * 2  # 0.5를 중심으로 역변환
            
            # 3. Trade Classification 점수
            scores['trade_class'] = trade_class['trade_aggression_index']
            
            # 4. Spread 점수 (낮은 adverse selection = 긍정적)
            if spreads['adverse_selection_component'] < self.thresholds['adverse_selection_high']:
                scores['spreads'] = 0.5
            else:
                scores['spreads'] = -0.5
            
            # 5. Toxicity 점수 (낮을수록 좋음)
            scores['toxicity'] = -(toxicity['toxicity_score'] - 0.5) * 2
            
            # 6. HFT 점수 (moderate HFT는 유동성 제공, extreme은 부정적)
            if 0.3 < hft['hft_activity_score'] < 0.7:
                scores['hft'] = 0.3
            else:
                scores['hft'] = -0.2
            
            # 7. Order Book Pressure 점수
            scores['ob_pressure'] = ob_pressure['net_pressure']
            
            # 8. Resilience 점수
            scores['resilience'] = (resilience['resilience_score'] - 0.5) * 2
            
            # 가중 평균 계산
            weights = {
                'ofi': 0.20,
                'vpin': 0.15,
                'trade_class': 0.18,
                'spreads': 0.12,
                'toxicity': 0.10,
                'hft': 0.05,
                'ob_pressure': 0.15,
                'resilience': 0.05
            }
            
            microstructure_score = sum(scores[k] * weights[k] for k in scores)
            microstructure_score = np.clip(microstructure_score, -1.0, 1.0)
            
            # 신호 생성
            if microstructure_score > 0.5:
                signal = 'STRONG_BUY'
                trading_recommendation = '강력한 매수 신호 - 공격적 진입 가능'
            elif microstructure_score > 0.2:
                signal = 'BUY'
                trading_recommendation = '매수 신호 - 점진적 진입 권장'
            elif microstructure_score < -0.5:
                signal = 'STRONG_SELL'
                trading_recommendation = '강력한 매도 신호 - 포지션 청산 고려'
            elif microstructure_score < -0.2:
                signal = 'SELL'
                trading_recommendation = '매도 신호 - 포지션 축소 권장'
            else:
                signal = 'NEUTRAL'
                trading_recommendation = '중립 - 관망 권장'
            
            # 신뢰도 계산
            confidence = self._calculate_microstructure_confidence(
                ofi, vpin, trade_class, spreads, toxicity, hft, ob_pressure, resilience
            )
            
            return {
                'microstructure_score': microstructure_score,
                'signal': signal,
                'confidence': confidence,
                'components': {
                    'ofi': ofi,
                    'vpin': vpin,
                    'trade_classification': trade_class,
                    'spreads': spreads,
                    'toxicity': toxicity,
                    'hft_activity': hft,
                    'orderbook_pressure': ob_pressure,
                    'depth_resilience': resilience
                },
                'component_scores': scores,
                'trading_recommendation': trading_recommendation,
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Comprehensive microstructure signal error: {e}")
            return {
                'microstructure_score': 0.0,
                'signal': 'NEUTRAL',
                'confidence': 0.5,
                'components': {},
                'trading_recommendation': '데이터 부족 - 거래 보류'
            }

    def _calculate_microstructure_confidence(self, ofi, vpin, trade_class, spreads, 
                                           toxicity, hft, ob_pressure, resilience):
        """마이크로스트럭처 신호 신뢰도 계산"""
        try:
            confidence_factors = []
            
            # 1. OFI 강도
            if ofi['imbalance_strength'] in ['STRONG', 'EXTREME']:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.5)
            
            # 2. VPIN 수준 (낮을수록 신뢰)
            confidence_factors.append(1.0 - vpin['vpin'])
            
            # 3. Trade Classification 명확성
            if abs(trade_class['trade_aggression_index']) > 0.3:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.5)
            
            # 4. Spread 안정성
            if spreads['adverse_selection_component'] < 0.5:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.4)
            
            # 5. 낮은 독성 = 높은 신뢰
            confidence_factors.append(1.0 - toxicity['toxicity_score'])
            
            # 6. Moderate HFT = 유동성 = 높은 신뢰
            if 0.3 < hft['hft_activity_score'] < 0.7:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
            
            # 7. Order Book Pressure 명확성
            if abs(ob_pressure['net_pressure']) > 0.3:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.5)
            
            # 8. 높은 Resilience = 높은 신뢰
            confidence_factors.append(resilience['resilience_score'])
            
            # 평균 신뢰도
            confidence = np.mean(confidence_factors)
            confidence = np.clip(confidence, 0.0, 1.0)
            
            return confidence

        except Exception as e:
            self.logger.debug(f"Microstructure confidence calculation error: {e}")
            return 0.5

    # ═══════════════════════════════════════════════════════════════════════
    # 🔟 종합 리포트
    # ═══════════════════════════════════════════════════════════════════════

    def get_comprehensive_microstructure_report(self, symbol='BTCUSDT'):
        """
        마켓 마이크로스트럭처 종합 리포트
        
        Returns:
            dict: 모든 마이크로스트럭처 분석 결과
        """
        try:
            # 종합 신호 생성
            comprehensive_signal = self.get_comprehensive_microstructure_signal(symbol)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'comprehensive_signal': comprehensive_signal,
                'historical_stats': {
                    'ofi_history_length': len(self.ofi_history),
                    'vpin_history_length': len(self.vpin_history),
                    'recent_ofi_avg': np.mean([h['ofi'] for h in list(self.ofi_history)[-10:]]) if len(self.ofi_history) > 0 else 0.0,
                    'recent_vpin_avg': np.mean([h['vpin'] for h in list(self.vpin_history)[-10:]]) if len(self.vpin_history) > 0 else 0.5
                }
            }

        except Exception as e:
            self.logger.error(f"Comprehensive microstructure report error: {e}")
            return {}


# ═══════════════════════════════════════════════════════════════════════
# 🔥🔥🔥 4️⃣ 다차원 Regime Confidence Scoring (분산 기반) 🔥🔥🔥
# ═══════════════════════════════════════════════════════════════════════

class MultiDimensionalConfidenceScorer:
    """
    🎯 다차원 Regime Confidence Scoring 시스템
    - 지표 일치도 분석 (Indicator Agreement)
    - 시계열 안정성 분석 (Temporal Stability)
    - 분산 기반 신뢰도 (Variance-Based Confidence)
    - 통계적 신뢰 구간 (Statistical Confidence Interval)
    - 앙상블 신뢰도 (Ensemble Confidence)
    - 불확실성 정량화 (Uncertainty Quantification)
    """

    def __init__(self):
        self.logger = get_logger("ConfidenceScorer")

        # 📊 히스토리 데이터 저장
        self.regime_score_history = deque(maxlen=100)  # Regime 점수 히스토리
        self.indicator_history = deque(maxlen=100)  # 지표 히스토리
        self.confidence_history = deque(maxlen=100)  # 신뢰도 히스토리

        # 🎚️ 신뢰도 레벨 임계값
        self.confidence_levels = {
            'very_high': 0.85,
            'high': 0.70,
            'medium': 0.55,
            'low': 0.40,
            'very_low': 0.25
        }

        # 🎯 앙상블 파라미터
        self.ensemble_config = {
            'bootstrap_samples': 50,  # Bootstrap 샘플 수
            'monte_carlo_iterations': 100,  # 몬테카를로 시뮬레이션 반복
            'confidence_interval': 0.95  # 95% 신뢰구간
        }

        # 📈 성능 메트릭
        self.performance_metrics = {
            'prediction_accuracy': deque(maxlen=50),
            'false_positive_rate': deque(maxlen=50),
            'false_negative_rate': deque(maxlen=50)
        }

    # ═══════════════════════════════════════════════════════════════════════
    # 1️⃣ 지표 일치도 분석 (Indicator Agreement Analysis)
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_indicator_agreement(self, indicators, regime):
        """
        여러 지표가 특정 regime과 얼마나 일치하는지 분석

        Returns:
            dict: {
                'agreement_score': float (0.0 ~ 1.0),
                'agreeing_indicators': list,
                'disagreeing_indicators': list,
                'neutral_indicators': list
            }
        """
        try:
            agreeing = []
            disagreeing = []
            neutral = []

            # Regime에 대한 각 지표의 일치도 평가

            # 1️⃣ Trend 일치도
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

            elif 'SIDEWAYS' in regime or regime in ['ACCUMULATION', 'DISTRIBUTION']:
                if trend == 'sideways':
                    agreeing.append(('trend', 1.0))
                else:
                    disagreeing.append(('trend', -0.5))

            # 2️⃣ Volatility 일치도
            volatility = indicators.get('volatility', 'medium')
            if 'VOLATILITY' in regime or 'CHOP' in regime:
                if volatility in ['high', 'extreme']:
                    agreeing.append(('volatility', 1.0))
                elif volatility == 'medium':
                    neutral.append(('volatility', 0.5))
                else:
                    disagreeing.append(('volatility', -1.0))

            elif 'CONSOLIDATION' in regime or 'COMPRESSION' in regime:
                if volatility in ['low', 'medium']:
                    agreeing.append(('volatility', 1.0))
                else:
                    disagreeing.append(('volatility', -1.0))

            # 3️⃣ Momentum 일치도
            momentum = indicators.get('momentum', 'neutral')
            if 'BULL' in regime:
                if momentum in ['bullish', 'overbought']:
                    agreeing.append(('momentum', 1.0))
                elif momentum == 'neutral':
                    neutral.append(('momentum', 0.5))
                else:
                    disagreeing.append(('momentum', -1.0))

            elif 'BEAR' in regime:
                if momentum in ['bearish', 'oversold']:
                    agreeing.append(('momentum', 1.0))
                elif momentum == 'neutral':
                    neutral.append(('momentum', 0.5))
                else:
                    disagreeing.append(('momentum', -1.0))

            # 4️⃣ Volume 일치도
            volume = indicators.get('volume', 'normal')
            if regime in ['ACCUMULATION', 'DISTRIBUTION']:
                if volume in ['high', 'normal']:
                    agreeing.append(('volume', 0.8))
                else:
                    disagreeing.append(('volume', -0.6))

            # 5️⃣ Breadth 일치도
            breadth = indicators.get('breadth', 0.5)
            if 'BULL' in regime or regime == 'ACCUMULATION':
                if breadth > 0.6:
                    agreeing.append(('breadth', 1.0))
                elif breadth > 0.4:
                    neutral.append(('breadth', 0.5))
                else:
                    disagreeing.append(('breadth', -1.0))

            elif 'BEAR' in regime or regime == 'DISTRIBUTION':
                if breadth < 0.4:
                    agreeing.append(('breadth', 1.0))
                elif breadth < 0.6:
                    neutral.append(('breadth', 0.5))
                else:
                    disagreeing.append(('breadth', -1.0))

            # 6️⃣ 온체인/매크로 신호 일치도
            onchain_macro_signals = indicators.get('onchain_macro_signals')
            if onchain_macro_signals:
                merged_signal = onchain_macro_signals['merged']
                merged_score = merged_signal['score']

                if 'BULL' in regime or regime == 'ACCUMULATION':
                    if merged_score > 0.3:
                        agreeing.append(('onchain_macro', merged_score))
                    elif merged_score > -0.3:
                        neutral.append(('onchain_macro', 0.5))
                    else:
                        disagreeing.append(('onchain_macro', merged_score))

                elif 'BEAR' in regime or regime == 'DISTRIBUTION':
                    if merged_score < -0.3:
                        agreeing.append(('onchain_macro', abs(merged_score)))
                    elif merged_score < 0.3:
                        neutral.append(('onchain_macro', 0.5))
                    else:
                        disagreeing.append(('onchain_macro', -merged_score))

            # 🔥 7️⃣ 유동성 신호 일치도
            liquidity_signals = indicators.get('liquidity_signals')
            if liquidity_signals:
                liquidity_regime = liquidity_signals.get('regime', 'MEDIUM_LIQUIDITY')
                
                # 유동성 체제와 시장 체제의 일치도
                if 'HIGH' in liquidity_regime:
                    # 높은 유동성은 일반적으로 긍정적
                    agreeing.append(('liquidity', 0.6))
                elif 'LOW' in liquidity_regime or 'VERY_LOW' in liquidity_regime:
                    # 낮은 유동성은 위험 신호
                    if 'VOLATILITY' in regime or 'CHOP' in regime:
                        agreeing.append(('liquidity', 0.8))  # 낮은 유동성 + 변동성 = 일치
                    else:
                        disagreeing.append(('liquidity', -0.7))

            # 일치도 점수 계산
            total_indicators = len(agreeing) + len(disagreeing) + len(neutral)
            if total_indicators == 0:
                agreement_score = 0.5
            else:
                agree_weight = sum(score for _, score in agreeing)
                disagree_weight = sum(abs(score) for _, score in disagreeing)
                neutral_weight = sum(score for _, score in neutral) * 0.5

                # 정규화된 일치도 점수
                agreement_score = (agree_weight + neutral_weight) / (
                            agree_weight + disagree_weight + neutral_weight + 1e-6)
                agreement_score = np.clip(agreement_score, 0.0, 1.0)

            return {
                'agreement_score': agreement_score,
                'agreeing_indicators': [name for name, _ in agreeing],
                'disagreeing_indicators': [name for name, _ in disagreeing],
                'neutral_indicators': [name for name, _ in neutral],
                'total_indicators': total_indicators,
                'agree_count': len(agreeing),
                'disagree_count': len(disagreeing),
                'neutral_count': len(neutral)
            }

        except Exception as e:
            self.logger.error(f"Indicator agreement calculation error: {e}")
            return {
                'agreement_score': 0.5,
                'agreeing_indicators': [],
                'disagreeing_indicators': [],
                'neutral_indicators': [],
                'total_indicators': 0,
                'agree_count': 0,
                'disagree_count': 0,
                'neutral_count': 0
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 2️⃣ 시계열 안정성 분석 (Temporal Stability Analysis)
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_temporal_stability(self, window=20):
        """
        최근 시계열 데이터의 안정성 분석

        Args:
            window: 분석할 히스토리 윈도우 크기

        Returns:
            dict: {
                'stability_score': float (0.0 ~ 1.0),
                'regime_consistency': float,
                'score_volatility': float,
                'trend_direction': str
            }
        """
        try:
            if len(self.regime_score_history) < 5:
                return {
                    'stability_score': 0.7,
                    'regime_consistency': 0.7,
                    'score_volatility': 0.0,
                    'trend_direction': 'stable'
                }

            # 최근 데이터 추출
            recent_data = list(self.regime_score_history)[-min(window, len(self.regime_score_history)):]
            recent_regimes = [d['regime'] for d in recent_data]
            recent_scores = [d['score'] for d in recent_data]

            # 1️⃣ Regime 일관성 (같은 regime이 유지되는 비율)
            if len(recent_regimes) > 0:
                most_common_regime = max(set(recent_regimes), key=recent_regimes.count)
                regime_consistency = recent_regimes.count(most_common_regime) / len(recent_regimes)
            else:
                regime_consistency = 1.0

            # 2️⃣ 점수 변동성 (낮을수록 안정적)
            if len(recent_scores) > 1:
                score_std = np.std(recent_scores)
                score_mean = np.mean(recent_scores)
                # 변동계수 (Coefficient of Variation)
                cv = score_std / (abs(score_mean) + 1e-6)
                score_volatility = min(cv, 1.0)
            else:
                score_volatility = 0.0

            # 3️⃣ 추세 방향성
            if len(recent_scores) > 3:
                # 선형 회귀로 추세 파악
                x = np.arange(len(recent_scores))
                slope, _ = np.polyfit(x, recent_scores, 1)

                if slope > 0.05:
                    trend_direction = 'strengthening'
                elif slope < -0.05:
                    trend_direction = 'weakening'
                else:
                    trend_direction = 'stable'
            else:
                trend_direction = 'stable'

            # 4️⃣ 종합 안정성 점수
            # 높은 일관성 + 낮은 변동성 = 높은 안정성
            stability_score = (
                    regime_consistency * 0.6 +  # Regime 일관성 (60%)
                    (1.0 - score_volatility) * 0.4  # 점수 안정성 (40%)
            )
            stability_score = np.clip(stability_score, 0.0, 1.0)

            return {
                'stability_score': stability_score,
                'regime_consistency': regime_consistency,
                'score_volatility': score_volatility,
                'trend_direction': trend_direction
            }

        except Exception as e:
            self.logger.error(f"Temporal stability calculation error: {e}")
            return {
                'stability_score': 0.7,
                'regime_consistency': 0.7,
                'score_volatility': 0.0,
                'trend_direction': 'stable'
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 3️⃣ 분산 기반 신뢰도 (Variance-Based Confidence)
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_variance_based_confidence(self, regime_scores_dict):
        """
        여러 regime 점수의 분산을 기반으로 신뢰도 계산

        Args:
            regime_scores_dict: {'REGIME_NAME': score, ...}

        Returns:
            dict: {
                'variance_confidence': float (0.0 ~ 1.0),
                'score_spread': float,
                'dominant_regime_margin': float,
                'entropy': float
            }
        """
        try:
            if not regime_scores_dict or len(regime_scores_dict) == 0:
                return {
                    'variance_confidence': 0.5,
                    'score_spread': 0.0,
                    'dominant_regime_margin': 0.0,
                    'entropy': 0.0
                }

            scores = np.array(list(regime_scores_dict.values()))
            scores = np.clip(scores, 0.0, 10.0)  # 이상치 제거

            # 1️⃣ 점수 분산 (낮을수록 명확함)
            score_variance = np.var(scores)
            score_std = np.std(scores)
            score_range = np.max(scores) - np.min(scores)

            # 2️⃣ 최고 점수와 2등 점수의 차이 (클수록 명확함)
            sorted_scores = np.sort(scores)[::-1]  # 내림차순
            if len(sorted_scores) >= 2:
                dominant_margin = sorted_scores[0] - sorted_scores[1]
            else:
                dominant_margin = sorted_scores[0] if len(sorted_scores) > 0 else 0.0

            # 3️⃣ 엔트로피 계산 (낮을수록 확실함)
            # 점수를 확률 분포로 변환
            if np.sum(scores) > 0:
                prob_dist = scores / np.sum(scores)
                score_entropy = entropy(prob_dist + 1e-10)  # 로그(0) 방지
                # 정규화 (최대 엔트로피는 log(N))
                max_entropy = np.log(len(scores)) if len(scores) > 1 else 1.0
                normalized_entropy = score_entropy / max_entropy if max_entropy > 0 else 0.0
            else:
                normalized_entropy = 1.0

            # 4️⃣ 종합 분산 기반 신뢰도
            # 높은 마진 + 낮은 분산 + 낮은 엔트로피 = 높은 신뢰도
            variance_confidence = (
                    min(dominant_margin / 2.0, 1.0) * 0.5 +  # 우세 마진 (50%)
                    (1.0 - min(score_std / 2.0, 1.0)) * 0.3 +  # 낮은 표준편차 (30%)
                    (1.0 - normalized_entropy) * 0.2  # 낮은 엔트로피 (20%)
            )
            variance_confidence = np.clip(variance_confidence, 0.0, 1.0)

            return {
                'variance_confidence': variance_confidence,
                'score_spread': score_range,
                'dominant_regime_margin': dominant_margin,
                'entropy': normalized_entropy,
                'score_std': score_std
            }

        except Exception as e:
            self.logger.error(f"Variance-based confidence calculation error: {e}")
            return {
                'variance_confidence': 0.5,
                'score_spread': 0.0,
                'dominant_regime_margin': 0.0,
                'entropy': 0.0,
                'score_std': 0.0
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 4️⃣ 통계적 신뢰 구간 (Statistical Confidence Interval)
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_statistical_confidence_interval(self, window=30):
        """
        Bootstrap 방법을 사용한 통계적 신뢰 구간 계산

        Args:
            window: 분석할 히스토리 윈도우 크기

        Returns:
            dict: {
                'mean_score': float,
                'confidence_interval': tuple (lower, upper),
                'interval_width': float,
                'statistical_confidence': float (0.0 ~ 1.0)
            }
        """
        try:
            if len(self.regime_score_history) < 10:
                return {
                    'mean_score': 0.0,
                    'confidence_interval': (0.0, 0.0),
                    'interval_width': 0.0,
                    'statistical_confidence': 0.6
                }

            # 최근 데이터 추출
            recent_scores = [d['score'] for d in
                             list(self.regime_score_history)[-min(window, len(self.regime_score_history)):]]
            recent_scores = np.array(recent_scores)

            # Bootstrap 샘플링
            bootstrap_means = []
            n_samples = len(recent_scores)

            for _ in range(self.ensemble_config['bootstrap_samples']):
                # 복원 추출
                bootstrap_sample = np.random.choice(recent_scores, size=n_samples, replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))

            bootstrap_means = np.array(bootstrap_means)

            # 신뢰 구간 계산 (95%)
            alpha = 1 - self.ensemble_config['confidence_interval']
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            ci_lower = np.percentile(bootstrap_means, lower_percentile)
            ci_upper = np.percentile(bootstrap_means, upper_percentile)
            ci_width = ci_upper - ci_lower

            mean_score = np.mean(recent_scores)

            # 신뢰도 계산 (구간이 좁을수록 높은 신뢰도)
            # 구간 폭을 점수 범위(-1 ~ 1)로 정규화
            normalized_width = ci_width / 2.0  # 최대 폭 = 2.0
            statistical_confidence = 1.0 - min(normalized_width, 1.0)
            statistical_confidence = np.clip(statistical_confidence, 0.0, 1.0)

            return {
                'mean_score': float(mean_score),
                'confidence_interval': (float(ci_lower), float(ci_upper)),
                'interval_width': float(ci_width),
                'statistical_confidence': float(statistical_confidence)
            }

        except Exception as e:
            self.logger.error(f"Statistical confidence interval calculation error: {e}")
            return {
                'mean_score': 0.0,
                'confidence_interval': (0.0, 0.0),
                'interval_width': 0.0,
                'statistical_confidence': 0.6
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 5️⃣ 앙상블 신뢰도 (Ensemble Confidence)
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_ensemble_confidence(self, indicators, regime_scores_dict):
        """
        다중 시간프레임 및 방법론을 결합한 앙상블 신뢰도

        Returns:
            dict: {
                'ensemble_confidence': float (0.0 ~ 1.0),
                'method_agreement': float,
                'robust_score': float
            }
        """
        try:
            # 여러 방법으로 계산된 신뢰도 수집
            confidence_scores = []

            # 1️⃣ 지표 일치도 기반
            agreement_result = self.calculate_indicator_agreement(
                indicators,
                max(regime_scores_dict, key=regime_scores_dict.get)
            )
            confidence_scores.append(agreement_result['agreement_score'])

            # 2️⃣ 시계열 안정성 기반
            stability_result = self.calculate_temporal_stability()
            confidence_scores.append(stability_result['stability_score'])

            # 3️⃣ 분산 기반
            variance_result = self.calculate_variance_based_confidence(regime_scores_dict)
            confidence_scores.append(variance_result['variance_confidence'])

            # 4️⃣ 통계적 신뢰 구간 기반
            statistical_result = self.calculate_statistical_confidence_interval()
            confidence_scores.append(statistical_result['statistical_confidence'])

            # 방법 간 일치도 계산
            if len(confidence_scores) > 1:
                method_std = np.std(confidence_scores)
                method_agreement = 1.0 - min(method_std / 0.5, 1.0)  # 표준편차가 0.5 이하면 높은 일치도
            else:
                method_agreement = 1.0

            # 앙상블 신뢰도 = 가중 평균
            weights = [0.30, 0.25, 0.25, 0.20]  # 각 방법의 가중치
            ensemble_confidence = np.average(confidence_scores, weights=weights)

            # 방법 간 일치도가 낮으면 페널티
            if method_agreement < 0.7:
                ensemble_confidence *= 0.9

            # 로버스트 점수 = 중앙값 (이상치에 강건)
            robust_score = np.median(confidence_scores)

            ensemble_confidence = np.clip(ensemble_confidence, 0.0, 1.0)

            return {
                'ensemble_confidence': float(ensemble_confidence),
                'method_agreement': float(method_agreement),
                'robust_score': float(robust_score),
                'individual_scores': {
                    'agreement': confidence_scores[0],
                    'stability': confidence_scores[1],
                    'variance': confidence_scores[2],
                    'statistical': confidence_scores[3]
                }
            }

        except Exception as e:
            self.logger.error(f"Ensemble confidence calculation error: {e}")
            return {
                'ensemble_confidence': 0.6,
                'method_agreement': 0.7,
                'robust_score': 0.6,
                'individual_scores': {}
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 6️⃣ 불확실성 정량화 (Uncertainty Quantification)
    # ═══════════════════════════════════════════════════════════════════════

    def quantify_uncertainty(self, regime, regime_scores_dict, indicators):
        """
        베이지안 및 몬테카를로 방법을 사용한 불확실성 정량화

        Returns:
            dict: {
                'uncertainty_score': float (0.0 ~ 1.0, 높을수록 불확실)
                'prediction_interval': tuple (lower, upper),
                'risk_level': str
            }
        """
        try:
            uncertainty_factors = []

            # 1️⃣ 점수 분산 기반 불확실성
            if regime_scores_dict:
                scores = list(regime_scores_dict.values())
                score_std = np.std(scores)
                normalized_std = min(score_std / 2.0, 1.0)
                uncertainty_factors.append(normalized_std)

            # 2️⃣ 지표 불일치 기반 불확실성
            agreement_result = self.calculate_indicator_agreement(indicators, regime)
            disagreement_ratio = agreement_result['disagree_count'] / max(agreement_result['total_indicators'], 1)
            uncertainty_factors.append(disagreement_ratio)

            # 3️⃣ 시계열 불안정성 기반 불확실성
            stability_result = self.calculate_temporal_stability()
            instability = 1.0 - stability_result['stability_score']
            uncertainty_factors.append(instability)

            # 4️⃣ 엔트로피 기반 불확실성
            variance_result = self.calculate_variance_based_confidence(regime_scores_dict)
            entropy_uncertainty = variance_result['entropy']
            uncertainty_factors.append(entropy_uncertainty)

            # 종합 불확실성 점수
            uncertainty_score = np.mean(uncertainty_factors)
            uncertainty_score = np.clip(uncertainty_score, 0.0, 1.0)

            # 몬테카를로 시뮬레이션으로 예측 구간 추정
            if len(self.regime_score_history) >= 10:
                recent_scores = [d['score'] for d in list(self.regime_score_history)[-20:]]
                mean_score = np.mean(recent_scores)
                std_score = np.std(recent_scores)

                # 정규분포 가정하에 예측 구간
                z_score = 1.96  # 95% 신뢰구간
                margin = z_score * std_score
                prediction_interval = (mean_score - margin, mean_score + margin)
            else:
                prediction_interval = (-1.0, 1.0)

            # 리스크 레벨 결정
            if uncertainty_score < 0.3:
                risk_level = 'LOW'
            elif uncertainty_score < 0.5:
                risk_level = 'MEDIUM'
            elif uncertainty_score < 0.7:
                risk_level = 'HIGH'
            else:
                risk_level = 'VERY_HIGH'

            return {
                'uncertainty_score': float(uncertainty_score),
                'prediction_interval': prediction_interval,
                'risk_level': risk_level,
                'uncertainty_components': {
                    'score_variance': uncertainty_factors[0] if len(uncertainty_factors) > 0 else 0.0,
                    'indicator_disagreement': uncertainty_factors[1] if len(uncertainty_factors) > 1 else 0.0,
                    'temporal_instability': uncertainty_factors[2] if len(uncertainty_factors) > 2 else 0.0,
                    'entropy': uncertainty_factors[3] if len(uncertainty_factors) > 3 else 0.0
                }
            }

        except Exception as e:
            self.logger.error(f"Uncertainty quantification error: {e}")
            return {
                'uncertainty_score': 0.5,
                'prediction_interval': (-1.0, 1.0),
                'risk_level': 'MEDIUM',
                'uncertainty_components': {}
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 7️⃣ 종합 다차원 신뢰도 계산 (Comprehensive Multi-Dimensional Confidence)
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_comprehensive_confidence(self, regime, regime_scores_dict, indicators):
        """
        모든 차원을 통합한 종합 신뢰도 점수 계산

        Returns:
            dict: {
                'overall_confidence': float (0.0 ~ 1.0),
                'confidence_level': str,
                'detailed_scores': dict,
                'risk_assessment': dict
            }
        """
        try:
            # 1️⃣ 앙상블 신뢰도
            ensemble_result = self.calculate_ensemble_confidence(indicators, regime_scores_dict)

            # 2️⃣ 불확실성 정량화
            uncertainty_result = self.quantify_uncertainty(regime, regime_scores_dict, indicators)

            # 3️⃣ 지표 일치도
            agreement_result = self.calculate_indicator_agreement(indicators, regime)

            # 4️⃣ 시계열 안정성
            stability_result = self.calculate_temporal_stability()

            # 5️⃣ 분산 기반 신뢰도
            variance_result = self.calculate_variance_based_confidence(regime_scores_dict)

            # 6️⃣ 통계적 신뢰도
            statistical_result = self.calculate_statistical_confidence_interval()

            # 종합 신뢰도 계산 (가중 평균)
            overall_confidence = (
                    ensemble_result['ensemble_confidence'] * 0.35 +  # 앙상블 (35%)
                    (1.0 - uncertainty_result['uncertainty_score']) * 0.25 +  # 불확실성 (25%, 역수)
                    agreement_result['agreement_score'] * 0.15 +  # 지표 일치도 (15%)
                    stability_result['stability_score'] * 0.15 +  # 시계열 안정성 (15%)
                    variance_result['variance_confidence'] * 0.10  # 분산 기반 (10%)
            )
            overall_confidence = np.clip(overall_confidence, 0.0, 1.0)

            # 신뢰도 레벨 결정
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

            # 히스토리 저장
            self.confidence_history.append({
                'timestamp': datetime.now(),
                'regime': regime,
                'confidence': overall_confidence,
                'level': confidence_level
            })

            # 상세 결과 반환
            return {
                'overall_confidence': float(overall_confidence),
                'confidence_level': confidence_level,
                'confidence_percentage': float(overall_confidence * 100),
                'detailed_scores': {
                    'ensemble': ensemble_result,
                    'uncertainty': uncertainty_result,
                    'agreement': agreement_result,
                    'stability': stability_result,
                    'variance': variance_result,
                    'statistical': statistical_result
                },
                'risk_assessment': {
                    'risk_level': uncertainty_result['risk_level'],
                    'uncertainty_score': uncertainty_result['uncertainty_score'],
                    'prediction_interval': uncertainty_result['prediction_interval']
                },
                'key_insights': self._generate_confidence_insights(
                    overall_confidence,
                    ensemble_result,
                    uncertainty_result,
                    agreement_result
                )
            }

        except Exception as e:
            self.logger.error(f"Comprehensive confidence calculation error: {e}")
            return {
                'overall_confidence': 0.6,
                'confidence_level': 'MEDIUM',
                'confidence_percentage': 60.0,
                'detailed_scores': {},
                'risk_assessment': {},
                'key_insights': []
            }

    def _generate_confidence_insights(self, overall_confidence, ensemble_result,
                                      uncertainty_result, agreement_result):
        """신뢰도 기반 인사이트 생성"""
        insights = []

        try:
            # 1️⃣ 전체 신뢰도 평가
            if overall_confidence >= 0.85:
                insights.append("🟢 매우 높은 신뢰도 - 강력한 신호")
            elif overall_confidence >= 0.70:
                insights.append("🟡 높은 신뢰도 - 신뢰 가능한 신호")
            elif overall_confidence >= 0.55:
                insights.append("🟠 중간 신뢰도 - 주의 필요")
            else:
                insights.append("🔴 낮은 신뢰도 - 불확실한 신호")

            # 2️⃣ 방법 간 일치도
            if ensemble_result['method_agreement'] >= 0.8:
                insights.append("✅ 모든 분석 방법이 일치")
            elif ensemble_result['method_agreement'] < 0.6:
                insights.append("⚠️ 분석 방법 간 불일치 존재")

            # 3️⃣ 불확실성 평가
            if uncertainty_result['risk_level'] == 'LOW':
                insights.append("💎 낮은 불확실성 - 안정적 예측")
            elif uncertainty_result['risk_level'] in ['HIGH', 'VERY_HIGH']:
                insights.append("⚡ 높은 불확실성 - 변동성 주의")

            # 4️⃣ 지표 일치도
            agree_ratio = agreement_result['agree_count'] / max(agreement_result['total_indicators'], 1)
            if agree_ratio >= 0.75:
                insights.append("🎯 지표 강한 일치")
            elif agreement_result['disagree_count'] >= agreement_result['agree_count']:
                insights.append("🔀 지표 간 불일치")

        except Exception as e:
            self.logger.debug(f"Insights generation error: {e}")

        return insights

    def update_history(self, regime, score, indicators):
        """히스토리 업데이트"""
        try:
            self.regime_score_history.append({
                'timestamp': datetime.now(),
                'regime': regime,
                'score': score
            })

            self.indicator_history.append({
                'timestamp': datetime.now(),
                'indicators': indicators.copy()
            })
        except Exception as e:
            self.logger.debug(f"History update error: {e}")

    def get_confidence_report(self):
        """신뢰도 분석 리포트 생성"""
        try:
            if len(self.confidence_history) == 0:
                return {}

            recent_confidences = [c['confidence'] for c in list(self.confidence_history)[-20:]]

            return {
                'current_confidence': self.confidence_history[-1]['confidence'] if len(
                    self.confidence_history) > 0 else 0.0,
                'average_confidence': np.mean(recent_confidences),
                'confidence_trend': 'increasing' if len(recent_confidences) > 1 and recent_confidences[-1] >
                                                    recent_confidences[0] else 'stable',
                'confidence_volatility': np.std(recent_confidences),
                'history_length': len(self.confidence_history)
            }
        except Exception as e:
            self.logger.error(f"Confidence report generation error: {e}")
            return {}


# ═══════════════════════════════════════════════════════════════════════
# 🔥🔥🔥 5️⃣ 다중 타임프레임 컨센서스 (Multi-Timeframe Consensus) 🔥🔥🔥
# ═══════════════════════════════════════════════════════════════════════

class MultiTimeframeConsensusEngine:
    """
    🎯 다중 타임프레임 컨센서스 엔진
    - 여러 타임프레임에서 동시에 regime 분석
    - 타임프레임 간 일치도 계산
    - 계층적 컨센서스 (장기 > 단기 가중치)
    - 타임프레임 간 모순 감지 및 해결
    - 다중 해상도 regime 맵 생성
    - Regime 진화 예측
    """

    def __init__(self, market_data_manager):
        self.market_data = market_data_manager
        self.logger = get_logger("MTFConsensus")

        # 📊 분석할 타임프레임 정의 (짧은 것부터 긴 것까지)
        self.timeframes = ['5m', '15m', '1h', '4h', '1d']

        # 🎯 타임프레임별 가중치 (장기 타임프레임이 더 중요)
        self.timeframe_weights = {
            '5m': 0.10,   # 단기 노이즈 필터링
            '15m': 0.15,  # 단기 추세
            '1h': 0.20,   # 중기 추세
            '4h': 0.25,   # 장기 추세
            '1d': 0.30    # 매크로 추세 (가장 중요)
        }

        # 📦 캐싱
        self._cache = {}
        self._cache_ttl = 180  # 3분 캐시

        # 📈 히스토리
        self.consensus_history = deque(maxlen=100)
        self.divergence_history = deque(maxlen=50)
        self.alignment_history = deque(maxlen=50)

        # 🎚️ 임계값 설정
        self.thresholds = {
            'strong_consensus': 0.80,      # 강한 컨센서스
            'moderate_consensus': 0.60,    # 중간 컨센서스
            'weak_consensus': 0.40,        # 약한 컨센서스
            'divergence_critical': 0.30,   # 심각한 분산
            'alignment_excellent': 0.85,   # 우수한 정렬
            'alignment_good': 0.70,        # 좋은 정렬
            'alignment_poor': 0.50         # 낮은 정렬
        }

        # 🔄 타임프레임 간 관계 매트릭스
        self.timeframe_relationships = self._build_timeframe_relationships()

        # 📊 타임프레임별 regime 캐시
        self.timeframe_regimes = {}

        # 🎯 계층적 구조 (부모-자식 관계)
        self.timeframe_hierarchy = {
            '1d': ['4h'],
            '4h': ['1h'],
            '1h': ['15m'],
            '15m': ['5m'],
            '5m': []
        }

    def _build_timeframe_relationships(self):
        """타임프레임 간 상관관계 매트릭스 구축"""
        try:
            # 타임프레임 간 예상 상관계수 (장기 > 단기)
            relationships = {}
            
            for i, tf1 in enumerate(self.timeframes):
                relationships[tf1] = {}
                for j, tf2 in enumerate(self.timeframes):
                    if i == j:
                        relationships[tf1][tf2] = 1.0  # 자기 자신
                    else:
                        # 거리에 따라 상관계수 감소
                        distance = abs(i - j)
                        correlation = max(0.5, 1.0 - (distance * 0.15))
                        relationships[tf1][tf2] = correlation

            return relationships

        except Exception as e:
            self.logger.error(f"Relationship matrix building error: {e}")
            return {}

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
    # 1️⃣ 모든 타임프레임 분석
    # ═══════════════════════════════════════════════════════════════════════

    def analyze_all_timeframes(self, regime_analyzer):
        """
        모든 타임프레임에서 regime 분석 수행

        Args:
            regime_analyzer: MarketRegimeAnalyzer 인스턴스

        Returns:
            dict: {timeframe: {'regime': str, 'score': float, 'confidence': float, 'indicators': dict}}
        """
        try:
            results = {}

            for timeframe in self.timeframes:
                cached = self._get_cached_data(f'timeframe_analysis_{timeframe}')
                if cached:
                    results[timeframe] = cached
                    continue

                try:
                    # 각 타임프레임에서 regime 분석
                    # (실제로는 타임프레임별 데이터를 사용해야 함)
                    # 여기서는 간소화된 버전으로 구현
                    
                    # Trend 분석
                    trend = regime_analyzer._get_macro_trend(timeframe)
                    
                    # Volatility 분석
                    volatility = regime_analyzer._get_market_volatility(timeframe)
                    
                    # Volume 분석
                    volume_profile = regime_analyzer._analyze_volume_profile(timeframe=timeframe)
                    
                    # Momentum 분석
                    momentum = regime_analyzer._calculate_market_momentum(timeframe)
                    
                    # Breadth 분석
                    market_breadth = regime_analyzer._get_market_breadth(timeframe)

                    # 간소화된 지표 딕셔너리
                    indicators = {
                        'trend': trend,
                        'volatility': volatility,
                        'volume': volume_profile,
                        'momentum': momentum,
                        'breadth': market_breadth
                    }

                    # Regime 결정 (간소화)
                    regime, score = self._determine_regime_for_timeframe(indicators)
                    
                    # 신뢰도 계산 (간소화)
                    confidence = self._calculate_timeframe_confidence(indicators, timeframe)

                    result = {
                        'regime': regime,
                        'score': score,
                        'confidence': confidence,
                        'indicators': indicators,
                        'timeframe': timeframe,
                        'timestamp': datetime.now()
                    }

                    results[timeframe] = result
                    self._set_cached_data(f'timeframe_analysis_{timeframe}', result)
                    self.timeframe_regimes[timeframe] = result

                except Exception as e:
                    self.logger.debug(f"Timeframe {timeframe} analysis error: {e}")
                    # 에러 시 중립 상태 반환
                    results[timeframe] = {
                        'regime': 'UNCERTAIN',
                        'score': 0.0,
                        'confidence': 0.5,
                        'indicators': {},
                        'timeframe': timeframe,
                        'timestamp': datetime.now()
                    }

            return results

        except Exception as e:
            self.logger.error(f"All timeframes analysis error: {e}")
            return {}

    def _determine_regime_for_timeframe(self, indicators):
        """
        특정 타임프레임의 지표를 기반으로 regime 결정

        Returns:
            tuple: (regime_name, score)
        """
        try:
            trend = indicators.get('trend', 'sideways')
            volatility = indicators.get('volatility', 'medium')
            momentum = indicators.get('momentum', 'neutral')
            breadth = indicators.get('breadth', 0.5)

            # 간소화된 regime 결정 로직
            if trend in ['uptrend', 'strong_uptrend']:
                if volatility in ['high', 'extreme']:
                    return 'BULL_VOLATILITY', 0.8
                else:
                    return 'BULL_CONSOLIDATION', 0.7

            elif trend in ['downtrend', 'strong_downtrend']:
                if volatility in ['high', 'extreme']:
                    return 'BEAR_VOLATILITY', 0.8
                else:
                    return 'BEAR_CONSOLIDATION', 0.7

            elif trend == 'sideways':
                if volatility == 'low':
                    return 'SIDEWAYS_COMPRESSION', 0.6
                else:
                    return 'SIDEWAYS_CHOP', 0.6

            return 'UNCERTAIN', 0.5

        except Exception as e:
            self.logger.debug(f"Regime determination error: {e}")
            return 'UNCERTAIN', 0.5

    def _calculate_timeframe_confidence(self, indicators, timeframe):
        """
        타임프레임별 신뢰도 계산

        Returns:
            float: 신뢰도 (0.0 ~ 1.0)
        """
        try:
            confidence = 0.7  # 기본 신뢰도

            # 지표 명확성에 따라 신뢰도 조정
            trend = indicators.get('trend', 'sideways')
            if trend in ['strong_uptrend', 'strong_downtrend']:
                confidence += 0.15
            elif trend == 'sideways':
                confidence -= 0.10

            volatility = indicators.get('volatility', 'medium')
            if volatility in ['extreme', 'low']:
                confidence += 0.10

            # 타임프레임별 가중치 (장기 타임프레임이 더 신뢰)
            tf_weight = self.timeframe_weights.get(timeframe, 0.5)
            confidence *= (0.8 + tf_weight * 0.4)  # 0.8 ~ 1.2 범위

            return np.clip(confidence, 0.0, 1.0)

        except Exception as e:
            self.logger.debug(f"Timeframe confidence calculation error: {e}")
            return 0.7

    # ═══════════════════════════════════════════════════════════════════════
    # 2️⃣ 타임프레임 간 컨센서스 계산
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_timeframe_consensus(self, timeframe_results):
        """
        여러 타임프레임의 분석 결과를 종합하여 컨센서스 계산

        Args:
            timeframe_results: {timeframe: result_dict}

        Returns:
            dict: {
                'consensus_regime': str,
                'consensus_score': float,
                'consensus_confidence': float,
                'alignment_score': float,
                'participating_timeframes': list
            }
        """
        try:
            if not timeframe_results:
                return {
                    'consensus_regime': 'UNCERTAIN',
                    'consensus_score': 0.0,
                    'consensus_confidence': 0.5,
                    'alignment_score': 0.5,
                    'participating_timeframes': []
                }

            # 각 regime의 가중 점수 계산
            regime_scores = {}
            total_weight = 0.0

            for timeframe, result in timeframe_results.items():
                regime = result['regime']
                score = result['score']
                confidence = result['confidence']
                
                # 타임프레임 가중치
                tf_weight = self.timeframe_weights.get(timeframe, 0.1)
                
                # 가중 점수 = 원점수 × 신뢰도 × 타임프레임 가중치
                weighted_score = score * confidence * tf_weight
                
                if regime not in regime_scores:
                    regime_scores[regime] = 0.0
                
                regime_scores[regime] += weighted_score
                total_weight += tf_weight

            # 정규화
            if total_weight > 0:
                regime_scores = {k: v / total_weight for k, v in regime_scores.items()}

            # 최고 점수 regime 선택
            if regime_scores:
                consensus_regime = max(regime_scores, key=regime_scores.get)
                consensus_score = regime_scores[consensus_regime]
            else:
                consensus_regime = 'UNCERTAIN'
                consensus_score = 0.0

            # 정렬도 계산
            alignment_score = self.calculate_alignment_score(timeframe_results)

            # 컨센서스 신뢰도 계산
            consensus_confidence = self._calculate_consensus_confidence(
                timeframe_results, 
                consensus_regime,
                alignment_score
            )

            result = {
                'consensus_regime': consensus_regime,
                'consensus_score': consensus_score,
                'consensus_confidence': consensus_confidence,
                'alignment_score': alignment_score,
                'participating_timeframes': list(timeframe_results.keys()),
                'regime_scores': regime_scores,
                'timestamp': datetime.now()
            }

            # 히스토리 저장
            self.consensus_history.append(result)

            return result

        except Exception as e:
            self.logger.error(f"Consensus calculation error: {e}")
            return {
                'consensus_regime': 'UNCERTAIN',
                'consensus_score': 0.0,
                'consensus_confidence': 0.5,
                'alignment_score': 0.5,
                'participating_timeframes': []
            }

    def _calculate_consensus_confidence(self, timeframe_results, consensus_regime, alignment_score):
        """
        컨센서스 신뢰도 계산

        Returns:
            float: 신뢰도 (0.0 ~ 1.0)
        """
        try:
            # 1️⃣ 일치하는 타임프레임 비율
            agreeing_count = sum(1 for result in timeframe_results.values() 
                                if result['regime'] == consensus_regime)
            agreement_ratio = agreeing_count / len(timeframe_results) if timeframe_results else 0.5

            # 2️⃣ 평균 신뢰도
            avg_confidence = np.mean([result['confidence'] for result in timeframe_results.values()]) if timeframe_results else 0.5

            # 3️⃣ 정렬도
            alignment_factor = alignment_score

            # 4️⃣ 점수 분산 (낮을수록 좋음)
            scores = [result['score'] for result in timeframe_results.values()]
            score_std = np.std(scores) if len(scores) > 1 else 0.0
            score_consistency = 1.0 - min(score_std, 1.0)

            # 종합 신뢰도
            consensus_confidence = (
                agreement_ratio * 0.35 +
                avg_confidence * 0.25 +
                alignment_factor * 0.25 +
                score_consistency * 0.15
            )

            return np.clip(consensus_confidence, 0.0, 1.0)

        except Exception as e:
            self.logger.debug(f"Consensus confidence calculation error: {e}")
            return 0.5

    # ═══════════════════════════════════════════════════════════════════════
    # 3️⃣ 타임프레임 간 정렬도 계산
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_alignment_score(self, timeframe_results):
        """
        타임프레임 간 정렬도 점수 계산
        모든 타임프레임이 같은 방향을 가리키는 정도

        Returns:
            float: 정렬도 (0.0 ~ 1.0)
        """
        try:
            if len(timeframe_results) < 2:
                return 1.0

            # regime을 방향성으로 변환
            regime_directions = {}
            for timeframe, result in timeframe_results.items():
                regime = result['regime']
                
                if 'BULL' in regime or regime == 'ACCUMULATION':
                    direction = 1  # 상승
                elif 'BEAR' in regime or regime == 'DISTRIBUTION':
                    direction = -1  # 하락
                else:
                    direction = 0  # 중립
                
                regime_directions[timeframe] = direction

            # 방향 일치도 계산
            directions = list(regime_directions.values())
            
            # 모두 같은 방향이면 1.0
            if len(set(directions)) == 1:
                alignment = 1.0
            else:
                # 다수 방향 찾기
                most_common_direction = max(set(directions), key=directions.count)
                alignment_count = directions.count(most_common_direction)
                alignment = alignment_count / len(directions)

            # 히스토리 저장
            self.alignment_history.append({
                'timestamp': datetime.now(),
                'alignment_score': alignment,
                'directions': regime_directions
            })

            return alignment

        except Exception as e:
            self.logger.debug(f"Alignment score calculation error: {e}")
            return 0.5

    # ═══════════════════════════════════════════════════════════════════════
    # 4️⃣ 타임프레임 간 분산 감지
    # ═══════════════════════════════════════════════════════════════════════

    def detect_timeframe_divergence(self, timeframe_results):
        """
        타임프레임 간 분산(Divergence) 감지

        Returns:
            dict: {
                'divergence_detected': bool,
                'divergence_score': float (0.0 ~ 1.0),
                'diverging_pairs': list,
                'dominant_timeframe': str
            }
        """
        try:
            diverging_pairs = []
            divergence_scores = []

            # 모든 타임프레임 쌍 비교
            timeframes_list = list(timeframe_results.keys())
            
            for i, tf1 in enumerate(timeframes_list):
                for tf2 in timeframes_list[i+1:]:
                    result1 = timeframe_results[tf1]
                    result2 = timeframe_results[tf2]

                    # Regime 비교
                    regime1 = result1['regime']
                    regime2 = result2['regime']

                    # 방향성 비교
                    dir1 = 1 if 'BULL' in regime1 or regime1 == 'ACCUMULATION' else (-1 if 'BEAR' in regime1 or regime1 == 'DISTRIBUTION' else 0)
                    dir2 = 1 if 'BULL' in regime2 or regime2 == 'ACCUMULATION' else (-1 if 'BEAR' in regime2 or regime2 == 'DISTRIBUTION' else 0)

                    # 분산 점수 계산
                    if dir1 * dir2 < 0:  # 반대 방향
                        divergence = 1.0
                        diverging_pairs.append((tf1, tf2, regime1, regime2))
                    elif dir1 == 0 or dir2 == 0:  # 한쪽이 중립
                        divergence = 0.5
                    else:  # 같은 방향
                        divergence = 0.0

                    divergence_scores.append(divergence)

            # 전체 분산 점수
            avg_divergence = np.mean(divergence_scores) if divergence_scores else 0.0

            # 분산 감지 여부
            divergence_detected = avg_divergence > self.thresholds['divergence_critical']

            # 지배적 타임프레임 찾기 (가장 높은 가중치의 타임프레임)
            dominant_timeframe = max(
                timeframe_results.keys(),
                key=lambda tf: self.timeframe_weights.get(tf, 0)
            )

            result = {
                'divergence_detected': divergence_detected,
                'divergence_score': avg_divergence,
                'diverging_pairs': diverging_pairs,
                'dominant_timeframe': dominant_timeframe,
                'timestamp': datetime.now()
            }

            # 히스토리 저장
            self.divergence_history.append(result)

            return result

        except Exception as e:
            self.logger.error(f"Divergence detection error: {e}")
            return {
                'divergence_detected': False,
                'divergence_score': 0.0,
                'diverging_pairs': [],
                'dominant_timeframe': '1d'
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 5️⃣ 충돌 해결
    # ═══════════════════════════════════════════════════════════════════════

    def resolve_conflicts(self, timeframe_results, divergence_info):
        """
        타임프레임 간 충돌 해결
        장기 타임프레임 우선, 신뢰도 고려

        Returns:
            dict: {
                'resolved_regime': str,
                'resolution_method': str,
                'confidence': float
            }
        """
        try:
            # 1️⃣ 분산이 없으면 컨센서스 그대로 사용
            if not divergence_info['divergence_detected']:
                consensus = self.calculate_timeframe_consensus(timeframe_results)
                return {
                    'resolved_regime': consensus['consensus_regime'],
                    'resolution_method': 'consensus',
                    'confidence': consensus['consensus_confidence']
                }

            # 2️⃣ 분산이 있으면 계층적 해결
            # 지배적 타임프레임(가장 긴 타임프레임) 우선
            dominant_tf = divergence_info['dominant_timeframe']
            dominant_result = timeframe_results[dominant_tf]

            # 3️⃣ 지배적 타임프레임의 신뢰도가 높으면 그대로 채택
            if dominant_result['confidence'] > 0.7:
                return {
                    'resolved_regime': dominant_result['regime'],
                    'resolution_method': 'dominant_timeframe',
                    'confidence': dominant_result['confidence'],
                    'dominant_timeframe': dominant_tf
                }

            # 4️⃣ 신뢰도가 낮으면 가중 평균
            weighted_regimes = {}
            total_weight = 0.0

            for timeframe, result in timeframe_results.items():
                regime = result['regime']
                confidence = result['confidence']
                tf_weight = self.timeframe_weights.get(timeframe, 0.1)
                
                weight = confidence * tf_weight
                
                if regime not in weighted_regimes:
                    weighted_regimes[regime] = 0.0
                
                weighted_regimes[regime] += weight
                total_weight += weight

            # 정규화
            if total_weight > 0:
                weighted_regimes = {k: v / total_weight for k, v in weighted_regimes.items()}

            # 최고 가중치 regime 선택
            resolved_regime = max(weighted_regimes, key=weighted_regimes.get)
            resolved_confidence = weighted_regimes[resolved_regime]

            return {
                'resolved_regime': resolved_regime,
                'resolution_method': 'weighted_average',
                'confidence': resolved_confidence
            }

        except Exception as e:
            self.logger.error(f"Conflict resolution error: {e}")
            return {
                'resolved_regime': 'UNCERTAIN',
                'resolution_method': 'error_fallback',
                'confidence': 0.5
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 6️⃣ 다중 해상도 Regime 맵 생성
    # ═══════════════════════════════════════════════════════════════════════

    def generate_multi_resolution_map(self, timeframe_results):
        """
        다중 해상도 regime 맵 생성
        각 타임프레임의 regime과 그 관계를 시각적으로 표현

        Returns:
            dict: 타임프레임별 regime 맵과 계층 구조
        """
        try:
            resolution_map = {
                'timeframe_regimes': {},
                'hierarchical_structure': {},
                'transition_points': [],
                'key_insights': []
            }

            # 1️⃣ 각 타임프레임의 regime 기록
            for timeframe, result in timeframe_results.items():
                resolution_map['timeframe_regimes'][timeframe] = {
                    'regime': result['regime'],
                    'score': result['score'],
                    'confidence': result['confidence'],
                    'indicators': result['indicators']
                }

            # 2️⃣ 계층 구조 분석
            for parent_tf, children_tf in self.timeframe_hierarchy.items():
                if parent_tf in timeframe_results:
                    parent_regime = timeframe_results[parent_tf]['regime']
                    
                    children_regimes = []
                    for child_tf in children_tf:
                        if child_tf in timeframe_results:
                            child_regime = timeframe_results[child_tf]['regime']
                            children_regimes.append({
                                'timeframe': child_tf,
                                'regime': child_regime,
                                'matches_parent': child_regime == parent_regime
                            })
                    
                    resolution_map['hierarchical_structure'][parent_tf] = {
                        'parent_regime': parent_regime,
                        'children': children_regimes
                    }

            # 3️⃣ 전환점 감지
            # 타임프레임 간 regime이 바뀌는 지점 찾기
            for i, timeframe in enumerate(self.timeframes[:-1]):
                current_tf = timeframe
                next_tf = self.timeframes[i + 1]
                
                if current_tf in timeframe_results and next_tf in timeframe_results:
                    current_regime = timeframe_results[current_tf]['regime']
                    next_regime = timeframe_results[next_tf]['regime']
                    
                    if current_regime != next_regime:
                        resolution_map['transition_points'].append({
                            'from_timeframe': current_tf,
                            'to_timeframe': next_tf,
                            'from_regime': current_regime,
                            'to_regime': next_regime
                        })

            # 4️⃣ 주요 인사이트 생성
            resolution_map['key_insights'] = self._generate_resolution_insights(
                timeframe_results,
                resolution_map
            )

            return resolution_map

        except Exception as e:
            self.logger.error(f"Multi-resolution map generation error: {e}")
            return {}

    def _generate_resolution_insights(self, timeframe_results, resolution_map):
        """해상도 맵 기반 인사이트 생성"""
        insights = []

        try:
            # 1️⃣ 전체 방향성 확인
            regimes = [result['regime'] for result in timeframe_results.values()]
            unique_regimes = set(regimes)
            
            if len(unique_regimes) == 1:
                insights.append(f"✅ 모든 타임프레임이 {regimes[0]} 상태로 일치")
            elif len(unique_regimes) > 3:
                insights.append("⚠️ 타임프레임 간 높은 분산 - 불확실한 시장")

            # 2️⃣ 장기 vs 단기 비교
            if '1d' in timeframe_results and '5m' in timeframe_results:
                long_term = timeframe_results['1d']['regime']
                short_term = timeframe_results['5m']['regime']
                
                if long_term != short_term:
                    insights.append(f"🔄 장단기 불일치: 장기({long_term}) vs 단기({short_term})")

            # 3️⃣ 전환점 분석
            if len(resolution_map['transition_points']) > 0:
                insights.append(f"📍 {len(resolution_map['transition_points'])}개의 regime 전환점 감지")

        except Exception as e:
            self.logger.debug(f"Resolution insights generation error: {e}")

        return insights

    # ═══════════════════════════════════════════════════════════════════════
    # 7️⃣ Regime 진화 예측
    # ═══════════════════════════════════════════════════════════════════════

    def predict_regime_evolution(self, timeframe_results):
        """
        타임프레임 간 패턴을 기반으로 regime 진화 예측

        Returns:
            dict: {
                'predicted_regime': str,
                'prediction_confidence': float,
                'evolution_direction': str,
                'time_horizon': str
            }
        """
        try:
            # 1️⃣ 단기 → 장기 추세 전파 분석
            # 단기 타임프레임의 변화가 장기로 확산되는지 확인
            
            short_term_regimes = []
            long_term_regimes = []

            for timeframe in ['5m', '15m']:
                if timeframe in timeframe_results:
                    short_term_regimes.append(timeframe_results[timeframe]['regime'])

            for timeframe in ['4h', '1d']:
                if timeframe in timeframe_results:
                    long_term_regimes.append(timeframe_results[timeframe]['regime'])

            # 2️⃣ 히스토리 기반 패턴 매칭
            if len(self.consensus_history) >= 3:
                recent_consensus = [c['consensus_regime'] for c in list(self.consensus_history)[-3:]]
                
                # 추세 파악
                if len(set(recent_consensus)) == 1:
                    evolution_direction = 'stable'
                elif recent_consensus[-1] != recent_consensus[0]:
                    evolution_direction = 'transitioning'
                else:
                    evolution_direction = 'oscillating'
            else:
                evolution_direction = 'unknown'

            # 3️⃣ 예측 로직
            predicted_regime = 'UNCERTAIN'
            prediction_confidence = 0.5

            # 단기와 장기가 일치하면 유지 예측
            if short_term_regimes and long_term_regimes:
                if short_term_regimes[0] == long_term_regimes[0]:
                    predicted_regime = short_term_regimes[0]
                    prediction_confidence = 0.75
                else:
                    # 장기 추세를 따를 가능성 높음
                    predicted_regime = long_term_regimes[0]
                    prediction_confidence = 0.60

            # 4️⃣ 시간 범위 추정
            if prediction_confidence > 0.7:
                time_horizon = 'short_term'  # 1-4시간
            elif prediction_confidence > 0.5:
                time_horizon = 'medium_term'  # 4-24시간
            else:
                time_horizon = 'long_term'  # 24시간 이상

            return {
                'predicted_regime': predicted_regime,
                'prediction_confidence': prediction_confidence,
                'evolution_direction': evolution_direction,
                'time_horizon': time_horizon,
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Regime evolution prediction error: {e}")
            return {
                'predicted_regime': 'UNCERTAIN',
                'prediction_confidence': 0.5,
                'evolution_direction': 'unknown',
                'time_horizon': 'unknown'
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 8️⃣ 종합 컨센서스 리포트
    # ═══════════════════════════════════════════════════════════════════════

    def get_consensus_report(self):
        """
        다중 타임프레임 컨센서스 종합 리포트

        Returns:
            dict: 전체 컨센서스 분석 결과
        """
        try:
            if len(self.consensus_history) == 0:
                return {}

            latest_consensus = self.consensus_history[-1]

            report = {
                'timestamp': datetime.now().isoformat(),
                'current_consensus': latest_consensus,
                'alignment_metrics': {
                    'current_alignment': self.alignment_history[-1] if len(self.alignment_history) > 0 else None,
                    'average_alignment': np.mean([a['alignment_score'] for a in self.alignment_history]) if len(self.alignment_history) > 0 else 0.5
                },
                'divergence_metrics': {
                    'current_divergence': self.divergence_history[-1] if len(self.divergence_history) > 0 else None,
                    'divergence_frequency': sum(1 for d in self.divergence_history if d['divergence_detected']) / max(len(self.divergence_history), 1)
                },
                'timeframe_regimes': self.timeframe_regimes,
                'consensus_trend': self._analyze_consensus_trend()
            }

            return report

        except Exception as e:
            self.logger.error(f"Consensus report generation error: {e}")
            return {}

    def _analyze_consensus_trend(self):
        """컨센서스 추세 분석"""
        try:
            if len(self.consensus_history) < 3:
                return 'insufficient_data'

            recent = [c['consensus_regime'] for c in list(self.consensus_history)[-5:]]
            
            if len(set(recent)) == 1:
                return 'stable'
            elif recent[-1] != recent[0]:
                return 'changing'
            else:
                return 'volatile'

        except Exception as e:
            self.logger.debug(f"Consensus trend analysis error: {e}")
            return 'unknown'


# ═══════════════════════════════════════════════════════════════════════
# MarketRegimeAnalyzer 클래스 (전체 시스템 통합 - 유동성 포함)
# ═══════════════════════════════════════════════════════════════════════

class MarketRegimeAnalyzer:
    """
    여러 요인을 종합하여 현재 시장 체제(Market Regime)를 분석합니다.
    고도화 버전 - 다중 지표 및 심층 분석 + 실시간 적응형 가중치 시스템 + 상태 지속성 추적 + 온체인/매크로 데이터 융합 + 다차원 신뢰도 스코어링 + 🔥🔥 다중 타임프레임 컨센서스 + 🔥🔥 유동성 상태 추정 🔥🔥
    """

    def __init__(self, market_data_manager):
        self.market_data = market_data_manager
        self.logger = get_logger("MarketRegime")

        # 캐싱 메커니즘 추가 (성능 향상)
        self._cache = {}
        self._cache_ttl = 60  # 60초 캐시

        # 분석에 사용할 주요 알트코인 리스트 확장
        self.major_alts = ['ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']

        # 🔥🔥🔥 온체인/매크로 데이터 매니저 초기화 🔥🔥🔥
        self.onchain_manager = OnChainDataManager()
        self.macro_manager = MacroDataManager(market_data_manager)

        # 🔥🔥🔥 4️⃣ 다차원 신뢰도 스코어러 초기화 🔥🔥🔥
        self.confidence_scorer = MultiDimensionalConfidenceScorer()

        # 🔥🔥🔥 5️⃣ 다중 타임프레임 컨센서스 엔진 초기화 🔥🔥🔥
        self.mtf_consensus = MultiTimeframeConsensusEngine(market_data_manager)

        # 🔥🔥🔥 6️⃣ 유동성 상태 탐지기 초기화 🔥🔥🔥
        self.liquidity_detector = LiquidityRegimeDetector(market_data_manager)

        # 🎯 기본 시장 체제별 가중치 설정 (Baseline)
        self.base_regime_weights = {
            'trend': 0.20,  # 추세 (유동성으로 인해 비중 조정)
            'volatility': 0.18,  # 변동성
            'volume': 0.12,  # 거래량
            'momentum': 0.10,  # 모멘텀
            'sentiment': 0.06,  # 시장 심리
            'onchain': 0.10,  # 🔥 온체인 신호
            'macro': 0.07,  # 🔥 매크로 신호
            'liquidity': 0.17  # 🔥🔥 유동성 (새로 추가, 중요도 높음)
        }

        # 🔥 실시간 적응형 가중치 시스템
        self.adaptive_weights = self.base_regime_weights.copy()

        # 🎯 시장 상태별 최적 가중치 프로파일 (유동성 포함)
        self.weight_profiles = {
            'high_volatility': {
                'trend': 0.12,
                'volatility': 0.30,
                'volume': 0.10,
                'momentum': 0.15,
                'sentiment': 0.04,
                'onchain': 0.08,
                'macro': 0.05,
                'liquidity': 0.16
            },
            'strong_trend': {
                'trend': 0.30,
                'volatility': 0.10,
                'volume': 0.10,
                'momentum': 0.15,
                'sentiment': 0.04,
                'onchain': 0.08,
                'macro': 0.06,
                'liquidity': 0.17
            },
            'sideways_market': {
                'trend': 0.08,
                'volatility': 0.15,
                'volume': 0.20,
                'momentum': 0.10,
                'sentiment': 0.08,
                'onchain': 0.12,
                'macro': 0.07,
                'liquidity': 0.20
            },
            'volume_spike': {
                'trend': 0.15,
                'volatility': 0.12,
                'volume': 0.25,
                'momentum': 0.10,
                'sentiment': 0.04,
                'onchain': 0.08,
                'macro': 0.06,
                'liquidity': 0.20
            },
            'momentum_driven': {
                'trend': 0.20,
                'volatility': 0.10,
                'volume': 0.10,
                'momentum': 0.25,
                'sentiment': 0.04,
                'onchain': 0.06,
                'macro': 0.06,
                'liquidity': 0.19
            },
            'sentiment_extreme': {
                'trend': 0.15,
                'volatility': 0.12,
                'volume': 0.10,
                'momentum': 0.12,
                'sentiment': 0.12,
                'onchain': 0.12,
                'macro': 0.06,
                'liquidity': 0.21
            },
            'onchain_dominant': {
                'trend': 0.12,
                'volatility': 0.08,
                'volume': 0.08,
                'momentum': 0.08,
                'sentiment': 0.04,
                'onchain': 0.30,
                'macro': 0.12,
                'liquidity': 0.18
            },
            'macro_dominant': {
                'trend': 0.12,
                'volatility': 0.08,
                'volume': 0.08,
                'momentum': 0.08,
                'sentiment': 0.04,
                'onchain': 0.12,
                'macro': 0.30,
                'liquidity': 0.18
            },
            'low_liquidity': {  # 🔥🔥 새로운 프로파일
                'trend': 0.10,
                'volatility': 0.18,
                'volume': 0.08,
                'momentum': 0.08,
                'sentiment': 0.06,
                'onchain': 0.08,
                'macro': 0.06,
                'liquidity': 0.36  # 유동성이 지배적
            }
        }

        # 📊 지표 신뢰도 추적 (Historical Performance)
        self.indicator_reliability = {
            'trend': deque(maxlen=50),
            'volatility': deque(maxlen=50),
            'volume': deque(maxlen=50),
            'momentum': deque(maxlen=50),
            'sentiment': deque(maxlen=50),
            'onchain': deque(maxlen=50),
            'macro': deque(maxlen=50),
            'liquidity': deque(maxlen=50)  # 🔥🔥 유동성 추가
        }

        # 📈 예측 성능 추적
        self.prediction_history = deque(maxlen=100)

        # ⚡ 가중치 조정 속도 (0.0 ~ 1.0, 높을수록 빠르게 적응)
        self.adaptation_speed = 0.3

        # 🎚️ 지표별 신뢰도 임계값
        self.reliability_thresholds = {
            'high': 0.75,
            'medium': 0.50,
            'low': 0.30
        }

        # ═══════════════════════════════════════════════════════════════════
        # 🔥🔥🔥 2️⃣ 상태 지속성 추적 (Regime Transition Stability) 🔥🔥🔥
        # ═══════════════════════════════════════════════════════════════════

        # 📌 현재 regime 상태 추적
        self.current_regime = None
        self.current_regime_start_time = None
        self.current_regime_duration = timedelta(0)

        # 📊 Regime 전환 히스토리 (최근 100개)
        self.regime_history = deque(maxlen=100)

        # 🎯 Regime 전환 파라미터
        self.transition_config = {
            'min_duration_seconds': 300,  # 최소 regime 지속 시간 (5분)
            'base_threshold': 0.15,  # 기본 전환 임계값 (15% 차이 필요)
            'max_threshold': 0.40,  # 최대 전환 임계값
            'stability_window': 20,  # 안정성 평가 윈도우 (최근 N개)
            'flip_penalty_duration': 600,  # 빈번한 전환 페널티 기간 (10분)
            'confidence_boost_stable': 1.2,  # 안정적인 regime 신뢰도 부스트
            'confidence_penalty_unstable': 0.8  # 불안정 regime 신뢰도 페널티
        }

        # 📈 Regime 강도 추적
        self.regime_strength_history = deque(maxlen=50)
        self.current_regime_strength = 0.0

        # 🔄 전환 임계값 동적 조정
        self.dynamic_threshold = self.transition_config['base_threshold']

        # ⚠️ 전환 카운터 (빈번한 전환 감지용)
        self.recent_transitions = deque(maxlen=10)
        self.flip_count = 0
        self.last_flip_time = None

        # 🎚️ Stability Score
        self.stability_score = 1.0  # 1.0 = 매우 안정적, 0.0 = 매우 불안정

        # 📊 Regime별 통계 추적
        self.regime_statistics = {}
        self._initialize_regime_statistics()

        # 🔮 전환 예측 메커니즘
        self.transition_warning = None
        self.transition_probability = 0.0

        # 🔥🔥🔥 온체인/매크로 데이터 통합 설정 🔥🔥🔥
        self.onchain_macro_config = {
            'onchain_weight': 0.40,  # 온체인 신호의 전체 가중치
            'macro_weight': 0.30,  # 매크로 신호의 전체 가중치
            'traditional_weight': 0.30,  # 기존 지표의 전체 가중치
            'signal_merge_method': 'weighted_average',  # weighted_average, max, consensus
            'contradiction_threshold': 0.5,  # 신호 모순 감지 임계값
            'min_confidence_threshold': 0.3  # 최소 신뢰도
        }

        # 📊 온체인/매크로 신호 히스토리
        self.onchain_signal_history = deque(maxlen=50)
        self.macro_signal_history = deque(maxlen=50)
        self.signal_correlation_history = deque(maxlen=50)

    def _initialize_regime_statistics(self):
        """Regime별 통계 초기화"""
        regime_types = [
            'BULL_VOLATILITY', 'BULL_CONSOLIDATION', 'BULL_EXHAUSTION',
            'BEAR_VOLATILITY', 'BEAR_CONSOLIDATION', 'BEAR_CAPITULATION',
            'SIDEWAYS_COMPRESSION', 'SIDEWAYS_CHOP',
            'ACCUMULATION', 'DISTRIBUTION', 'UNCERTAIN'
        ]

        for regime in regime_types:
            self.regime_statistics[regime] = {
                'total_count': 0,
                'total_duration': timedelta(0),
                'avg_duration': timedelta(0),
                'avg_strength': 0.0,
                'transition_from': {},  # 어떤 regime에서 전환되어 왔는지
                'transition_to': {},  # 어떤 regime으로 전환되었는지
                'success_rate': 0.0  # 예측 성공률
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 🔥🔥🔥 온체인/매크로 데이터 융합 핵심 메서드 🔥🔥🔥
    # ═══════════════════════════════════════════════════════════════════════

    def _get_onchain_macro_signals(self):
        """
        온체인 및 매크로 신호 수집
        Returns: {'onchain': dict, 'macro': dict, 'merged': dict}
        """
        try:
            # 온체인 신호 수집
            onchain_signal = self.onchain_manager.get_comprehensive_onchain_signal()

            # 매크로 신호 수집
            macro_signal = self.macro_manager.get_comprehensive_macro_signal()

            # 히스토리 저장
            self.onchain_signal_history.append({
                'timestamp': datetime.now(),
                'signal': onchain_signal['signal'],
                'score': onchain_signal['score']
            })

            self.macro_signal_history.append({
                'timestamp': datetime.now(),
                'signal': macro_signal['signal'],
                'score': macro_signal['score']
            })

            # 신호 병합
            merged_signal = self._merge_onchain_macro_signals(onchain_signal, macro_signal)

            return {
                'onchain': onchain_signal,
                'macro': macro_signal,
                'merged': merged_signal
            }

        except Exception as e:
            self.logger.error(f"Onchain/Macro signal collection error: {e}")
            return {
                'onchain': {'score': 0.0, 'signal': 'NEUTRAL', 'details': {}, 'component_scores': {}},
                'macro': {'score': 0.0, 'signal': 'NEUTRAL', 'details': {}, 'component_scores': {}},
                'merged': {'score': 0.0, 'signal': 'NEUTRAL', 'confidence': 0.5}
            }

    def _merge_onchain_macro_signals(self, onchain_signal, macro_signal):
        """
        온체인 및 매크로 신호를 병합하여 통합 신호 생성
        """
        try:
            method = self.onchain_macro_config['signal_merge_method']
            onchain_weight = self.onchain_macro_config['onchain_weight']
            macro_weight = self.onchain_macro_config['macro_weight']

            # 가중 평균 방식
            if method == 'weighted_average':
                merged_score = (
                                       onchain_signal['score'] * onchain_weight +
                                       macro_signal['score'] * macro_weight
                               ) / (onchain_weight + macro_weight)

            # 최대값 방식 (더 강한 신호 채택)
            elif method == 'max':
                if abs(onchain_signal['score']) > abs(macro_signal['score']):
                    merged_score = onchain_signal['score']
                else:
                    merged_score = macro_signal['score']

            # 합의 방식 (두 신호가 일치할 때만 강한 신호)
            elif method == 'consensus':
                if (onchain_signal['score'] > 0 and macro_signal['score'] > 0) or \
                        (onchain_signal['score'] < 0 and macro_signal['score'] < 0):
                    # 같은 방향이면 평균
                    merged_score = (onchain_signal['score'] + macro_signal['score']) / 2
                else:
                    # 다른 방향이면 약화
                    merged_score = (onchain_signal['score'] + macro_signal['score']) / 4

            else:
                merged_score = (onchain_signal['score'] + macro_signal['score']) / 2

            # 신호 모순 감지
            contradiction_level = self._detect_signal_contradiction(
                onchain_signal, macro_signal
            )

            # 신뢰도 계산
            confidence = self._calculate_merged_confidence(
                onchain_signal, macro_signal, contradiction_level
            )

            # 최종 신호 생성
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

            return {
                'score': merged_score,
                'signal': signal,
                'confidence': confidence,
                'contradiction_level': contradiction_level,
                'onchain_contribution': onchain_signal['score'] * onchain_weight,
                'macro_contribution': macro_signal['score'] * macro_weight
            }

        except Exception as e:
            self.logger.error(f"Signal merge error: {e}")
            return {
                'score': 0.0,
                'signal': 'NEUTRAL',
                'confidence': 0.5,
                'contradiction_level': 0.0,
                'onchain_contribution': 0.0,
                'macro_contribution': 0.0
            }

    def _detect_signal_contradiction(self, onchain_signal, macro_signal):
        """
        온체인과 매크로 신호 간 모순 감지
        Returns: float (0.0 ~ 1.0, 높을수록 모순 심각)
        """
        try:
            onchain_score = onchain_signal['score']
            macro_score = macro_signal['score']

            # 방향 차이 계산
            if (onchain_score > 0 and macro_score < 0) or \
                    (onchain_score < 0 and macro_score > 0):
                # 반대 방향
                contradiction = min(abs(onchain_score - macro_score), 2.0) / 2.0
            else:
                # 같은 방향이지만 강도 차이
                contradiction = abs(abs(onchain_score) - abs(macro_score)) / 2.0

            return np.clip(contradiction, 0.0, 1.0)

        except Exception as e:
            self.logger.debug(f"Contradiction detection error: {e}")
            return 0.0

    def _calculate_merged_confidence(self, onchain_signal, macro_signal, contradiction_level):
        """
        병합된 신호의 신뢰도 계산
        """
        try:
            # 기본 신뢰도: 두 신호의 강도 평균
            base_confidence = (abs(onchain_signal['score']) + abs(macro_signal['score'])) / 2

            # 모순 페널티
            contradiction_penalty = contradiction_level * 0.3

            # 일치 보너스 (두 신호가 같은 방향이면)
            if (onchain_signal['score'] > 0 and macro_signal['score'] > 0) or \
                    (onchain_signal['score'] < 0 and macro_signal['score'] < 0):
                alignment_bonus = 0.2
            else:
                alignment_bonus = 0.0

            # 최종 신뢰도
            confidence = base_confidence + alignment_bonus - contradiction_penalty

            # 히스토리 기반 조정
            if len(self.onchain_signal_history) >= 5 and len(self.macro_signal_history) >= 5:
                recent_onchain = [s['score'] for s in list(self.onchain_signal_history)[-5:]]
                recent_macro = [s['score'] for s in list(self.macro_signal_history)[-5:]]

                # 일관성 체크
                onchain_std = np.std(recent_onchain)
                macro_std = np.std(recent_macro)

                consistency_bonus = (2.0 - onchain_std - macro_std) / 20  # 최대 0.1
                confidence += consistency_bonus

            return np.clip(confidence, 0.0, 1.0)

        except Exception as e:
            self.logger.debug(f"Merged confidence calculation error: {e}")
            return 0.5

    def _integrate_onchain_macro_to_regime(self, base_regime_score, merged_signal):
        """
        온체인/매크로 신호를 기존 regime 점수에 통합
        """
        try:
            traditional_weight = self.onchain_macro_config['traditional_weight']
            onchain_macro_weight = 1.0 - traditional_weight

            # 병합 신호의 기여도
            signal_contribution = merged_signal['score'] * merged_signal['confidence']

            # 통합 점수 계산
            integrated_score = (
                    base_regime_score * traditional_weight +
                    signal_contribution * onchain_macro_weight
            )

            # 신호 강도에 따른 조정
            if abs(merged_signal['score']) > 0.7:
                # 강한 온체인/매크로 신호는 더 큰 영향
                integrated_score *= 1.1

            # 모순이 심하면 신뢰도 감소
            if merged_signal['contradiction_level'] > self.onchain_macro_config['contradiction_threshold']:
                integrated_score *= 0.9

            return np.clip(integrated_score, -1.0, 1.0)

        except Exception as e:
            self.logger.error(f"Regime integration error: {e}")
            return base_regime_score

    def _adjust_regime_based_on_onchain_macro(self, base_regime, merged_signal):
        """
        온체인/매크로 신호를 기반으로 regime 조정
        특정 조건에서는 regime을 override할 수 있음
        """
        try:
            # 매우 강한 신호는 regime override 가능
            if abs(merged_signal['score']) > 0.8 and merged_signal['confidence'] > 0.7:

                if merged_signal['signal'] == 'STRONG_BULLISH':
                    # 온체인/매크로가 강한 상승을 나타내면
                    if 'BEAR' in base_regime or base_regime == 'UNCERTAIN':
                        self.logger.info(
                            f"🔥 온체인/매크로 신호로 Regime Override: "
                            f"{base_regime} -> ACCUMULATION (신호: {merged_signal['signal']})"
                        )
                        return 'ACCUMULATION'

                elif merged_signal['signal'] == 'STRONG_BEARISH':
                    # 온체인/매크로가 강한 하락을 나타내면
                    if 'BULL' in base_regime or base_regime == 'UNCERTAIN':
                        self.logger.info(
                            f"🔥 온체인/매크로 신호로 Regime Override: "
                            f"{base_regime} -> DISTRIBUTION (신호: {merged_signal['signal']})"
                        )
                        return 'DISTRIBUTION'

            return base_regime

        except Exception as e:
            self.logger.error(f"Regime adjustment error: {e}")
            return base_regime

    def get_onchain_macro_report(self):
        """
        온체인/매크로 데이터 상세 리포트
        """
        try:
            signals = self._get_onchain_macro_signals()

            report = {
                'timestamp': datetime.now().isoformat(),
                'onchain_signal': {
                    'overall': signals['onchain']['signal'],
                    'score': signals['onchain']['score'],
                    'details': signals['onchain']['details'],
                    'component_scores': signals['onchain']['component_scores']
                },
                'macro_signal': {
                    'overall': signals['macro']['signal'],
                    'score': signals['macro']['score'],
                    'details': signals['macro']['details'],
                    'component_scores': signals['macro']['component_scores']
                },
                'merged_signal': signals['merged'],
                'historical_correlation': self._calculate_signal_correlation(),
                'signal_quality_metrics': self._calculate_signal_quality_metrics()
            }

            return report

        except Exception as e:
            self.logger.error(f"Onchain/Macro report generation error: {e}")
            return {}

    def _calculate_signal_correlation(self):
        """온체인과 매크로 신호 간 상관관계 계산"""
        try:
            if len(self.onchain_signal_history) < 10 or len(self.macro_signal_history) < 10:
                return 0.0

            onchain_scores = [s['score'] for s in list(self.onchain_signal_history)[-20:]]
            macro_scores = [s['score'] for s in list(self.macro_signal_history)[-20:]]

            correlation = np.corrcoef(onchain_scores, macro_scores)[0, 1]

            # 상관관계 히스토리 저장
            self.signal_correlation_history.append({
                'timestamp': datetime.now(),
                'correlation': correlation
            })

            return correlation

        except Exception as e:
            self.logger.debug(f"Signal correlation calculation error: {e}")
            return 0.0

    def _calculate_signal_quality_metrics(self):
        """신호 품질 메트릭 계산"""
        try:
            metrics = {
                'onchain_consistency': 0.0,
                'macro_consistency': 0.0,
                'overall_reliability': 0.0,
                'prediction_accuracy': 0.0
            }

            # 온체인 일관성
            if len(self.onchain_signal_history) >= 10:
                recent_signals = [s['signal'] for s in list(self.onchain_signal_history)[-10:]]
                unique_signals = len(set(recent_signals))
                metrics['onchain_consistency'] = 1.0 - (unique_signals / 10)

            # 매크로 일관성
            if len(self.macro_signal_history) >= 10:
                recent_signals = [s['signal'] for s in list(self.macro_signal_history)[-10:]]
                unique_signals = len(set(recent_signals))
                metrics['macro_consistency'] = 1.0 - (unique_signals / 10)

            # 전체 신뢰도
            metrics['overall_reliability'] = (
                    metrics['onchain_consistency'] * 0.5 +
                    metrics['macro_consistency'] * 0.5
            )

            return metrics

        except Exception as e:
            self.logger.debug(f"Signal quality metrics calculation error: {e}")
            return {
                'onchain_consistency': 0.0,
                'macro_consistency': 0.0,
                'overall_reliability': 0.0,
                'prediction_accuracy': 0.0
            }

    # (이하 기존 메서드들 생략 - 너무 길어서 주요 부분만 포함)
    # ... (나머지 메서드들은 market_regime_analyzer5.py와 동일)

    def get_comprehensive_analysis_report(self):
        """
        종합 분석 리포트 (모든 시스템의 정보 통합 + 🔥🔥 유동성 포함 🔥🔥)
        """
        try:
            # 다중 타임프레임 컨센서스 리포트
            mtf_report = self.mtf_consensus.get_consensus_report()
            
            # 🔥🔥 유동성 종합 리포트
            liquidity_report = self.liquidity_detector.get_comprehensive_liquidity_report()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'regime_analysis': {
                    'current_regime': self.current_regime,
                    'regime_strength': self.current_regime_strength,
                    'regime_duration': self.current_regime_duration.total_seconds() if self.current_regime_duration else 0,
                    'stability_score': self.stability_score,
                    'transition_probability': self.transition_probability
                },
                'adaptive_weights': self.get_current_weights(),
                'regime_stability': self.get_regime_stability_report(),
                'onchain_macro': self.get_onchain_macro_report(),
                'multidimensional_confidence': self.confidence_scorer.get_confidence_report(),
                'multi_timeframe_consensus': mtf_report,
                'liquidity_analysis': liquidity_report  # 🔥🔥🔥 유동성 분석 추가 🔥🔥🔥
            }
        except Exception as e:
            self.logger.error(f"Comprehensive analysis report error: {e}")
            return {}

    # 기존 메서드들 (market_regime_analyzer5.py와 동일하므로 생략)
    # _get_cached_data, _set_cached_data, _calculate_rsi, _calculate_macd, 
    # _analyze_volume_profile, _get_market_breadth, _get_macro_trend,
    # _get_market_volatility, _get_altcoin_fund_flow, _calculate_market_momentum,
    # _estimate_market_sentiment, _detect_market_state, _calculate_indicator_confidence,
    # _update_adaptive_weights, _calculate_weighted_regime_score, 
    # _track_indicator_performance, _evaluate_prediction_performance,
    # get_current_weights, get_weight_history,
    # _calculate_regime_strength, _calculate_stability_score, _update_dynamic_threshold,
    # _should_transition_to_new_regime, _is_frequent_flip, _predict_transition_probability,
    # _record_regime_transition, _update_regime_statistics, get_regime_stability_report,
    # analyze (메인 분석 함수), _calculate_confidence_score
    # ... 등등 (너무 길어져서 생략)

# 🎯 사용 예시:
# analyzer = MarketRegimeAnalyzer(market_data_manager)
#
# # 기본 분석 (기존과 동일한 인터페이스)
# regime, fund_flow = analyzer.analyze()
#
# # 🔥🔥🔥 추가 1: 유동성 종합 리포트 🔥🔥🔥
# liquidity_report = analyzer.liquidity_detector.get_comprehensive_liquidity_report()
# print(f"유동성 리포트: {liquidity_report}")
#
# # 🔥🔥🔥 추가 2: 종합 분석 리포트 (모든 시스템 통합 + 유동성) 🔥🔥🔥
# comprehensive_report = analyzer.get_comprehensive_analysis_report()
# print(f"종합 분석 리포트: {comprehensive_report}")
