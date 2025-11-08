"""
Integration tests for FastAPI REST API.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from src.api.main import app


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    df = pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 102 + np.random.randn(100).cumsum(),
        'low': 98 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    # Ensure high >= low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test GET /health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data['status'] == 'healthy'
        assert 'version' in data
        assert data['version'] == '2.0.0'

    def test_root_health_check(self, client):
        """Test GET / endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert data['status'] == 'ok'
        assert data['version'] == '2.0.0'


class TestDataEndpoints:
    """Test data fetching endpoints."""

    @patch('src.api.routes.data.get_ohlcv')
    def test_fetch_ohlcv_basic(self, mock_get_ohlcv, client, sample_ohlcv_df):
        """Test POST /data/ohlcv endpoint."""
        mock_get_ohlcv.return_value = sample_ohlcv_df

        response = client.post(
            "/data/ohlcv",
            json={
                "symbol": "AAPL",
                "interval": "1d"
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data['symbol'] == 'AAPL'
        assert 'data' in data
        assert 'rows' in data
        assert 'columns' in data
        assert data['rows'] == 100

        mock_get_ohlcv.assert_called_once()

    @patch('src.api.routes.data.get_ohlcv')
    def test_fetch_ohlcv_with_dates(self, mock_get_ohlcv, client, sample_ohlcv_df):
        """Test fetching OHLCV with date range."""
        mock_get_ohlcv.return_value = sample_ohlcv_df

        response = client.post(
            "/data/ohlcv",
            json={
                "symbol": "BTC/USDT",
                "start": "2024-01-01",
                "end": "2024-03-31",
                "interval": "1h",
                "source": "yfinance"
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data['symbol'] == 'BTC/USDT'

        # Check call arguments
        call_args = mock_get_ohlcv.call_args
        assert call_args[1]['symbol'] == 'BTC/USDT'
        assert call_args[1]['start'] == '2024-01-01'
        assert call_args[1]['end'] == '2024-03-31'
        assert call_args[1]['interval'] == '1h'
        assert call_args[1]['source'] == 'yfinance'

    @patch('src.api.routes.data.get_ohlcv')
    def test_fetch_ohlcv_error(self, mock_get_ohlcv, client):
        """Test error handling in OHLCV endpoint."""
        mock_get_ohlcv.side_effect = Exception("Data fetch failed")

        response = client.post(
            "/data/ohlcv",
            json={"symbol": "INVALID"}
        )

        assert response.status_code == 500
        data = response.json()
        assert 'detail' in data


class TestIndicatorEndpoints:
    """Test indicator calculation endpoints."""

    @patch('src.api.routes.indicators.get_ohlcv')
    @patch('src.api.routes.indicators.run_indicator')
    def test_run_indicator_rsi(
        self,
        mock_run_indicator,
        mock_get_ohlcv,
        client,
        sample_ohlcv_df
    ):
        """Test POST /indicators/run endpoint with RSI."""
        mock_get_ohlcv.return_value = sample_ohlcv_df

        result_df = sample_ohlcv_df.copy()
        result_df['rsi'] = 50 + 10 * np.random.randn(100)
        mock_run_indicator.return_value = result_df

        response = client.post(
            "/indicators/run",
            json={
                "symbol": "AAPL",
                "indicator": "RSI",
                "params": {"length": 14}
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data['symbol'] == 'AAPL'
        assert data['indicator'] == 'RSI'
        assert data['params'] == {"length": 14}
        assert 'data' in data
        assert 'rows' in data

        mock_run_indicator.assert_called_once()

    @patch('src.api.routes.indicators.get_ohlcv')
    @patch('src.api.routes.indicators.run_indicator')
    def test_run_indicator_macd(
        self,
        mock_run_indicator,
        mock_get_ohlcv,
        client,
        sample_ohlcv_df
    ):
        """Test indicator endpoint with MACD."""
        mock_get_ohlcv.return_value = sample_ohlcv_df

        result_df = sample_ohlcv_df.copy()
        result_df['macd'] = np.random.randn(100)
        result_df['macd_signal'] = np.random.randn(100)
        result_df['macd_hist'] = np.random.randn(100)
        mock_run_indicator.return_value = result_df

        response = client.post(
            "/indicators/run",
            json={
                "symbol": "BTC/USDT",
                "indicator": "MACD",
                "params": {"fast": 12, "slow": 26, "signal": 9}
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data['indicator'] == 'MACD'

    @patch('src.api.routes.indicators.get_ohlcv')
    def test_run_indicator_error(self, mock_get_ohlcv, client):
        """Test error handling in indicator endpoint."""
        mock_get_ohlcv.side_effect = Exception("Failed to fetch data")

        response = client.post(
            "/indicators/run",
            json={
                "symbol": "AAPL",
                "indicator": "RSI"
            }
        )

        assert response.status_code == 500


class TestCandlestickEndpoints:
    """Test candlestick pattern endpoints."""

    @patch('src.api.routes.candlesticks.get_ohlcv')
    @patch('src.api.routes.candlesticks.CandlestickDetector')
    def test_detect_candlesticks(
        self,
        mock_detector_class,
        mock_get_ohlcv,
        client,
        sample_ohlcv_df
    ):
        """Test POST /candlesticks/detect endpoint."""
        mock_get_ohlcv.return_value = sample_ohlcv_df

        # Mock the detector instance
        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector

        # Mock detect method to return a Series with some detections
        import pandas as pd
        mock_series = pd.Series(0, index=sample_ohlcv_df.index)
        mock_series.iloc[10] = 100
        mock_detector.detect.return_value = mock_series

        response = client.post(
            "/candlesticks/detect",
            json={
                "symbol": "AAPL",
                "patterns": ["doji", "hammer"]
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data['symbol'] == 'AAPL'
        assert 'patterns_detected' in data


class TestVWAPEndpoints:
    """Test VWAP calculation endpoints."""

    @patch('src.api.routes.vwap.get_ohlcv')
    @patch('src.api.routes.vwap.calculate_vwap')
    def test_calculate_vwap(
        self,
        mock_calc_vwap,
        mock_get_ohlcv,
        client,
        sample_ohlcv_df
    ):
        """Test POST /vwap/anchored endpoint."""
        mock_get_ohlcv.return_value = sample_ohlcv_df

        # Mock calculate_vwap to return a DataFrame with VWAP columns
        vwap_df = pd.DataFrame({
            'vwap': 100 + np.random.randn(100)
        }, index=sample_ohlcv_df.index)
        mock_calc_vwap.return_value = vwap_df

        response = client.post(
            "/vwap/anchored",
            json={
                "symbol": "AAPL",
                "anchor_type": "session"
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data['symbol'] == 'AAPL'
        assert data['anchor_type'] == 'session'
        assert 'data' in data


class TestZonesEndpoints:
    """Test supply/demand zones endpoints."""

    @patch('src.api.routes.zones.get_ohlcv')
    @patch('src.api.routes.zones.detect_zones')
    def test_detect_zones(
        self,
        mock_detect,
        mock_get_ohlcv,
        client,
        sample_ohlcv_df
    ):
        """Test POST /zones/detect endpoint."""
        mock_get_ohlcv.return_value = sample_ohlcv_df

        # Mock zones response with Zone objects
        from src.zones import Zone

        mock_zone = Zone(
            zone_type='demand',
            top=95.0,
            bottom=90.0,
            start_idx=10,
            end_idx=15,
            impulse_idx=16,
            strength=75.0,
            touches=2,
            fresh=True
        )
        mock_detect.return_value = [mock_zone]

        response = client.post(
            "/zones/detect",
            json={
                "symbol": "AAPL",
                "consolidation_bars": 5,
                "impulse_threshold": 0.02
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data['symbol'] == 'AAPL'
        assert 'zones' in data
        assert len(data['zones']) == 1
        assert data['zones'][0]['type'] == 'demand'


class TestRequestValidation:
    """Test request validation."""

    def test_missing_required_field(self, client):
        """Test error when required field is missing."""
        response = client.post(
            "/data/ohlcv",
            json={}  # Missing 'symbol'
        )

        assert response.status_code == 422  # Validation error

    def test_invalid_field_type(self, client):
        """Test error when field has wrong type."""
        response = client.post(
            "/indicators/run",
            json={
                "symbol": "AAPL",
                "indicator": "RSI",
                "params": "invalid"  # Should be dict
            }
        )

        assert response.status_code == 422


class TestCORS:
    """Test CORS middleware."""

    def test_cors_middleware_configured(self, client):
        """Test that CORS middleware is configured in the app."""
        from src.api.main import app
        from fastapi.middleware.cors import CORSMiddleware

        # Check that CORS middleware is in the middleware stack
        cors_middlewares = [
            m for m in app.user_middleware
            if m.cls == CORSMiddleware
        ]

        assert len(cors_middlewares) > 0, "CORS middleware should be configured"


class TestDocumentation:
    """Test API documentation endpoints."""

    def test_openapi_schema(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()

        assert schema['info']['title'] == "TradingSystemStack API"
        assert 'paths' in schema

    def test_docs_endpoint(self, client):
        """Test that Swagger UI is available."""
        response = client.get("/docs")

        assert response.status_code == 200

    def test_redoc_endpoint(self, client):
        """Test that ReDoc is available."""
        response = client.get("/redoc")

        assert response.status_code == 200
