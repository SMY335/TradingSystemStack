"""
Integration tests for institutional trading frameworks
Tests Nautilus Trader, Backtrader, Riskfolio-Lib, and ArcticDB installations
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_nautilus_installation():
    """Test Nautilus Trader installation and basic functionality"""
    try:
        from nautilus_trader.backtest.engine import BacktestEngine
        from nautilus_trader.model.identifiers import Venue
        
        # Create a basic backtest engine
        engine = BacktestEngine()
        assert engine is not None
        
        # Test venue creation
        venue = Venue("BINANCE")
        assert venue.value == "BINANCE"
        
        print("✅ Nautilus Trader: OK")
        return True
    except Exception as e:
        print(f"❌ Nautilus Trader: FAILED - {str(e)}")
        return False


def test_backtrader_installation():
    """Test Backtrader installation and basic functionality"""
    try:
        import backtrader as bt
        
        # Create a basic cerebro instance
        cerebro = bt.Cerebro()
        assert cerebro is not None
        
        # Test strategy addition
        class TestStrategy(bt.Strategy):
            def next(self):
                pass
        
        cerebro.addstrategy(TestStrategy)
        assert len(cerebro.strats) > 0
        
        print("✅ Backtrader: OK")
        return True
    except Exception as e:
        print(f"❌ Backtrader: FAILED - {str(e)}")
        return False


def test_riskfolio_installation():
    """Test Riskfolio-Lib installation and basic functionality"""
    try:
        import riskfolio as rp
        import pandas as pd
        import numpy as np
        
        # Test Portfolio class exists
        assert hasattr(rp, 'Portfolio')
        
        # Create a simple portfolio object
        port = rp.Portfolio(returns=pd.DataFrame(
            np.random.randn(100, 3), 
            columns=['A', 'B', 'C']
        ))
        assert port is not None
        
        print("✅ Riskfolio-Lib: OK")
        return True
    except Exception as e:
        print(f"❌ Riskfolio-Lib: FAILED - {str(e)}")
        return False


def test_arctic_installation():
    """Test ArcticDB installation and library creation"""
    try:
        from src.infrastructure.arctic_manager import ArcticManager
        import shutil
        from pathlib import Path
        
        # Use a temporary test database
        test_db_path = "data/arctic_db_test"
        
        # Clean up if exists
        if Path(test_db_path).exists():
            shutil.rmtree(test_db_path)
        
        # Create manager
        manager = ArcticManager(db_path=test_db_path)
        
        # Check libraries
        libs = manager.arctic.list_libraries()
        assert 'market_data' in libs
        assert 'orderbook' in libs
        assert 'trades' in libs
        assert 'backtest_results' in libs
        
        print(f"✅ ArcticDB: OK ({len(libs)} libraries)")
        
        # Clean up test database
        shutil.rmtree(test_db_path)
        
        return True
    except Exception as e:
        print(f"❌ ArcticDB: FAILED - {str(e)}")
        return False


def test_ta_lib_installation():
    """Test TA-Lib installation"""
    try:
        import talib
        import numpy as np
        
        # Test a simple indicator
        close = np.random.randn(100)
        sma = talib.SMA(close, timeperiod=10)
        assert sma is not None
        
        print("✅ TA-Lib: OK")
        return True
    except Exception as e:
        print(f"❌ TA-Lib: FAILED - {str(e)}")
        return False


def test_sklearn_installation():
    """Test scikit-learn installation"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Test basic model creation
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        
        assert clf.score(X, y) > 0.5
        
        print("✅ Scikit-learn: OK")
        return True
    except Exception as e:
        print(f"❌ Scikit-learn: FAILED - {str(e)}")
        return False


def test_scipy_installation():
    """Test SciPy installation"""
    try:
        from scipy import stats
        import numpy as np
        
        # Test statistical function
        data = np.random.randn(100)
        result = stats.normaltest(data)
        
        assert result is not None
        
        print("✅ SciPy: OK")
        return True
    except Exception as e:
        print(f"❌ SciPy: FAILED - {str(e)}")
        return False


def test_statsmodels_installation():
    """Test statsmodels installation"""
    try:
        import statsmodels.api as sm
        import numpy as np
        
        # Test basic regression
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()
        
        assert results is not None
        
        print("✅ Statsmodels: OK")
        return True
    except Exception as e:
        print(f"❌ Statsmodels: FAILED - {str(e)}")
        return False


def test_hummingbot_installation():
    """Test Hummingbot installation"""
    try:
        import hummingbot
        
        # Just check import works
        assert hummingbot is not None
        
        print("✅ Hummingbot: OK")
        return True
    except Exception as e:
        print(f"❌ Hummingbot: FAILED - {str(e)}")
        return False


def run_all_tests():
    """Run all framework tests and report results"""
    print("=" * 70)
    print("INSTITUTIONAL TRADING FRAMEWORKS VALIDATION")
    print("=" * 70)
    print()
    
    results = {
        "Nautilus Trader": test_nautilus_installation(),
        "Backtrader": test_backtrader_installation(),
        "Riskfolio-Lib": test_riskfolio_installation(),
        "ArcticDB": test_arctic_installation(),
        "TA-Lib": test_ta_lib_installation(),
        "Scikit-learn": test_sklearn_installation(),
        "SciPy": test_scipy_installation(),
        "Statsmodels": test_statsmodels_installation(),
        "Hummingbot": test_hummingbot_installation(),
    }
    
    print()
    print("=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} ({passed/total*100:.1f}%)")
    print()
    
    if passed == total:
        print("🎉 ALL FRAMEWORKS ARE OPERATIONAL!")
        return 0
    else:
        print("⚠️  Some frameworks failed. Please review the errors above.")
        failed = [k for k, v in results.items() if not v]
        print(f"\nFailed frameworks: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
