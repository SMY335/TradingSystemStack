"""
Candlestick patterns API routes.
"""
from fastapi import APIRouter, HTTPException
import logging

from src.data import get_ohlcv
from src.candlesticks import CandlestickDetector
from src.api.models import CandlestickRequest, PatternResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/candlesticks", tags=["candlesticks"])


@router.post("/detect", response_model=PatternResponse)
async def detect_patterns(request: CandlestickRequest):
    """Detect candlestick patterns.

    Args:
        request: Pattern detection request

    Returns:
        Detected patterns
    """
    try:
        # Fetch OHLCV data
        df = get_ohlcv(
            symbol=request.symbol,
            start=request.start,
            end=request.end,
            interval=request.interval
        )

        # Detect patterns
        detector = CandlestickDetector()

        if request.patterns:
            # Detect specific patterns
            results = []
            for pattern in request.patterns:
                pattern_df = detector.detect(df, pattern)
                # Find non-zero signals
                detected = pattern_df[pattern_df != 0]
                if len(detected) > 0:
                    results.append({
                        'pattern': pattern,
                        'dates': detected.index.astype(str).tolist(),
                        'signals': detected.tolist()
                    })
        else:
            # Detect all patterns
            all_patterns = detector.detect_all(df)
            results = []
            for col in all_patterns.columns:
                detected = all_patterns[col][all_patterns[col] != 0]
                if len(detected) > 0:
                    results.append({
                        'pattern': col,
                        'dates': detected.index.astype(str).tolist(),
                        'signals': detected.tolist()
                    })

        return PatternResponse(
            symbol=request.symbol,
            patterns_detected=results,
            total_patterns=len(results)
        )

    except Exception as e:
        logger.error(f"Pattern detection failed for {request.symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pattern detection failed: {str(e)}"
        )
