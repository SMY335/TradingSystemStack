"""
Indicators API routes.
"""
from fastapi import APIRouter, HTTPException
import logging

from src.data import get_ohlcv
from src.indicators import run_indicator
from src.api.models import IndicatorRequest, IndicatorResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/indicators", tags=["indicators"])


@router.post("/run", response_model=IndicatorResponse)
async def run_indicator_endpoint(request: IndicatorRequest):
    """Calculate indicator for a symbol.

    Args:
        request: Indicator request parameters

    Returns:
        Indicator results with OHLCV data
    """
    try:
        # Fetch OHLCV data
        df = get_ohlcv(
            symbol=request.symbol,
            start=request.start,
            end=request.end,
            interval=request.interval
        )

        # Calculate indicator
        result = run_indicator(
            indicator_name=request.indicator,
            df=df,
            params=request.params
        )

        # Convert to dict
        data = result.reset_index().to_dict(orient='records')

        return IndicatorResponse(
            symbol=request.symbol,
            indicator=request.indicator,
            params=request.params,
            data=data,
            rows=len(result)
        )

    except Exception as e:
        logger.error(f"Failed to calculate {request.indicator} for {request.symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Indicator calculation failed: {str(e)}"
        )
