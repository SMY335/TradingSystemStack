"""
Data API routes.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

from src.data import get_ohlcv
from src.api.models import OHLCVRequest, DataResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data", tags=["data"])


@router.post("/ohlcv", response_model=DataResponse)
async def fetch_ohlcv(request: OHLCVRequest):
    """Fetch OHLCV data for a symbol.

    Args:
        request: OHLCV request parameters

    Returns:
        OHLCV data with metadata

    Raises:
        HTTPException: If data loading fails
    """
    try:
        df = get_ohlcv(
            symbol=request.symbol,
            start=request.start,
            end=request.end,
            interval=request.interval,
            source=request.source
        )

        # Convert to dict for JSON response
        data = df.reset_index().to_dict(orient='records')

        return DataResponse(
            symbol=request.symbol,
            data=data,
            rows=len(df),
            columns=list(df.columns)
        )

    except Exception as e:
        logger.error(f"Failed to fetch OHLCV for {request.symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch data: {str(e)}"
        )
