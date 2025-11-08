"""
VWAP API routes.
"""
from fastapi import APIRouter, HTTPException
import logging

from src.data import get_ohlcv
from src.vwap import calculate_vwap
from src.api.models import VWAPRequest, VWAPResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vwap", tags=["vwap"])


@router.post("/anchored", response_model=VWAPResponse)
async def calculate_anchored_vwap(request: VWAPRequest):
    """Calculate anchored VWAP.

    Args:
        request: VWAP request parameters

    Returns:
        VWAP data with optional bands
    """
    try:
        # Fetch OHLCV data
        df = get_ohlcv(
            symbol=request.symbol,
            start=request.start,
            end=request.end,
            interval=request.interval
        )

        # Calculate VWAP
        vwap_df = calculate_vwap(
            df,
            anchor_type=request.anchor_type,
            include_bands=request.include_bands,
            num_std=request.num_std
        )

        # Combine with OHLCV
        result = df.join(vwap_df)
        data = result.reset_index().to_dict(orient='records')

        return VWAPResponse(
            symbol=request.symbol,
            anchor_type=request.anchor_type,
            data=data,
            rows=len(result)
        )

    except Exception as e:
        logger.error(f"VWAP calculation failed for {request.symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"VWAP calculation failed: {str(e)}"
        )
