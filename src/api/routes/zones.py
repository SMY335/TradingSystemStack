"""
Supply/Demand zones API routes.
"""
from fastapi import APIRouter, HTTPException
import logging

from src.data import get_ohlcv
from src.zones import detect_zones
from src.api.models import ZonesRequest, ZonesResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/zones", tags=["zones"])


@router.post("/detect", response_model=ZonesResponse)
async def detect_supply_demand_zones(request: ZonesRequest):
    """Detect supply and demand zones.

    Args:
        request: Zone detection request

    Returns:
        Detected zones
    """
    try:
        # Fetch OHLCV data
        df = get_ohlcv(
            symbol=request.symbol,
            start=request.start,
            end=request.end,
            interval=request.interval
        )

        # Detect zones
        zones = detect_zones(
            df,
            consolidation_bars=request.consolidation_bars,
            impulse_threshold=request.impulse_threshold,
            max_zone_size=request.max_zone_size
        )

        # Convert zones to dict
        zones_data = []
        for zone in zones:
            zones_data.append({
                'type': zone.zone_type,
                'top': zone.top,
                'bottom': zone.bottom,
                'strength': zone.strength,
                'touches': zone.touches,
                'fresh': zone.fresh,
                'start_idx': zone.start_idx,
                'end_idx': zone.end_idx
            })

        return ZonesResponse(
            symbol=request.symbol,
            zones=zones_data,
            total_zones=len(zones)
        )

    except Exception as e:
        logger.error(f"Zone detection failed for {request.symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Zone detection failed: {str(e)}"
        )
