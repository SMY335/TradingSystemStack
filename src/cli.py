"""
Command-line interface for TradingSystemStack.

Usage:
    python -m src.cli data fetch --symbol AAPL --start 2023-01-01
    python -m src.cli indicators run --symbol AAPL --indicator RSI --params '{"length":14}'
    python -m src.cli candlesticks detect --symbol AAPL --patterns HAMMER,DOJI
"""
import sys
from typing import Optional
import json
import logging

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    TYPER_AVAILABLE = True
except ImportError:
    TYPER_AVAILABLE = False
    print("Typer not available. Install: pip install typer rich")
    sys.exit(1)

from src.data import get_ohlcv
from src.indicators import run_indicator
from src.candlesticks import CandlestickDetector
from src.vwap import calculate_vwap
from src.zones import detect_zones

# Setup
app = typer.Typer(help="TradingSystemStack CLI")
console = Console()


@app.command()
def data_fetch(
    symbol: str = typer.Option(..., help="Symbol to fetch"),
    start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD)"),
    interval: str = typer.Option("1d", help="Timeframe"),
    output: Optional[str] = typer.Option(None, help="Output file (CSV/parquet)")
):
    """Fetch OHLCV data for a symbol."""
    try:
        console.print(f"[bold blue]Fetching {symbol}...[/bold blue]")

        df = get_ohlcv(
            symbol=symbol,
            start=start,
            end=end,
            interval=interval
        )

        console.print(f"[green]✓ Fetched {len(df)} rows[/green]")

        # Show preview
        table = Table(title=f"{symbol} OHLCV Data")
        for col in df.columns:
            table.add_column(col)

        for idx, row in df.head(10).iterrows():
            table.add_row(*[str(v) for v in row])

        console.print(table)

        # Save if requested
        if output:
            if output.endswith('.csv'):
                df.to_csv(output)
            elif output.endswith('.parquet'):
                df.to_parquet(output)
            console.print(f"[green]✓ Saved to {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def indicator_run(
    symbol: str = typer.Option(..., help="Symbol"),
    indicator: str = typer.Option(..., help="Indicator name (RSI, MACD, etc.)"),
    params: str = typer.Option("{}", help="Params as JSON"),
    start: Optional[str] = typer.Option(None, help="Start date"),
    end: Optional[str] = typer.Option(None, help="End date"),
    interval: str = typer.Option("1d", help="Timeframe")
):
    """Calculate indicator for a symbol."""
    try:
        console.print(f"[bold blue]Calculating {indicator} for {symbol}...[/bold blue]")

        # Fetch data
        df = get_ohlcv(symbol=symbol, start=start, end=end, interval=interval)

        # Parse params
        indicator_params = json.loads(params)

        # Calculate indicator
        result = run_indicator(indicator, df, params=indicator_params)

        console.print(f"[green]✓ Calculated {indicator}[/green]")

        # Show last 10 rows
        console.print(result.tail(10))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def candlestick_detect(
    symbol: str = typer.Option(..., help="Symbol"),
    patterns: Optional[str] = typer.Option(None, help="Comma-separated patterns"),
    start: Optional[str] = typer.Option(None, help="Start date"),
    end: Optional[str] = typer.Option(None, help="End date")
):
    """Detect candlestick patterns."""
    try:
        console.print(f"[bold blue]Detecting patterns for {symbol}...[/bold blue]")

        # Fetch data
        df = get_ohlcv(symbol=symbol, start=start, end=end)

        # Detect patterns
        detector = CandlestickDetector()

        if patterns:
            pattern_list = patterns.split(',')
            for pattern in pattern_list:
                result = detector.detect(df, pattern.strip())
                detected = result[result != 0]
                if len(detected) > 0:
                    console.print(f"[green]{pattern}: {len(detected)} occurrences[/green]")
        else:
            all_patterns = detector.detect_all(df)
            for col in all_patterns.columns:
                detected = all_patterns[col][all_patterns[col] != 0]
                if len(detected) > 0:
                    console.print(f"[green]{col}: {len(detected)} occurrences[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def vwap_calc(
    symbol: str = typer.Option(..., help="Symbol"),
    anchor: str = typer.Option("session", help="Anchor type"),
    start: Optional[str] = typer.Option(None, help="Start date"),
    end: Optional[str] = typer.Option(None, help="End date")
):
    """Calculate anchored VWAP."""
    try:
        console.print(f"[bold blue]Calculating VWAP for {symbol}...[/bold blue]")

        # Fetch data
        df = get_ohlcv(symbol=symbol, start=start, end=end)

        # Calculate VWAP
        vwap = calculate_vwap(df, anchor_type=anchor)

        console.print(f"[green]✓ Calculated {anchor} VWAP[/green]")
        console.print(vwap.tail(10))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def zones_detect(
    symbol: str = typer.Option(..., help="Symbol"),
    start: Optional[str] = typer.Option(None, help="Start date"),
    end: Optional[str] = typer.Option(None, help="End date")
):
    """Detect supply/demand zones."""
    try:
        console.print(f"[bold blue]Detecting zones for {symbol}...[/bold blue]")

        # Fetch data
        df = get_ohlcv(symbol=symbol, start=start, end=end)

        # Detect zones
        zones = detect_zones(df)

        console.print(f"[green]✓ Found {len(zones)} zones[/green]")

        for i, zone in enumerate(zones[:10], 1):
            console.print(
                f"{i}. {zone.zone_type.upper()}: "
                f"{zone.bottom:.2f}-{zone.top:.2f} "
                f"(strength={zone.strength:.1f})"
            )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
