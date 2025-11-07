# Plan Maître - Plateforme de Trading Institutionnelle Complète

**Date**: 2025-11-07
**Objectif**: Fusionner TradingSystemStack existant + 18 modules professionnels en UNE plateforme complète
**Criticité**: MAXIMUM - Production ready pour argent réel

---

## Vision : Plateforme de Trading Institutionnelle Tout-en-Un

### Capacités Cibles
1. ✅ **Analyse Technique Avancée** - 200+ indicateurs (TA-Lib, pandas-ta)
2. ✅ **Patterns & Structures** - Chandeliers, chart patterns, trendlines
3. ✅ **Analyse Institutionnelle** - ICT, Order Flow, Supply/Demand, Market Breadth
4. ✅ **Analyse Fondamentale** - Ratios, états financiers, 30+ ans d'historique
5. ✅ **Analyse Macro** - Indicateurs FRED (CPI, GDP, rates)
6. ✅ **Analyse Relative** - Force relative, matrices RS
7. ✅ **Analyse Sentiment** - News, Twitter, Reddit, Fear/Greed Index
8. ✅ **Scanner Avancé** - DSL JSON pour scans déclaratifs multi-conditions
9. ✅ **Backtesting Vectorisé** - VectorBT haute performance
10. ✅ **Portfolio Management** - Risk, optimization, attribution
11. ✅ **Paper/Live Trading** - Exécution avec CCXT
12. ✅ **API REST** - FastAPI exposant toutes capacités
13. ✅ **CLI Puissant** - Interface ligne de commande complète
14. ✅ **Dashboards** - Streamlit multi-vues
15. ✅ **Machine Learning** - Feature engineering, prédictions

---

## Phase 0 : Audit Complet de l'Existant

### 0.1 Inventaire TradingSystemStack Actuel

#### Modules Existants (À CONSERVER et AMÉLIORER)

**src/strategies/** - Stratégies de trading
- ✅ `base_strategy.py` - Classe abstraite
- ✅ `ema_strategy.py` - EMA crossover (à refactoriser)
- ✅ `macd_strategy.py` - MACD (à refactoriser)
- ✅ `rsi_strategy.py` - RSI (à refactoriser)
- ✅ `bollinger_strategy.py` - Bollinger Bands (à refactoriser)
- ✅ `supertrend_strategy.py` - SuperTrend (à refactoriser)
- ✅ `ichimoku_strategy.py` - Ichimoku Cloud (à refactoriser)
- **STATUS**: Calculs manuels → MIGRER vers TA-Lib/pandas-ta

**src/backtesting/** - Moteur de backtest
- ✅ `engine.py` - BacktestEngine avec VectorBT
- ✅ `advanced_engine.py` - Walk-forward, Monte Carlo
- ✅ `metrics.py` - KPIs avancés
- ✅ `monte_carlo.py` - Simulations MC
- **STATUS**: EXCELLENT - Garder et étendre

**src/portfolio/** - Gestion de portefeuille
- ✅ `portfolio_manager.py` - Gestion positions
- ✅ `risk_manager.py` - VaR, CVaR, max drawdown
- ✅ `optimizer.py` - Optimisation Riskfolio-Lib
- ✅ `performance_attribution.py` - Attribution de performance
- **STATUS**: SOLIDE - Garder

**src/ict_strategies/** - Smart Money Concepts
- ✅ `order_blocks.py` - Détection order blocks
- ✅ `fair_value_gaps.py` - FVG detection
- ✅ `liquidity_pools.py` - Liquidity sweeps
- ✅ `ict_strategy.py` - Stratégie complète ICT
- **STATUS**: UNIQUE - Garder et compléter avec smartmoneyconcepts

**src/ml/** - Machine Learning
- ✅ `ml_predictor.py` - Modèles prédictifs
- ✅ `feature_engineering.py` - 50+ features (utilise `ta` library)
- **STATUS**: BON - Garder et étendre

**src/paper_trading/** - Trading papier/live
- ✅ `engine.py` - Moteur de trading
- ✅ `live_bot.py` - Bot live
- ✅ `telegram_notifier.py` - Notifications
- **STATUS**: FONCTIONNEL - Garder

**src/adapters/** - Adapters multi-framework
- ✅ `ema_adapter.py` - Utilise TA-Lib (BIEN!)
- ✅ `macd_adapter.py` - Utilise TA-Lib (BIEN!)
- ✅ `rsi_adapter.py` - Utilise TA-Lib (BIEN!)
- ✅ `ict_adapter.py` - ICT pour Nautilus/Backtrader
- ✅ `pairs_adapter.py` - Pairs trading
- ✅ `mm_adapter.py` - Market making
- **STATUS**: BON - Garder comme référence

**src/dashboard/** - Interfaces utilisateur
- ✅ `app.py` - Dashboard principal Streamlit
- ✅ `live_dashboard.py` - Monitoring live
- ✅ `risk_dashboard.py` - Vue risques
- ✅ `portfolio_dashboard.py` - Vue portfolio
- ✅ `attribution_dashboard.py` - Attribution
- ✅ `nlp_strategy_editor.py` - Éditeur stratégies NLP
- **STATUS**: RICHE - Garder et unifier

**src/data_sources/** - Sources de données
- ✅ `crypto_data.py` - Données crypto
- **STATUS**: BASIQUE - Étendre avec module data/ unifié

**src/infrastructure/** - Infrastructure données
- ✅ `arctic_manager.py` - Stockage Arctic
- ✅ `data_manager.py` - Gestion données
- **STATUS**: BON - Intégrer avec nouveau module data/

**src/optimization/** - Optimisation stratégies
- ✅ `walk_forward.py` - Walk-forward analysis
- **STATUS**: BON - Garder

**src/quant_strategies/** - Stratégies quantitatives
- ✅ `pairs_trading.py` - Pairs trading
- **STATUS**: BON - Garder

**src/market_making/** - Market making
- ✅ `simple_mm.py` - Market maker simple
- **STATUS**: BON - Garder

**src/nlp_strategy/** - NLP pour stratégies
- ✅ `strategy_parser.py` - Parser NLP
- ✅ `strategy_pipeline.py` - Pipeline
- ✅ `code_generator.py` - Génération code (sécurisé PR#2)
- **STATUS**: INNOVANT - Garder

**tests/unit/** - Tests unitaires
- ✅ `test_backtesting_validation.py` - 15 tests
- ✅ `test_strategy_validation.py` - 18 tests
- ✅ `test_portfolio_validation.py` - 28 tests
- ✅ `test_bollinger_validation.py` - 24 tests
- ✅ `test_supertrend_validation.py` - 26 tests
- ✅ `test_ichimoku_validation.py` - 30 tests
- **Total**: 141 tests
- **STATUS**: EXCELLENT - Étendre pour nouveaux modules

#### Fichiers de Configuration
- ✅ `pyproject.toml` - Configuration projet
- ✅ `requirements_frameworks.txt` - Frameworks (Nautilus, Backtrader, VectorBT)
- **STATUS**: À fusionner avec requirements.txt unifié

### 0.2 Modules Manquants (À CRÉER)

| Module | Priorité | Complexité | Temps Estimé |
|--------|----------|------------|--------------|
| **indicators/** (refactorisé) | P0 - CRITIQUE | Élevée | 16h |
| **candlesticks/** (60+ patterns) | P0 - CRITIQUE | Moyenne | 8h |
| **patterns/** (chart patterns) | P1 - Haute | Élevée | 12h |
| **trendlines/** (auto-detection) | P1 - Haute | Élevée | 10h |
| **vwap/** (anchored) | P1 - Haute | Moyenne | 6h |
| **zones/** (supply/demand) | P1 - Haute | Moyenne | 8h |
| **breadth/** (market breadth) | P2 - Moyenne | Moyenne | 6h |
| **relativereturns/** (RS) | P2 - Moyenne | Moyenne | 8h |
| **raindrop/** (volume profile) | P3 - Basse | Faible | 4h |
| **fundamentals/** (ratios) | P2 - Moyenne | Faible | 6h |
| **economics/** (FRED) | P2 - Moyenne | Faible | 4h |
| **sentiment/** (news/social) | P1 - Haute | Élevée | 12h |
| **scanner/** (DSL engine) | P1 - Haute | Très Élevée | 16h |
| **api/** (FastAPI) | P0 - CRITIQUE | Moyenne | 12h |
| **cli/** (CLI unifié) | P1 - Haute | Moyenne | 8h |
| **utils/** (types, registry) | P0 - CRITIQUE | Faible | 4h |
| **data/** (loaders unifiés) | P0 - CRITIQUE | Moyenne | 8h |
| **tests/** (nouveaux modules) | P0 - CRITIQUE | Élevée | 16h |

**TOTAL ESTIMÉ: 164 heures (20-25 jours de travail rigoureux)**

---

## Phase 1 : Architecture Unifiée

### 1.1 Structure Cible Finale

```
TradingSystemStack/
├── src/
│   ├── __init__.py
│   │
│   ├── utils/                      # ⭐ NOUVEAU - Utilitaires core
│   │   ├── __init__.py
│   │   ├── types.py                # TypedDict OHLCV, Universe, Params
│   │   ├── io.py                   # Lecture/écriture parquet/csv
│   │   ├── timeframes.py           # Conversions timeframes
│   │   ├── registry.py             # Registre indicateurs/scanners
│   │   └── logging_config.py       # Configuration logging unifiée
│   │
│   ├── data/                       # ⭐ REFONTE - Ingestion unifiée
│   │   ├── __init__.py
│   │   ├── loaders.py              # yfinance, CCXT, CSV, Parquet
│   │   ├── fred.py                 # pandas_datareader FRED
│   │   ├── normalizers.py          # Normalisation colonnes OHLCV
│   │   └── cache.py                # Cache parquet intelligent
│   │   # INTÉGRER: src/data_sources/*, src/infrastructure/*
│   │
│   ├── indicators/                 # ⭐ REFACTORISÉ - Indicateurs pro
│   │   ├── __init__.py
│   │   ├── base.py                 # Classes abstraites
│   │   ├── config.py               # Configuration globale
│   │   ├── exceptions.py           # Exceptions personnalisées
│   │   ├── core.py                 # run_indicator() dynamique
│   │   ├── factory.py              # Création dynamique
│   │   ├── validators.py           # Validation inputs
│   │   ├── wrappers/
│   │   │   ├── __init__.py
│   │   │   ├── talib_wrapper.py    # TA-Lib (150+ indicateurs)
│   │   │   ├── pandas_ta_wrapper.py # pandas-ta (130+ indicateurs)
│   │   │   ├── ta_wrapper.py       # ta library (volume, VWAP)
│   │   │   ├── smc_wrapper.py      # smartmoneyconcepts
│   │   │   └── vectorbt_wrapper.py # VectorBT multi-params
│   │   └── performance.py          # Benchmarking
│   │
│   ├── candlesticks/               # ⭐ NOUVEAU - Motifs chandeliers
│   │   ├── __init__.py
│   │   ├── talib_patterns.py       # 60+ patterns TA-Lib
│   │   ├── pandas_ta_patterns.py   # Fallback pandas-ta
│   │   └── detector.py             # Détection unifiée
│   │
│   ├── patterns/                   # ⭐ NOUVEAU - Chart patterns
│   │   ├── __init__.py
│   │   ├── chart_patterns.py       # Triangles, ETE, double top/bottom
│   │   ├── geometry.py             # Calculs géométriques
│   │   └── models.py               # Dataclasses pour patterns
│   │
│   ├── trendlines/                 # ⭐ NOUVEAU - Trendlines auto
│   │   ├── __init__.py
│   │   ├── detector.py             # Détection S/R automatique
│   │   ├── breakout.py             # Détection cassures
│   │   └── algorithms.py           # Algos (Hough transform, regression)
│   │
│   ├── vwap/                       # ⭐ NOUVEAU - Anchored VWAP
│   │   ├── __init__.py
│   │   ├── anchored_vwap.py        # VWAP avec ancres multiples
│   │   └── anchors.py              # Définitions ancres
│   │
│   ├── raindrop/                   # ⭐ NOUVEAU - Volume profile candles
│   │   ├── __init__.py
│   │   └── raindrop_plotly.py      # Bougies avec profil volume
│   │
│   ├── breadth/                    # ⭐ NOUVEAU - Market breadth
│   │   ├── __init__.py
│   │   ├── breadth_core.py         # % above SMA, A/D line, McClellan
│   │   └── universe.py             # Gestion univers multi-symboles
│   │
│   ├── relativereturns/            # ⭐ NOUVEAU - Force relative
│   │   ├── __init__.py
│   │   ├── rs_core.py              # RS ratio, RS matrix
│   │   └── ranking.py              # Classements RS
│   │
│   ├── zones/                      # ⭐ NOUVEAU - Supply/Demand formalisé
│   │   ├── __init__.py
│   │   ├── supply_demand.py        # Détection zones SD
│   │   └── merger.py               # Fusion zones proches
│   │
│   ├── fundamentals/               # ⭐ NOUVEAU - Analyse fondamentale
│   │   ├── __init__.py
│   │   ├── fundamentals_ft.py      # FinanceToolkit wrapper
│   │   └── ratios.py               # Ratios clés (PE, ROE, etc.)
│   │
│   ├── economics/                  # ⭐ NOUVEAU - Indicateurs macro
│   │   ├── __init__.py
│   │   ├── macro.py                # CPI, GDP, Unemployment, Rates
│   │   └── fred_client.py          # Client FRED avec cache
│   │
│   ├── sentiment/                  # ⭐ NOUVEAU - Analyse sentiment
│   │   ├── __init__.py
│   │   ├── news.py                 # News sentiment (NewsAPI, Finnhub)
│   │   ├── social.py               # Twitter/Reddit (snscrape, PRAW)
│   │   ├── fear_greed.py           # Fear & Greed Index
│   │   ├── nlp.py                  # NLP processing (transformers)
│   │   └── aggregator.py           # Agrégation scores sentiment
│   │
│   ├── scanner/                    # ⭐ NOUVEAU - Scanner DSL
│   │   ├── __init__.py
│   │   ├── dsl.py                  # Grammaire JSON scans
│   │   ├── parser.py               # Parser DSL → AST
│   │   ├── engine.py               # Exécution parallèle
│   │   ├── operators.py            # Opérateurs (>, <, ==, crosses, etc.)
│   │   └── examples/               # Exemples de scans JSON
│   │
│   ├── strategies/                 # ✅ EXISTANT - À refactoriser
│   │   ├── __init__.py
│   │   ├── base_strategy.py        # Garder
│   │   ├── ema_strategy.py         # REFACTORISER avec TA-Lib
│   │   ├── macd_strategy.py        # REFACTORISER avec TA-Lib
│   │   ├── rsi_strategy.py         # REFACTORISER avec TA-Lib
│   │   ├── bollinger_strategy.py   # REFACTORISER avec TA-Lib
│   │   ├── supertrend_strategy.py  # REFACTORISER avec pandas-ta
│   │   └── ichimoku_strategy.py    # REFACTORISER avec pandas-ta
│   │
│   ├── ict_strategies/             # ✅ EXISTANT - Garder et améliorer
│   │   ├── __init__.py
│   │   ├── order_blocks.py         # Garder
│   │   ├── fair_value_gaps.py      # Garder
│   │   ├── liquidity_pools.py      # Garder
│   │   └── ict_strategy.py         # Garder + intégrer smc lib
│   │
│   ├── backtesting/                # ✅ EXISTANT - Excellent, étendre
│   │   ├── __init__.py
│   │   ├── engine.py               # Garder
│   │   ├── advanced_engine.py      # Garder
│   │   ├── metrics.py              # Garder
│   │   ├── monte_carlo.py          # Garder
│   │   ├── vbt_wrappers.py         # NOUVEAU - Wrappers VectorBT avancés
│   │   └── strategies.py           # NOUVEAU - Stratégies prédéfinies
│   │
│   ├── portfolio/                  # ✅ EXISTANT - Garder
│   │   ├── __init__.py
│   │   ├── models.py               # Garder
│   │   ├── portfolio_manager.py    # Garder
│   │   ├── risk_manager.py         # Garder
│   │   ├── optimizer.py            # Garder
│   │   └── performance_attribution.py # Garder
│   │
│   ├── ml/                         # ✅ EXISTANT - Garder et étendre
│   │   ├── __init__.py
│   │   ├── ml_predictor.py         # Garder
│   │   └── feature_engineering.py  # Garder + intégrer nouveaux indicateurs
│   │
│   ├── paper_trading/              # ✅ EXISTANT - Garder
│   │   ├── __init__.py
│   │   ├── engine.py               # Garder
│   │   ├── live_bot.py             # Garder
│   │   ├── models.py               # Garder
│   │   ├── logger_config.py        # Garder
│   │   └── telegram_notifier.py    # Garder
│   │
│   ├── adapters/                   # ✅ EXISTANT - Garder comme référence
│   │   ├── __init__.py
│   │   ├── base_strategy_adapter.py # Garder
│   │   ├── ema_adapter.py          # Référence TA-Lib usage
│   │   ├── macd_adapter.py         # Référence TA-Lib usage
│   │   ├── rsi_adapter.py          # Référence TA-Lib usage
│   │   ├── ict_adapter.py          # Garder
│   │   ├── pairs_adapter.py        # Garder
│   │   ├── mm_adapter.py           # Garder
│   │   └── strategy_factory.py     # Garder
│   │
│   ├── optimization/               # ✅ EXISTANT - Garder
│   │   ├── __init__.py
│   │   └── walk_forward.py         # Garder
│   │
│   ├── quant_strategies/           # ✅ EXISTANT - Garder
│   │   ├── __init__.py
│   │   └── pairs_trading.py        # Garder
│   │
│   ├── market_making/              # ✅ EXISTANT - Garder
│   │   ├── __init__.py
│   │   └── simple_mm.py            # Garder
│   │
│   ├── nlp_strategy/               # ✅ EXISTANT - Garder
│   │   ├── __init__.py
│   │   ├── strategy_parser.py      # Garder
│   │   ├── strategy_pipeline.py    # Garder
│   │   └── code_generator.py       # Garder (sécurisé)
│   │
│   ├── api/                        # ⭐ NOUVEAU - API REST FastAPI
│   │   ├── __init__.py
│   │   ├── main.py                 # Application FastAPI principale
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── data.py             # /data/ohlcv, /data/fred
│   │   │   ├── indicators.py       # /indicators/run
│   │   │   ├── candlesticks.py     # /candlesticks/detect
│   │   │   ├── patterns.py         # /patterns/chart
│   │   │   ├── trendlines.py       # /trendlines/compute
│   │   │   ├── vwap.py             # /vwap/anchored
│   │   │   ├── raindrop.py         # /raindrop/plot
│   │   │   ├── breadth.py          # /breadth/pct_above_sma
│   │   │   ├── rs.py               # /rs/ratio, /rs/matrix
│   │   │   ├── zones.py            # /zones/supply_demand
│   │   │   ├── fundamentals.py     # /fundamentals/ratios
│   │   │   ├── economics.py        # /economics/fred
│   │   │   ├── sentiment.py        # /sentiment/news, /sentiment/social
│   │   │   ├── scanner.py          # /scanner/run
│   │   │   └── backtest.py         # /backtest/run
│   │   ├── models/                 # Pydantic models
│   │   │   ├── __init__.py
│   │   │   ├── requests.py         # Request models
│   │   │   └── responses.py        # Response models
│   │   └── middleware.py           # Auth, CORS, rate limiting
│   │
│   ├── dashboard/                  # ✅ EXISTANT - Unifier
│   │   ├── __init__.py
│   │   ├── app.py                  # Dashboard principal unifié
│   │   ├── pages/
│   │   │   ├── live_dashboard.py   # Garder
│   │   │   ├── risk_dashboard.py   # Garder
│   │   │   ├── portfolio_dashboard.py # Garder
│   │   │   ├── attribution_dashboard.py # Garder
│   │   │   ├── scanner_dashboard.py # NOUVEAU
│   │   │   ├── sentiment_dashboard.py # NOUVEAU
│   │   │   └── patterns_dashboard.py # NOUVEAU
│   │   └── nlp_strategy_editor.py  # Garder
│   │
│   └── cli.py                      # ⭐ NOUVEAU - CLI unifié Typer
│
├── tests/
│   ├── __init__.py
│   ├── unit/                       # ✅ EXISTANT - Étendre
│   │   ├── test_backtesting_validation.py # 15 tests - Garder
│   │   ├── test_strategy_validation.py    # 18 tests - Garder
│   │   ├── test_portfolio_validation.py   # 28 tests - Garder
│   │   ├── test_bollinger_validation.py   # 24 tests - Garder
│   │   ├── test_supertrend_validation.py  # 26 tests - Garder
│   │   ├── test_ichimoku_validation.py    # 30 tests - Garder
│   │   ├── test_indicators.py      # NOUVEAU - Tests indicateurs
│   │   ├── test_candlesticks.py    # NOUVEAU
│   │   ├── test_patterns.py        # NOUVEAU
│   │   ├── test_trendlines.py      # NOUVEAU
│   │   ├── test_vwap.py            # NOUVEAU
│   │   ├── test_raindrop.py        # NOUVEAU
│   │   ├── test_breadth.py         # NOUVEAU
│   │   ├── test_rs.py              # NOUVEAU
│   │   ├── test_zones.py           # NOUVEAU
│   │   ├── test_sentiment.py       # NOUVEAU
│   │   └── test_scanner.py         # NOUVEAU
│   ├── integration/
│   │   ├── test_api_smoke.py       # NOUVEAU - Tests API
│   │   ├── test_backtest.py        # NOUVEAU - Tests backtest complets
│   │   └── test_frameworks.py      # Garder
│   └── data/                       # Datasets de test
│       ├── reference/              # Résultats de référence
│       └── input/                  # Données d'entrée
│
├── docs/                           # ✅ EXISTANT - Étendre
│   ├── REFACTORING_PLAN.md        # Garder (référence historique)
│   ├── MASTER_PLAN.md             # CE DOCUMENT
│   ├── ARCHITECTURE.md            # NOUVEAU - Architecture détaillée
│   ├── API_REFERENCE.md           # NOUVEAU - Docs API
│   ├── CLI_REFERENCE.md           # NOUVEAU - Docs CLI
│   ├── SCANNER_DSL.md             # NOUVEAU - Grammaire DSL
│   ├── INDICATORS_CATALOG.md      # NOUVEAU - Catalogue 200+ indicateurs
│   └── MIGRATION_GUIDE.md         # NOUVEAU - Guide migration
│
├── examples/                       # ⭐ NOUVEAU - Exemples d'usage
│   ├── notebooks/
│   │   ├── 01_indicators_demo.ipynb
│   │   ├── 02_patterns_demo.ipynb
│   │   ├── 03_trendlines_demo.ipynb
│   │   ├── 04_vwap_demo.ipynb
│   │   ├── 05_breadth_demo.ipynb
│   │   ├── 06_rs_matrix_demo.ipynb
│   │   ├── 07_zones_demo.ipynb
│   │   ├── 08_sentiment_demo.ipynb
│   │   ├── 09_scanner_demo.ipynb
│   │   └── 10_backtest_demo.ipynb
│   ├── scripts/
│   │   ├── compute_indicators.py
│   │   ├── detect_patterns.py
│   │   ├── run_scanner.py
│   │   └── run_backtest.py
│   └── scans/
│       ├── ema_golden_cross.json
│       ├── hammer_at_support.json
│       ├── rs_leaders.json
│       └── supply_demand_zones.json
│
├── data/                           # ⭐ NOUVEAU - Données locales
│   ├── cache/                      # Cache parquet
│   ├── fred/                       # Données FRED
│   └── fundamentals/               # Données fondamentales
│
├── README.md                       # ✅ REFONTE COMPLÈTE
├── pyproject.toml                  # ✅ Mise à jour
├── requirements.txt                # ✅ Unifié et complet
├── Makefile                        # ⭐ NOUVEAU - Scripts installation
└── .env.example                    # ⭐ NOUVEAU - Variables environnement
```

### 1.2 Dépendances Unifiées

**requirements.txt final** :

```txt
# Core Data & Computation
pandas>=2.2
numpy>=1.26
numba>=0.60
polars>=1.9
pyarrow>=17.0
scipy>=1.13
scikit-learn>=1.5

# Technical Analysis - Core Libraries
TA-Lib>=0.4.28              # 150+ indicators (C-based, fastest)
pandas-ta>=0.4.67           # 130+ indicators (SuperTrend, Ichimoku, etc.)
ta>=0.11.0                  # Volume indicators, VWAP
smartmoneyconcepts>=0.0.3   # ICT indicators (Order Blocks, FVG, BOS, CHOCH)

# Backtesting & Portfolio
vectorbt>=0.24              # Vectorized backtesting
quantstats>=0.0.62          # Performance metrics
empyrical-reloaded>=0.5.9   # Pyfolio metrics
riskfolio-lib>=6.0          # Portfolio optimization

# Trading Frameworks (for adapters)
nautilus_trader>=1.198.0    # Production trading
backtrader>=1.9.78.123      # Backtesting framework
freqtrade>=2024.1           # Crypto trading bot

# Data Sources
yfinance>=0.2               # Yahoo Finance
ccxt>=4.0                   # Crypto exchanges
pandas-datareader>=0.10     # FRED, World Bank, etc.
financetoolkit>=1.6         # Fundamentals (30+ years, 150+ metrics)

# Sentiment Analysis
newsapi-python>=0.2.7       # News API
praw>=7.7                   # Reddit API
snscrape>=0.7               # Twitter scraping (no API key needed)
transformers>=4.35          # NLP models (FinBERT, etc.)
torch>=2.1                  # PyTorch for transformers
requests>=2.32              # HTTP requests

# Chart Patterns & Trendlines
opencv-python>=4.8          # Image processing for patterns
scikit-image>=0.22          # Hough transform for trendlines

# Visualization
plotly>=5.24                # Interactive charts
matplotlib>=3.7.0           # Static charts
seaborn>=0.12.0             # Statistical viz
kaleido>=0.2.1              # Static image export Plotly

# API & Web
fastapi>=0.115              # REST API
uvicorn>=0.30               # ASGI server
pydantic>=2.8               # Data validation
python-multipart>=0.0.6     # File uploads
python-jose[cryptography]>=3.3  # JWT tokens
passlib[bcrypt]>=1.7        # Password hashing

# CLI
typer>=0.9                  # Modern CLI
rich>=13.7                  # Rich terminal output
click>=8.1                  # CLI framework

# Dashboard
streamlit>=1.28             # Web dashboards

# Data Storage
arcticdb>=4.0.0             # TimeSeries database
sqlalchemy>=2.0             # SQL toolkit
redis>=5.0                  # Caching

# Utilities
python-dateutil>=2.9        # Date utilities
tqdm>=4.66                  # Progress bars
tenacity>=9.0               # Retry logic
loguru>=0.7.0               # Logging
pyyaml>=6.0                 # YAML config
python-dotenv>=1.0          # Environment variables

# Testing
pytest>=8.3                 # Testing framework
pytest-cov>=5.0             # Coverage
pytest-asyncio>=0.21.0      # Async tests
pytest-mock>=3.12           # Mocking
httpx>=0.25                 # Async HTTP client for API tests

# Development
black>=23.11                # Code formatter
flake8>=6.1                 # Linter
mypy>=1.7                   # Type checker
pre-commit>=3.5             # Git hooks
```

---

## Phase 2 : Plan d'Implémentation Séquentiel

### Approche : Développement Incrémental par Priorité

**Principe** : Chaque module est développé, testé, documenté et intégré AVANT de passer au suivant.

### 2.1 Priorité P0 - Fondations Critiques (48h)

#### Module 1 : utils/ - Utilitaires Core (4h)
**Objectif** : Fondations communes à tous les modules

**Livrables** :
- [ ] `types.py` - TypedDict (OHLCV, Universe, Params, Result)
- [ ] `io.py` - Lecture/écriture parquet/csv avec validation
- [ ] `timeframes.py` - Conversions (1m, 5m, 1h, 1D, 1W, 1M)
- [ ] `registry.py` - Registre dynamique (indicateurs, patterns, scans)
- [ ] `logging_config.py` - Configuration logging unifiée

**Tests** :
- Test lecture/écriture parquet
- Test conversions timeframes
- Test registry (register, lookup, list)

**Validation** :
- ✅ Tous les tests passent
- ✅ Type hints corrects
- ✅ Documentation complète

---

#### Module 2 : data/ - Ingestion Unifiée (8h)
**Objectif** : Source unique pour toutes les données

**Livrables** :
- [ ] `loaders.py` - get_ohlcv() unifié (yfinance, CCXT, CSV, parquet)
- [ ] `fred.py` - get_series() pour indicateurs macro
- [ ] `normalizers.py` - Normalisation colonnes (Open/open → standard)
- [ ] `cache.py` - Cache parquet intelligent avec TTL

**Intégrations** :
- Fusionner `src/data_sources/crypto_data.py`
- Fusionner `src/infrastructure/arctic_manager.py`
- Fusionner `src/infrastructure/data_manager.py`

**Tests** :
- Test chargement yfinance (AAPL, SPY)
- Test chargement CCXT (BTC/USDT)
- Test FRED (CPI, GDP)
- Test cache (hit/miss)
- Test normalisation colonnes

**Validation** :
- ✅ Charge AAPL 1Y en < 2s
- ✅ Cache fonctionne
- ✅ Toutes colonnes normalisées

---

#### Module 3 : indicators/ - Refactorisation Complète (16h)

**Objectif** : Couche d'indicateurs professionnelle avec TA-Lib/pandas-ta

**Sous-modules** :

**3.1 Base & Config (2h)**
- [ ] `base.py` - BaseIndicator abstract class
- [ ] `config.py` - Configuration globale (sources prioritaires)
- [ ] `exceptions.py` - IndicatorError, ValidationError
- [ ] `validators.py` - Validation DataFrame et paramètres

**3.2 Wrappers (8h)**
- [ ] `talib_wrapper.py` - Wrapper TA-Lib (EMA, SMA, MACD, RSI, BBANDS, ATR, ADX, STOCH, CCI, MOM, WILLR, ROC, etc.)
- [ ] `pandas_ta_wrapper.py` - Wrapper pandas-ta (SuperTrend, Ichimoku, Heikin Ashi, ZLMA, etc.)
- [ ] `ta_wrapper.py` - Wrapper ta library (VWAP, OBV, CMF, etc.)
- [ ] `smc_wrapper.py` - Wrapper smartmoneyconcepts (OB, FVG, BOS, CHOCH)
- [ ] `vectorbt_wrapper.py` - Wrapper VectorBT pour calculs multi-params

**3.3 Core & Factory (4h)**
- [ ] `core.py` - run_indicator(name, df, **params) dynamique
- [ ] `factory.py` - Création indicateurs depuis spec JSON
- [ ] `performance.py` - Benchmarking utils

**3.4 Refactorisation Stratégies (2h)**
- [ ] `strategies/ema_strategy.py` - Utiliser TALibWrapper
- [ ] `strategies/macd_strategy.py` - Utiliser TALibWrapper
- [ ] `strategies/rsi_strategy.py` - Utiliser TALibWrapper
- [ ] `strategies/bollinger_strategy.py` - Utiliser TALibWrapper
- [ ] `strategies/supertrend_strategy.py` - Utiliser PandasTAWrapper
- [ ] `strategies/ichimoku_strategy.py` - Utiliser PandasTAWrapper

**Tests** :
- Test chaque wrapper (10 indicateurs minimum)
- Test run_indicator() dynamique
- Test factory depuis JSON
- Test équivalence stratégies refactorisées vs originales
- Test performance (benchmark)
- Test fallback TA-Lib → pandas-ta

**Validation** :
- ✅ Tous indicateurs fonctionnels
- ✅ Stratégies donnent résultats identiques
- ✅ Performance >= ancienne version
- ✅ Fallback fonctionne si TA-Lib absent
- ✅ 141 tests existants passent toujours

---

#### Module 4 : api/ - API REST FastAPI (12h)

**Objectif** : Exposer toutes capacités via API REST

**Sous-modules** :

**4.1 Structure & Main (2h)**
- [ ] `main.py` - Application FastAPI
- [ ] `middleware.py` - CORS, rate limiting, logging
- [ ] `models/requests.py` - Pydantic request models
- [ ] `models/responses.py` - Pydantic response models

**4.2 Routes Core (4h)**
- [ ] `routes/data.py` - GET /data/ohlcv, /data/fred
- [ ] `routes/indicators.py` - POST /indicators/run
- [ ] `routes/backtest.py` - POST /backtest/run

**4.3 Routes Avancées (6h)**
- Routes pour tous les autres modules (implémentation progressive)

**Tests** :
- Test /health endpoint
- Test /data/ohlcv
- Test /indicators/run
- Test validation Pydantic
- Test rate limiting
- Test error handling

**Validation** :
- ✅ API démarre sans erreur
- ✅ Endpoints core fonctionnels
- ✅ Documentation OpenAPI générée

---

#### Module 5 : cli/ - CLI Unifié (8h)

**Objectif** : Interface ligne de commande complète

**Commandes** :
- [ ] `data fetch` - Télécharger données
- [ ] `indicators run` - Calculer indicateurs
- [ ] `backtest run` - Lancer backtest
- [ ] `scan run` - Lancer scanner
- [ ] `plot show` - Afficher graphiques

**Tests** :
- Test chaque commande
- Test arguments parsing
- Test error handling

**Validation** :
- ✅ Toutes commandes fonctionnelles
- ✅ Help messages clairs
- ✅ Rich output

---

### 2.2 Priorité P1 - Modules Avancés (60h)

#### Module 6 : candlesticks/ - Motifs Chandeliers (8h)
- [ ] Wrapper TA-Lib 60+ patterns
- [ ] Fallback pandas-ta
- [ ] Détection unifiée
- [ ] Tests 10+ patterns

#### Module 7 : patterns/ - Chart Patterns (12h)
- [ ] Détection triangles (asc, desc, sym)
- [ ] Détection ETE/ETE inversée
- [ ] Détection double top/bottom
- [ ] Détection canaux
- [ ] Calculs géométriques
- [ ] Dataclasses pour patterns
- [ ] Tests chaque pattern

#### Module 8 : trendlines/ - Détection Auto (10h)
- [ ] Algorithme détection S/R (Hough transform)
- [ ] Calcul pentes, ordonnées
- [ ] Détection touches
- [ ] Détection cassures (breakout)
- [ ] Tests avec données réelles

#### Module 9 : vwap/ - Anchored VWAP (6h)
- [ ] VWAP avec ancres multiples
- [ ] Ancres prédéfinies (session, week, month, swing)
- [ ] Ancres custom (timestamp, callable)
- [ ] Tests chaque type d'ancre

#### Module 10 : zones/ - Supply/Demand (8h)
- [ ] Détection base (consolidation)
- [ ] Détection impulsion
- [ ] Projection zones
- [ ] Merge zones proches
- [ ] Scoring (force, fraîcheur)
- [ ] Tests détection

#### Module 11 : sentiment/ - Analyse Sentiment (12h)
- [ ] News sentiment (NewsAPI, Finnhub)
- [ ] Social sentiment (Twitter, Reddit)
- [ ] Fear & Greed Index
- [ ] NLP processing (FinBERT)
- [ ] Agrégation scores
- [ ] Tests chaque source

#### Module 12 : scanner/ - DSL Engine (16h)
- [ ] Parser DSL JSON → AST
- [ ] Moteur exécution parallèle
- [ ] Opérateurs (>, <, ==, crosses, etc.)
- [ ] Intégration tous modules
- [ ] Exemples scans
- [ ] Tests scans complexes

---

### 2.3 Priorité P2 - Modules Complémentaires (32h)

#### Module 13 : breadth/ - Market Breadth (6h)
- [ ] % above SMA
- [ ] A/D line
- [ ] McClellan Oscillator
- [ ] Support univers multi-symboles
- [ ] Tests

#### Module 14 : relativereturns/ - Force Relative (8h)
- [ ] RS ratio (target/bench)
- [ ] RS matrix (tournoi round-robin)
- [ ] Ranking
- [ ] Tests

#### Module 15 : fundamentals/ - Analyse Fondamentale (6h)
- [ ] Wrapper FinanceToolkit
- [ ] Ratios (PE, ROE, margins, etc.)
- [ ] États financiers
- [ ] Tests

#### Module 16 : economics/ - Indicateurs Macro (4h)
- [ ] Wrapper FRED
- [ ] CPI, GDP, Unemployment, Rates
- [ ] Cache local
- [ ] Tests

#### Module 17 : raindrop/ - Volume Profile Candles (4h)
- [ ] Bougies avec profil volume
- [ ] Plotly rendering
- [ ] Tests figure générée

---

### 2.4 Tests & Documentation (16h)

#### Tests Complets (10h)
- [ ] Coverage > 90% pour tous nouveaux modules
- [ ] Tests intégration API complets
- [ ] Tests intégration CLI complets
- [ ] Tests backtest end-to-end
- [ ] Tests scanner end-to-end

#### Documentation (6h)
- [ ] ARCHITECTURE.md détaillé
- [ ] API_REFERENCE.md complet
- [ ] CLI_REFERENCE.md complet
- [ ] SCANNER_DSL.md avec grammaire complète
- [ ] INDICATORS_CATALOG.md (200+ indicateurs)
- [ ] MIGRATION_GUIDE.md
- [ ] README.md refonte complète
- [ ] 10 notebooks d'exemples

---

## Phase 3 : Timeline Réaliste

### Vue d'Ensemble

| Phase | Modules | Heures | Jours (8h/j) | Calendrier |
|-------|---------|--------|--------------|------------|
| **P0 - Fondations** | utils, data, indicators, api, cli | 48h | 6j | Jours 1-6 |
| **P1 - Avancés** | candlesticks, patterns, trendlines, vwap, zones, sentiment, scanner | 72h | 9j | Jours 7-15 |
| **P2 - Complémentaires** | breadth, rs, fundamentals, economics, raindrop | 28h | 3.5j | Jours 16-19 |
| **P3 - Tests & Docs** | Tests exhaustifs, documentation | 16h | 2j | Jours 20-21 |
| **Buffer & Intégration** | Bugs, raffinements, intégration finale | 16h | 2j | Jours 22-23 |
| **TOTAL** | 18 modules complets | **180h** | **22.5j** | **3-4 semaines** |

### Planning Détaillé

**Semaine 1 (Jours 1-5)** : Fondations P0
- Jour 1 : utils/ + data/
- Jour 2-4 : indicators/ (refactorisation complète)
- Jour 5 : api/ (structure + routes core)

**Semaine 2 (Jours 6-10)** : CLI + Modules Avancés P1
- Jour 6 : cli/ + candlesticks/
- Jour 7-8 : patterns/ + trendlines/
- Jour 9 : vwap/ + zones/
- Jour 10 : sentiment/ (partie 1)

**Semaine 3 (Jours 11-15)** : Fin P1 + Début P2
- Jour 11-12 : sentiment/ (partie 2) + scanner/ (partie 1)
- Jour 13-14 : scanner/ (partie 2)
- Jour 15 : breadth/ + relativereturns/

**Semaine 4 (Jours 16-21)** : Fin P2 + Tests & Docs
- Jour 16 : fundamentals/ + economics/
- Jour 17 : raindrop/ + intégrations finales
- Jour 18-19 : Tests exhaustifs
- Jour 20-21 : Documentation complète

**Jours 22-23** : Buffer & polissage
- Corrections bugs
- Raffinements
- Intégration finale
- Validation production

---

## Phase 4 : Spécifications Techniques Détaillées

### 4.1 Module sentiment/ - Analyse de Sentiment

**Sources de Données** :

1. **News APIs**
   - NewsAPI (newsapi.org) - 150+ sources, 80k articles/jour
   - Finnhub (finnhub.io) - News + sentiment scores
   - Alpha Vantage - News sentiment

2. **Social Media**
   - Twitter/X - snscrape (pas d'API key requis) ou Twitter API v2
   - Reddit - PRAW (Reddit API)
   - StockTwits - StockTwits API

3. **Indices de Sentiment**
   - CNN Fear & Greed Index
   - CBOE VIX (volatility index)
   - Put/Call Ratio

4. **NLP Models**
   - FinBERT (transformers) - Sentiment financier
   - VADER - Sentiment général
   - Custom models fine-tunés

**Architecture** :

```python
# src/sentiment/news.py
class NewsAnalyzer:
    def get_news(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Récupère news avec scores sentiment"""
        # Colonnes: timestamp, headline, source, url, sentiment_score, relevance

    def aggregate_sentiment(news_df: pd.DataFrame, freq: str = '1D') -> pd.Series:
        """Agrège sentiment par période"""
        # Weighted average par relevance

# src/sentiment/social.py
class SocialAnalyzer:
    def get_twitter_sentiment(symbol: str, lookback: str = '7D') -> pd.Series:
        """Scrape Twitter, analyse sentiment"""

    def get_reddit_sentiment(symbol: str, subreddits: list[str]) -> pd.Series:
        """Analyse Reddit posts/comments"""

# src/sentiment/fear_greed.py
def get_fear_greed_index(start: datetime, end: datetime) -> pd.Series:
    """CNN Fear & Greed Index (0-100)"""

# src/sentiment/nlp.py
class SentimentNLP:
    def __init__(self, model: str = 'finbert'):
        """Initialize NLP model (FinBERT, VADER, etc.)"""

    def analyze_text(text: str) -> dict:
        """Retourne {label: 'positive/negative/neutral', score: 0-1}"""

# src/sentiment/aggregator.py
def aggregate_all_sentiment(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Agrège toutes sources de sentiment"""
    # Colonnes: news_sentiment, twitter_sentiment, reddit_sentiment, fear_greed, composite_score
```

**Scoring Composite** :
```python
composite_score = (
    0.4 * news_sentiment +
    0.3 * social_sentiment +
    0.2 * fear_greed +
    0.1 * put_call_ratio
)
```

**Tests** :
- Test chaque source individuellement
- Test agrégation
- Test NLP models
- Mock APIs pour tests unitaires

---

### 4.2 Module scanner/ - DSL Engine

**Grammaire DSL JSON** :

```json
{
  "name": "Golden Cross + Sentiment Bullish",
  "universe": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
  "timeframe": "1D",
  "lookback": "1Y",
  "conditions": [
    {
      "type": "indicator_cross",
      "indicator1": {"name": "SMA", "params": {"length": 50}},
      "indicator2": {"name": "SMA", "params": {"length": 200}},
      "direction": "above",
      "within_bars": 5
    },
    {
      "type": "indicator_compare",
      "indicator": {"name": "RSI", "params": {"length": 14}},
      "operator": ">",
      "value": 50
    },
    {
      "type": "sentiment",
      "source": "composite",
      "operator": ">",
      "value": 0.6,
      "lookback": "7D"
    },
    {
      "type": "pattern",
      "name": "CDLHAMMER",
      "within_bars": 3
    },
    {
      "type": "trendline_breakout",
      "lookback": 250,
      "touches_min": 3,
      "direction": "bullish"
    },
    {
      "type": "volume",
      "operator": ">",
      "value": {"indicator": "SMA", "params": {"length": 20, "field": "volume"}},
      "multiplier": 1.5
    }
  ],
  "output": {
    "fields": ["close", "volume", "SMA_50", "SMA_200", "RSI_14", "sentiment_score"],
    "explain": true
  }
}
```

**Architecture** :

```python
# src/scanner/dsl.py
from pydantic import BaseModel
from typing import Literal, Union, Any

class IndicatorSpec(BaseModel):
    name: str
    params: dict[str, Any]

class ConditionIndicatorCross(BaseModel):
    type: Literal["indicator_cross"]
    indicator1: IndicatorSpec
    indicator2: IndicatorSpec
    direction: Literal["above", "below"]
    within_bars: int = 1

class ConditionIndicatorCompare(BaseModel):
    type: Literal["indicator_compare"]
    indicator: IndicatorSpec
    operator: Literal[">", "<", ">=", "<=", "==", "!="]
    value: Union[float, dict]  # dict si comparaison avec indicateur

class ConditionSentiment(BaseModel):
    type: Literal["sentiment"]
    source: Literal["news", "social", "composite"]
    operator: str
    value: float
    lookback: str

class ConditionPattern(BaseModel):
    type: Literal["pattern"]
    name: str
    within_bars: int = 1

class ConditionTrendlineBreakout(BaseModel):
    type: Literal["trendline_breakout"]
    lookback: int
    touches_min: int
    direction: Literal["bullish", "bearish"]

ScanCondition = Union[
    ConditionIndicatorCross,
    ConditionIndicatorCompare,
    ConditionSentiment,
    ConditionPattern,
    ConditionTrendlineBreakout,
]

class ScanSpec(BaseModel):
    name: str
    universe: list[str]
    timeframe: str
    lookback: str = "1Y"
    conditions: list[ScanCondition]
    output: dict

# src/scanner/engine.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.data.loaders import get_ohlcv
from src.indicators.core import run_indicator
from src.sentiment.aggregator import aggregate_all_sentiment

class ScanEngine:
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers

    def run_scan(self, spec: ScanSpec) -> pd.DataFrame:
        """Execute scan en parallèle sur univers"""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._scan_symbol, symbol, spec): symbol
                for symbol in spec.universe
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")

        return pd.DataFrame(results)

    def _scan_symbol(self, symbol: str, spec: ScanSpec) -> Optional[dict]:
        """Scan un symbole"""
        # 1. Charger données
        df = get_ohlcv(symbol, lookback=spec.lookback, timeframe=spec.timeframe)

        # 2. Évaluer chaque condition
        conditions_met = []
        for condition in spec.conditions:
            met, explanation = self._evaluate_condition(df, symbol, condition)
            conditions_met.append(met)
            if not met:
                return None  # Toutes conditions doivent être vraies

        # 3. Si toutes conditions vraies, retourner résultat
        return {
            "symbol": symbol,
            "matched_at": df.index[-1],
            "conditions_met": len(spec.conditions),
            **self._extract_output_fields(df, spec.output)
        }

    def _evaluate_condition(self, df: pd.DataFrame, symbol: str, condition: ScanCondition) -> tuple[bool, str]:
        """Évalue une condition"""
        if condition.type == "indicator_cross":
            return self._eval_indicator_cross(df, condition)
        elif condition.type == "indicator_compare":
            return self._eval_indicator_compare(df, condition)
        elif condition.type == "sentiment":
            return self._eval_sentiment(symbol, condition)
        elif condition.type == "pattern":
            return self._eval_pattern(df, condition)
        elif condition.type == "trendline_breakout":
            return self._eval_trendline_breakout(df, condition)
        else:
            raise ValueError(f"Unknown condition type: {condition.type}")
```

**Tests** :
- Test parsing DSL JSON
- Test chaque type de condition
- Test scan complet
- Test parallélisation
- Test gestion erreurs

---

### 4.3 API Endpoints Détaillés

**Endpoints Complets** :

```python
# src/api/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI(
    title="TradingSystemStack API",
    description="API Institutionnelle de Trading - 18 Modules",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
from src.api.routes import (
    data, indicators, candlesticks, patterns, trendlines,
    vwap, raindrop, breadth, rs, zones, fundamentals,
    economics, sentiment, scanner, backtest
)

app.include_router(data.router, prefix="/data", tags=["data"])
app.include_router(indicators.router, prefix="/indicators", tags=["indicators"])
app.include_router(candlesticks.router, prefix="/candlesticks", tags=["candlesticks"])
app.include_router(patterns.router, prefix="/patterns", tags=["patterns"])
app.include_router(trendlines.router, prefix="/trendlines", tags=["trendlines"])
app.include_router(vwap.router, prefix="/vwap", tags=["vwap"])
app.include_router(raindrop.router, prefix="/raindrop", tags=["raindrop"])
app.include_router(breadth.router, prefix="/breadth", tags=["breadth"])
app.include_router(rs.router, prefix="/rs", tags=["relativereturns"])
app.include_router(zones.router, prefix="/zones", tags=["zones"])
app.include_router(fundamentals.router, prefix="/fundamentals", tags=["fundamentals"])
app.include_router(economics.router, prefix="/economics", tags=["economics"])
app.include_router(sentiment.router, prefix="/sentiment", tags=["sentiment"])
app.include_router(scanner.router, prefix="/scanner", tags=["scanner"])
app.include_router(backtest.router, prefix="/backtest", tags=["backtest"])

@app.get("/")
def root():
    return {"message": "TradingSystemStack API v2.0", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}
```

**Exemple Route** :

```python
# src/api/routes/sentiment.py
from fastapi import APIRouter, Query
from datetime import datetime
from src.api.models.requests import SentimentRequest
from src.api.models.responses import SentimentResponse
from src.sentiment.aggregator import aggregate_all_sentiment

router = APIRouter()

@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyse sentiment multi-sources pour un symbole"""
    try:
        sentiment_df = aggregate_all_sentiment(
            symbol=request.symbol,
            start=request.start,
            end=request.end
        )

        return SentimentResponse(
            symbol=request.symbol,
            start=request.start,
            end=request.end,
            data=sentiment_df.to_dict(orient='records'),
            sources={
                "news": True,
                "social": True,
                "fear_greed": True
            },
            composite_score=sentiment_df['composite_score'].iloc[-1]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fear-greed")
async def get_fear_greed(
    start: datetime = Query(...),
    end: datetime = Query(...)
):
    """Récupère Fear & Greed Index"""
    from src.sentiment.fear_greed import get_fear_greed_index
    series = get_fear_greed_index(start, end)
    return {"data": series.to_dict()}
```

---

## Phase 5 : Gestion des Risques et Qualité

### 5.1 Risques Identifiés

| Risque | Impact | Probabilité | Mitigation |
|--------|--------|-------------|------------|
| **Scope trop large** | Critique | Élevée | Plan séquentiel avec priorités P0/P1/P2 |
| **Bugs dans intégration** | Élevé | Moyenne | Tests exhaustifs à chaque module |
| **Performance dégradée** | Élevé | Moyenne | Benchmarks systématiques |
| **APIs externes down** | Moyen | Moyenne | Cache + fallbacks + retry logic |
| **TA-Lib installation fail** | Moyen | Moyenne | Fallback pandas-ta documenté |
| **Dépassement timeline** | Moyen | Élevée | Buffer 2 jours + ajustements |
| **Breaking changes existant** | Critique | Faible | Tests régression 141 tests |

### 5.2 Standards de Qualité

**Code Quality** :
- ✅ PEP8 (black, flake8)
- ✅ Type hints partout
- ✅ Docstrings Google format
- ✅ Logging structuré
- ✅ Error handling robuste

**Testing** :
- ✅ Coverage > 90% pour nouveaux modules
- ✅ Tests unitaires pour chaque fonction core
- ✅ Tests intégration pour chaque module
- ✅ Tests API pour chaque endpoint
- ✅ Tests CLI pour chaque commande
- ✅ Tests régression (141 tests existants)

**Documentation** :
- ✅ Docstrings complètes
- ✅ README avec quick start
- ✅ Architecture docs
- ✅ API reference
- ✅ CLI reference
- ✅ Examples notebooks

**Performance** :
- ✅ Benchmarks avant/après
- ✅ Pas de régression > 10%
- ✅ Vectorisation prioritaire
- ✅ Caching intelligent

---

## Phase 6 : Livrables Finaux

### 6.1 Checklist Complète

**Code** :
- [ ] 18 modules fonctionnels
- [ ] 6 stratégies refactorisées
- [ ] API REST complète (15+ endpoints)
- [ ] CLI complet (10+ commandes)
- [ ] Dashboard Streamlit unifié
- [ ] 200+ tests (coverage > 90%)

**Documentation** :
- [ ] README.md complet
- [ ] ARCHITECTURE.md détaillé
- [ ] API_REFERENCE.md
- [ ] CLI_REFERENCE.md
- [ ] SCANNER_DSL.md
- [ ] INDICATORS_CATALOG.md
- [ ] MIGRATION_GUIDE.md
- [ ] 10 notebooks d'exemples

**Configuration** :
- [ ] requirements.txt unifié
- [ ] pyproject.toml à jour
- [ ] Makefile pour installation
- [ ] .env.example
- [ ] Docker (optionnel)

**Tests & Validation** :
- [ ] Tous tests passent
- [ ] Coverage > 90%
- [ ] Performance validée
- [ ] API démarrée et testée
- [ ] CLI testé
- [ ] Notebooks exécutés

---

## Phase 7 : Post-Déploiement

### 7.1 Monitoring

- [ ] Logging centralisé
- [ ] Métriques API (requêtes, latence, erreurs)
- [ ] Alertes sur erreurs critiques
- [ ] Dashboard monitoring

### 7.2 Maintenance

- [ ] Plan de updates réguliers
- [ ] Veille nouvelles bibliothèques
- [ ] Feedback utilisateurs
- [ ] Roadmap futures fonctionnalités

---

## Conclusion

### Résumé Exécutif

**Plateforme Cible** : TradingSystemStack v2.0 - Plateforme Institutionnelle Complète

**Modules** : 18 modules professionnels (12 nouveaux + 6 refactorisés)

**Timeline** : 22-23 jours de travail rigoureux (3-4 semaines calendaires)

**Effort** : ~180 heures de développement

**Bénéfices** :
1. ✅ **Précision** - Indicateurs basés sur TA-Lib/pandas-ta (standard industrie)
2. ✅ **Performance** - Calculs optimisés (C, Numba, vectorisation)
3. ✅ **Complétude** - 200+ indicateurs, patterns, sentiment, scanner, API, CLI
4. ✅ **Production-Ready** - Tests exhaustifs, documentation complète
5. ✅ **Maintenabilité** - Architecture modulaire, code propre
6. ✅ **Extensibilité** - Registry dynamique, plugins futurs

### Prochaines Étapes

1. **Validation du plan** par vous
2. **Démarrage Phase 0** - Module utils/ (4h)
3. **Développement séquentiel** selon priorités
4. **Validation checkpoints** après chaque module
5. **Livraison finale** après 22-23 jours

---

**PRÊT À EXÉCUTER AVEC RIGUEUR ABSOLUE.**

Ce plan sera le guide unique pour les 3-4 prochaines semaines.
Chaque module sera développé, testé, documenté et validé avant de passer au suivant.

**AUCUN RACCOURCI. QUALITÉ PRODUCTION.**
