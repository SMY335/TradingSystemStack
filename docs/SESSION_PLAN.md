# Plan d'Exécution par Sessions - TradingSystemStack v2.0

**Date**: 2025-11-07
**Contrainte**: 200k tokens maximum par session
**Objectif**: Découpage en 4 sessions autonomes et complètes

---

## Vue d'Ensemble

### Budget Tokens par Session

| Composant | Tokens Estimés |
|-----------|----------------|
| Contexte initial session | 10k |
| Lecture fichiers existants | 5-10k |
| Création module complet | 8-15k |
| Tests module | 3-5k |
| Documentation | 2-3k |
| Commits/validations | 1-2k |
| Marge erreurs/refactoring | 20-30k |
| **Total utilisable par session** | **~160k tokens** |

### Modules par Session

**Session 1**: Fondations Critiques (5 modules P0) - ~70k tokens
**Session 2**: Indicateurs + API + CLI (3 modules P0) - ~80k tokens
**Session 3**: Modules Avancés P1 (6 modules) - ~90k tokens
**Session 4**: Modules P2 + Tests + Docs (4 modules) - ~70k tokens

**TOTAL: 4 sessions pour 18 modules complets**

---

## SESSION 1 - Fondations Critiques (5 modules)

### Objectif
Créer les fondations nécessaires pour tous les autres modules.

### Budget Tokens: ~70k / 200k

### Durée Estimée
8 heures de travail (temps développement, pas temps réel)

### Modules à Compléter

#### 1.1 Module utils/ (8k tokens)

**Fichiers à créer**:
- [ ] `src/utils/__init__.py`
- [ ] `src/utils/types.py` - TypedDict (OHLCV, Universe, Params, Result)
- [ ] `src/utils/io.py` - Lecture/écriture parquet/csv
- [ ] `src/utils/timeframes.py` - Conversions timeframes
- [ ] `src/utils/registry.py` - Registre dynamique
- [ ] `src/utils/logging_config.py` - Configuration logging

**Tests**:
- [ ] `tests/unit/test_utils_io.py` - Tests lecture/écriture
- [ ] `tests/unit/test_utils_timeframes.py` - Tests conversions
- [ ] `tests/unit/test_utils_registry.py` - Tests registry

**Documentation**:
- [ ] Docstrings complètes
- [ ] Exemples d'usage dans docstrings

**Validation**:
- ✅ Tous tests passent
- ✅ Coverage > 90%
- ✅ Type hints corrects

---

#### 1.2 Module data/ (12k tokens)

**Fichiers à créer**:
- [ ] `src/data/__init__.py`
- [ ] `src/data/loaders.py` - get_ohlcv() unifié (yfinance, CSV, parquet)
- [ ] `src/data/fred.py` - get_series() pour FRED
- [ ] `src/data/normalizers.py` - Normalisation colonnes
- [ ] `src/data/cache.py` - Cache parquet avec TTL

**Fichiers à intégrer/refactoriser**:
- [ ] Fusionner `src/data_sources/crypto_data.py` → `src/data/loaders.py`
- [ ] Fusionner `src/infrastructure/arctic_manager.py` → `src/data/cache.py`
- [ ] Fusionner `src/infrastructure/data_manager.py` → `src/data/loaders.py`

**Tests**:
- [ ] `tests/unit/test_data_loaders.py` - Tests yfinance, CSV, parquet
- [ ] `tests/unit/test_data_fred.py` - Tests FRED
- [ ] `tests/unit/test_data_cache.py` - Tests cache
- [ ] `tests/unit/test_data_normalizers.py` - Tests normalisation

**Validation**:
- ✅ Charge AAPL 1Y < 2s
- ✅ Cache fonctionne (hit/miss)
- ✅ Colonnes normalisées
- ✅ FRED functional

---

#### 1.3 Module candlesticks/ (10k tokens)

**Fichiers à créer**:
- [ ] `src/candlesticks/__init__.py`
- [ ] `src/candlesticks/talib_patterns.py` - 60+ patterns TA-Lib
- [ ] `src/candlesticks/pandas_ta_patterns.py` - Fallback pandas-ta
- [ ] `src/candlesticks/detector.py` - Détection unifiée

**Tests**:
- [ ] `tests/unit/test_candlesticks.py` - 10+ patterns testés

**Validation**:
- ✅ TA-Lib patterns fonctionnels (si installé)
- ✅ Fallback pandas-ta fonctionne
- ✅ Détection unifiée correcte

---

#### 1.4 Module vwap/ (8k tokens)

**Fichiers à créer**:
- [ ] `src/vwap/__init__.py`
- [ ] `src/vwap/anchored_vwap.py` - VWAP avec ancres multiples
- [ ] `src/vwap/anchors.py` - Définitions ancres prédéfinies

**Tests**:
- [ ] `tests/unit/test_vwap.py` - Tests chaque type ancre

**Validation**:
- ✅ Ancres prédéfinies fonctionnelles (session, week, month, swing)
- ✅ Ancre custom (timestamp, callable)
- ✅ Calculs corrects

---

#### 1.5 Module zones/ (10k tokens)

**Fichiers à créer**:
- [ ] `src/zones/__init__.py`
- [ ] `src/zones/supply_demand.py` - Détection zones SD
- [ ] `src/zones/merger.py` - Fusion zones proches

**Tests**:
- [ ] `tests/unit/test_zones.py` - Tests détection

**Validation**:
- ✅ Détection base (consolidation)
- ✅ Détection impulsion
- ✅ Projection zones
- ✅ Merge zones
- ✅ Scoring (force, fraîcheur)

---

### Livrables Session 1

**Code**:
- ✅ 5 modules complets et fonctionnels
- ✅ ~2000 lignes de code production
- ✅ ~1000 lignes de tests

**Tests**:
- ✅ ~50 tests unitaires
- ✅ Coverage > 90% pour nouveaux modules
- ✅ 141 tests existants passent toujours

**Documentation**:
- ✅ Docstrings Google format
- ✅ Type hints partout
- ✅ Exemples dans docstrings

**Git**:
- ✅ Commit après chaque module
- ✅ Push à la fin de session
- ✅ Branche: `claude/session-1-foundations-[ID]`

**État à la fin**:
- ✅ Fondations prêtes pour autres modules
- ✅ Data loading fonctionnel
- ✅ 3 modules analytiques fonctionnels (candlesticks, vwap, zones)

---

### Checkpoint Session 1

**Vous devez valider**:
1. Tests: `pytest tests/unit/test_utils* tests/unit/test_data* tests/unit/test_candlesticks* tests/unit/test_vwap* tests/unit/test_zones* -v`
2. Imports: `python -c "from src.utils import types, io, timeframes, registry; from src.data import loaders, fred; from src.candlesticks import detector; from src.vwap import anchored_vwap; from src.zones import supply_demand; print('OK')"`
3. Coverage: `pytest --cov=src/utils --cov=src/data --cov=src/candlesticks --cov=src/vwap --cov=src/zones`

**Puis vous dites**: "Session 1 validée - GO Session 2"

---

## SESSION 2 - Indicateurs + API + CLI (3 modules)

### Objectif
Refactoriser indicateurs + Créer API REST + CLI (modules les plus critiques)

### Budget Tokens: ~80k / 200k

### Durée Estimée
12 heures de travail

### Modules à Compléter

#### 2.1 Module indicators/ - REFACTORISATION COMPLÈTE (35k tokens)

**C'est le module le PLUS important et complexe**

**Fichiers à créer**:
- [ ] `src/indicators/__init__.py`
- [ ] `src/indicators/base.py` - BaseIndicator abstract class
- [ ] `src/indicators/config.py` - Configuration globale
- [ ] `src/indicators/exceptions.py` - Exceptions
- [ ] `src/indicators/validators.py` - Validation DataFrame/params
- [ ] `src/indicators/core.py` - run_indicator() dynamique
- [ ] `src/indicators/factory.py` - Création depuis spec JSON
- [ ] `src/indicators/performance.py` - Benchmarking

**Wrappers**:
- [ ] `src/indicators/wrappers/__init__.py`
- [ ] `src/indicators/wrappers/talib_wrapper.py` - TA-Lib (20+ indicateurs)
- [ ] `src/indicators/wrappers/pandas_ta_wrapper.py` - pandas-ta (15+ indicateurs)
- [ ] `src/indicators/wrappers/ta_wrapper.py` - ta library (VWAP, volume)
- [ ] `src/indicators/wrappers/smc_wrapper.py` - smartmoneyconcepts
- [ ] `src/indicators/wrappers/vectorbt_wrapper.py` - VectorBT

**Refactorisation Stratégies** (CRITIQUE):
- [ ] `src/strategies/ema_strategy.py` - Utiliser TALibWrapper au lieu de pandas
- [ ] `src/strategies/macd_strategy.py` - Utiliser TALibWrapper
- [ ] `src/strategies/rsi_strategy.py` - Utiliser TALibWrapper
- [ ] `src/strategies/bollinger_strategy.py` - Utiliser TALibWrapper
- [ ] `src/strategies/supertrend_strategy.py` - Utiliser PandasTAWrapper
- [ ] `src/strategies/ichimoku_strategy.py` - Utiliser PandasTAWrapper

**Tests**:
- [ ] `tests/unit/test_indicators_talib.py` - Tests TA-Lib wrapper
- [ ] `tests/unit/test_indicators_pandas_ta.py` - Tests pandas-ta wrapper
- [ ] `tests/unit/test_indicators_core.py` - Tests run_indicator()
- [ ] `tests/unit/test_indicators_factory.py` - Tests factory
- [ ] `tests/unit/test_strategies_refactored.py` - Tests ÉQUIVALENCE stratégies

**Validation CRITIQUE**:
- ✅ Tous wrappers fonctionnels
- ✅ run_indicator() dynamique fonctionne
- ✅ **ÉQUIVALENCE**: Stratégies refactorisées donnent MÊMES résultats
- ✅ **PERFORMANCE**: >= ancienne version (benchmark)
- ✅ **FALLBACK**: Si TA-Lib absent, utilise pandas-ta
- ✅ **RÉGRESSION**: 141 tests existants passent TOUJOURS

---

#### 2.2 Module api/ - API REST FastAPI (25k tokens)

**Fichiers à créer**:
- [ ] `src/api/__init__.py`
- [ ] `src/api/main.py` - Application FastAPI
- [ ] `src/api/middleware.py` - CORS, rate limiting, logging

**Pydantic Models**:
- [ ] `src/api/models/__init__.py`
- [ ] `src/api/models/requests.py` - Request models
- [ ] `src/api/models/responses.py` - Response models

**Routes (Phase 1 - Core)**:
- [ ] `src/api/routes/__init__.py`
- [ ] `src/api/routes/data.py` - GET /data/ohlcv, /data/fred
- [ ] `src/api/routes/indicators.py` - POST /indicators/run
- [ ] `src/api/routes/candlesticks.py` - POST /candlesticks/detect
- [ ] `src/api/routes/vwap.py` - POST /vwap/anchored
- [ ] `src/api/routes/zones.py` - POST /zones/supply_demand

**Tests**:
- [ ] `tests/integration/test_api_smoke.py` - Tests /health
- [ ] `tests/integration/test_api_data.py` - Tests /data/*
- [ ] `tests/integration/test_api_indicators.py` - Tests /indicators/run
- [ ] `tests/integration/test_api_candlesticks.py` - Tests /candlesticks/detect

**Validation**:
- ✅ API démarre: `uvicorn src.api.main:app --reload`
- ✅ OpenAPI docs: http://localhost:8000/docs
- ✅ /health retourne 200
- ✅ /data/ohlcv fonctionne
- ✅ /indicators/run fonctionne
- ✅ Rate limiting fonctionne

---

#### 2.3 Module cli/ - CLI Unifié (20k tokens)

**Fichiers à créer**:
- [ ] `src/cli.py` - CLI principal Typer
- [ ] `src/cli/__init__.py`
- [ ] `src/cli/commands/__init__.py`
- [ ] `src/cli/commands/data.py` - Commandes data fetch
- [ ] `src/cli/commands/indicators.py` - Commandes indicators run
- [ ] `src/cli/commands/backtest.py` - Commandes backtest run
- [ ] `src/cli/commands/plot.py` - Commandes plot show

**Commandes à implémenter**:
```bash
# Data
python -m src.cli data fetch --symbol AAPL --start 2023-01-01 --end 2024-01-01

# Indicators
python -m src.cli indicators run --indicator RSI --symbol AAPL --params '{"length":14}'

# Candlesticks
python -m src.cli candlesticks detect --symbol AAPL --patterns HAMMER,DOJI

# Backtest (basic)
python -m src.cli backtest run --strategy ema_cross --symbols AAPL --start 2023-01-01
```

**Tests**:
- [ ] `tests/integration/test_cli_data.py` - Tests commandes data
- [ ] `tests/integration/test_cli_indicators.py` - Tests commandes indicators

**Validation**:
- ✅ Toutes commandes fonctionnelles
- ✅ Help messages clairs
- ✅ Rich output
- ✅ Error handling propre

---

### Livrables Session 2

**Code**:
- ✅ 3 modules critiques complets
- ✅ 6 stratégies refactorisées
- ✅ ~3500 lignes de code production
- ✅ API REST fonctionnelle
- ✅ CLI fonctionnel

**Tests**:
- ✅ ~80 tests (50 unitaires + 30 intégration)
- ✅ Coverage > 90%
- ✅ **CRITIQUE**: Tests équivalence stratégies passent
- ✅ **CRITIQUE**: 141 tests existants passent

**Documentation**:
- ✅ API OpenAPI auto-générée
- ✅ CLI --help complet
- ✅ README mis à jour avec exemples

**Git**:
- ✅ Commit après chaque module
- ✅ Push à la fin de session
- ✅ Branche: `claude/session-2-core-[ID]`

**État à la fin**:
- ✅ Indicateurs professionnels (TA-Lib/pandas-ta)
- ✅ Stratégies refactorisées et équivalentes
- ✅ API REST utilisable
- ✅ CLI utilisable

---

### Checkpoint Session 2

**Vous devez valider**:
1. **Tests Stratégies**: `pytest tests/unit/test_strategies_refactored.py -v` → DOIT PASSER
2. **Tests Régression**: `pytest tests/unit/test_*_validation.py -v` → 141 tests DOIVENT passer
3. **API**: Démarrer `uvicorn src.api.main:app --reload` → http://localhost:8000/docs
4. **CLI**: `python -m src.cli indicators run --indicator RSI --symbol AAPL --params '{"length":14}'`

**CRITIQUE**: Si tests stratégies ou régression échouent, **NE PAS CONTINUER**

**Puis vous dites**: "Session 2 validée - GO Session 3"

---

## SESSION 3 - Modules Avancés P1 (6 modules)

### Objectif
Créer modules analytiques avancés (patterns, trendlines, sentiment, breadth, RS)

### Budget Tokens: ~90k / 200k

### Durée Estimée
14 heures de travail

### Modules à Compléter

#### 3.1 Module patterns/ (15k tokens)

**Fichiers à créer**:
- [ ] `src/patterns/__init__.py`
- [ ] `src/patterns/chart_patterns.py` - Détection triangles, H&S, double tops
- [ ] `src/patterns/geometry.py` - Calculs géométriques
- [ ] `src/patterns/models.py` - Dataclasses pour patterns

**Tests**:
- [ ] `tests/unit/test_patterns.py` - Tests chaque pattern

**Validation**:
- ✅ Détection triangles (asc, desc, sym)
- ✅ Détection H&S (head and shoulders)
- ✅ Détection double top/bottom
- ✅ Score confiance > 0

---

#### 3.2 Module trendlines/ (15k tokens)

**Fichiers à créer**:
- [ ] `src/trendlines/__init__.py`
- [ ] `src/trendlines/detector.py` - Détection S/R auto
- [ ] `src/trendlines/breakout.py` - Détection cassures
- [ ] `src/trendlines/algorithms.py` - Hough transform, regression

**Tests**:
- [ ] `tests/unit/test_trendlines.py` - Tests détection + breakout

**Validation**:
- ✅ Détection supports/résistances
- ✅ Calcul pentes, ordonnées
- ✅ Détection touches
- ✅ Détection breakouts

---

#### 3.3 Module sentiment/ (20k tokens)

**Fichiers à créer**:
- [ ] `src/sentiment/__init__.py`
- [ ] `src/sentiment/news.py` - News sentiment (NewsAPI, Finnhub)
- [ ] `src/sentiment/social.py` - Twitter/Reddit
- [ ] `src/sentiment/fear_greed.py` - Fear & Greed Index
- [ ] `src/sentiment/nlp.py` - NLP processing (FinBERT)
- [ ] `src/sentiment/aggregator.py` - Agrégation scores

**Tests**:
- [ ] `tests/unit/test_sentiment.py` - Tests chaque source + agrégation

**Validation**:
- ✅ News sentiment fonctionnel
- ✅ Social sentiment fonctionnel (avec mocks)
- ✅ Fear & Greed Index récupéré
- ✅ Agrégation composite

---

#### 3.4 Module breadth/ (12k tokens)

**Fichiers à créer**:
- [ ] `src/breadth/__init__.py`
- [ ] `src/breadth/breadth_core.py` - % above SMA, A/D line, McClellan
- [ ] `src/breadth/universe.py` - Gestion univers multi-symboles

**Tests**:
- [ ] `tests/unit/test_breadth.py` - Tests chaque indicateur

**Validation**:
- ✅ % above SMA fonctionne
- ✅ A/D line calculée
- ✅ McClellan si données dispo

---

#### 3.5 Module relativereturns/ (12k tokens)

**Fichiers à créer**:
- [ ] `src/relativereturns/__init__.py`
- [ ] `src/relativereturns/rs_core.py` - RS ratio, RS matrix
- [ ] `src/relativereturns/ranking.py` - Classements RS

**Tests**:
- [ ] `tests/unit/test_rs.py` - Tests RS ratio + matrix

**Validation**:
- ✅ RS ratio (target/bench)
- ✅ RS matrix (round-robin)
- ✅ Ranking correct

---

#### 3.6 Module raindrop/ (8k tokens)

**Fichiers à créer**:
- [ ] `src/raindrop/__init__.py`
- [ ] `src/raindrop/raindrop_plotly.py` - Bougies avec profil volume

**Tests**:
- [ ] `tests/unit/test_raindrop.py` - Tests figure générée

**Validation**:
- ✅ Figure Plotly générée
- ✅ Profil volume up/down visible

---

### Livrables Session 3

**Code**:
- ✅ 6 modules avancés complets
- ✅ ~2500 lignes de code production

**Tests**:
- ✅ ~60 tests
- ✅ Coverage > 90%

**API Routes** (à ajouter):
- [ ] `src/api/routes/patterns.py`
- [ ] `src/api/routes/trendlines.py`
- [ ] `src/api/routes/sentiment.py`
- [ ] `src/api/routes/breadth.py`
- [ ] `src/api/routes/rs.py`
- [ ] `src/api/routes/raindrop.py`

**Git**:
- ✅ Commit après chaque module
- ✅ Push à la fin de session
- ✅ Branche: `claude/session-3-advanced-[ID]`

**État à la fin**:
- ✅ Patterns chartistes détectables
- ✅ Trendlines auto + breakouts
- ✅ Sentiment multi-sources
- ✅ Market breadth
- ✅ Relative strength analysis

---

### Checkpoint Session 3

**Vous devez valider**:
1. Tests: `pytest tests/unit/test_patterns* tests/unit/test_trendlines* tests/unit/test_sentiment* tests/unit/test_breadth* tests/unit/test_rs* tests/unit/test_raindrop* -v`
2. API: Tester nouveaux endpoints patterns, trendlines, sentiment
3. Imports: Vérifier tous modules importables

**Puis vous dites**: "Session 3 validée - GO Session 4"

---

## SESSION 4 - Finalisations (4 modules + Tests + Docs)

### Objectif
Modules P2 restants + Scanner + Tests exhaustifs + Documentation complète

### Budget Tokens: ~70k / 200k

### Durée Estimée
12 heures de travail

### Modules à Compléter

#### 4.1 Module fundamentals/ (10k tokens)

**Fichiers à créer**:
- [ ] `src/fundamentals/__init__.py`
- [ ] `src/fundamentals/fundamentals_ft.py` - FinanceToolkit wrapper
- [ ] `src/fundamentals/ratios.py` - Ratios clés

**Tests**:
- [ ] `tests/unit/test_fundamentals.py`

**Validation**:
- ✅ Ratios (PE, ROE, margins) récupérables
- ✅ États financiers accessibles

---

#### 4.2 Module economics/ (8k tokens)

**Fichiers à créer**:
- [ ] `src/economics/__init__.py`
- [ ] `src/economics/macro.py` - CPI, GDP, Unemployment, Rates
- [ ] `src/economics/fred_client.py` - Client FRED avec cache

**Tests**:
- [ ] `tests/unit/test_economics.py`

**Validation**:
- ✅ Indicateurs FRED récupérables
- ✅ Cache fonctionne

---

#### 4.3 Module scanner/ (25k tokens)

**LE MODULE LE PLUS COMPLEXE APRÈS indicators/**

**Fichiers à créer**:
- [ ] `src/scanner/__init__.py`
- [ ] `src/scanner/dsl.py` - Grammaire JSON (Pydantic models)
- [ ] `src/scanner/parser.py` - Parser DSL → AST
- [ ] `src/scanner/engine.py` - Exécution parallèle
- [ ] `src/scanner/operators.py` - Opérateurs (>, <, crosses, etc.)

**Exemples de scans**:
- [ ] `examples/scans/ema_golden_cross.json`
- [ ] `examples/scans/hammer_at_support.json`
- [ ] `examples/scans/rs_leaders.json`
- [ ] `examples/scans/supply_demand_zones.json`

**Tests**:
- [ ] `tests/unit/test_scanner_dsl.py` - Tests parsing DSL
- [ ] `tests/unit/test_scanner_operators.py` - Tests opérateurs
- [ ] `tests/integration/test_scanner_engine.py` - Tests scans complets

**Validation**:
- ✅ Parser DSL JSON fonctionne
- ✅ Exécution parallèle fonctionne
- ✅ 4 exemples de scans fonctionnels
- ✅ Toutes conditions supportées

---

#### 4.4 Tests Exhaustifs (15k tokens)

**Objectif**: Atteindre 90%+ coverage sur TOUS les modules

**Tests à créer/compléter**:
- [ ] `tests/integration/test_api_complete.py` - Tests API complets (tous endpoints)
- [ ] `tests/integration/test_cli_complete.py` - Tests CLI complets (toutes commandes)
- [ ] `tests/integration/test_backtest_strategies.py` - Tests backtest avec nouvelles stratégies
- [ ] `tests/integration/test_scanner_workflows.py` - Tests workflows scanner complets
- [ ] `tests/integration/test_end_to_end.py` - Test complet data → indicators → scanner → backtest

**Tests de régression**:
- [ ] Vérifier que les 141 tests existants passent TOUJOURS
- [ ] Vérifier équivalence stratégies refactorisées

**Coverage Report**:
- [ ] Générer rapport: `pytest --cov=src --cov-report=html --cov-report=term`
- [ ] Vérifier > 90% pour tous nouveaux modules

---

#### 4.5 Documentation Complète (12k tokens)

**Documents à créer/mettre à jour**:

**README.md** (complet):
- [ ] Getting Started
- [ ] Installation (avec TA-Lib)
- [ ] Quick Examples
- [ ] API endpoints
- [ ] CLI commands
- [ ] Scanner DSL
- [ ] Architecture overview

**Documentation Technique**:
- [ ] `docs/ARCHITECTURE.md` - Architecture détaillée
- [ ] `docs/API_REFERENCE.md` - Référence API complète
- [ ] `docs/CLI_REFERENCE.md` - Référence CLI complète
- [ ] `docs/SCANNER_DSL.md` - Grammaire DSL scanner
- [ ] `docs/INDICATORS_CATALOG.md` - Catalogue 200+ indicateurs
- [ ] `docs/MIGRATION_GUIDE.md` - Guide migration v1 → v2

**Exemples**:
- [ ] `examples/notebooks/01_indicators_demo.ipynb`
- [ ] `examples/notebooks/02_patterns_demo.ipynb`
- [ ] `examples/notebooks/03_sentiment_demo.ipynb`
- [ ] `examples/notebooks/04_scanner_demo.ipynb`
- [ ] `examples/notebooks/05_backtest_demo.ipynb`

**Scripts d'installation**:
- [ ] `Makefile` - Installation automatisée
- [ ] `.env.example` - Variables environnement
- [ ] `requirements.txt` - Dépendances complètes

---

### Livrables Session 4

**Code**:
- ✅ 4 derniers modules complets
- ✅ Scanner DSL fonctionnel
- ✅ ~1500 lignes de code production

**Tests**:
- ✅ Tests exhaustifs (200+ tests au total)
- ✅ Coverage > 90% sur TOUS modules
- ✅ Tests intégration end-to-end

**Documentation**:
- ✅ README complet
- ✅ 6 documents techniques
- ✅ 5 notebooks d'exemples
- ✅ API/CLI reference complètes

**Configuration**:
- ✅ requirements.txt finalisé
- ✅ Makefile installation
- ✅ .env.example

**Git**:
- ✅ Commit final
- ✅ Push
- ✅ Branche: `claude/session-4-finalization-[ID]`

**État Final**:
- ✅ **18 modules complets et testés**
- ✅ **200+ tests (coverage > 90%)**
- ✅ **API REST complète**
- ✅ **CLI complet**
- ✅ **Scanner DSL fonctionnel**
- ✅ **Documentation exhaustive**
- ✅ **Prêt pour production**

---

### Checkpoint Session 4 - FINAL

**Vous devez valider**:

1. **Tests Complets**:
```bash
pytest tests/ -v --cov=src --cov-report=html --cov-report=term
```
- ✅ 200+ tests passent
- ✅ Coverage > 90%

2. **API Complète**:
```bash
uvicorn src.api.main:app --reload
# Tester tous endpoints via http://localhost:8000/docs
```

3. **CLI Complet**:
```bash
python -m src.cli --help
python -m src.cli indicators run --indicator RSI --symbol AAPL --params '{"length":14}'
python -m src.cli scanner run --spec examples/scans/ema_golden_cross.json
```

4. **Scanner**:
```bash
python -m src.cli scanner run --spec examples/scans/ema_golden_cross.json
# Doit retourner résultats
```

5. **Documentation**:
- ✅ README.md clair et complet
- ✅ docs/ complètes
- ✅ Notebooks exécutables

6. **Régression**:
```bash
pytest tests/unit/test_*_validation.py -v
# 141 tests existants doivent TOUJOURS passer
```

---

## Récapitulatif des 4 Sessions

| Session | Modules | Tokens | Durée | État Final |
|---------|---------|--------|-------|------------|
| **1** | utils, data, candlesticks, vwap, zones | ~70k | 8h | Fondations prêtes |
| **2** | indicators (refacto), api, cli | ~80k | 12h | Core fonctionnel |
| **3** | patterns, trendlines, sentiment, breadth, rs, raindrop | ~90k | 14h | Modules avancés |
| **4** | fundamentals, economics, scanner, tests, docs | ~70k | 12h | Production ready |
| **TOTAL** | **18 modules** | **~310k** | **46h** | **Plateforme complète** |

**Note**: 310k tokens répartis sur 4 sessions de 200k chacune = OK (marge sécurité)

---

## Protocole Entre Sessions

### Quand une session se termine :

**MOI** :
1. ✅ Commit tous changements
2. ✅ Push vers remote
3. ✅ Génère rapport session :
   - Modules complétés
   - Tests passés
   - Coverage atteinte
   - Prochaine session prête
4. ✅ Message : "Session X terminée - Checkpoint"

**VOUS** :
1. ✅ Validez tests (commandes fournies dans checkpoint)
2. ✅ Vérifiez fonctionnalités critiques
3. ✅ Si OK : "Session X validée - GO Session Y"
4. ✅ Si problème : "STOP - Problème : [description]"

**MOI** (nouvelle session) :
1. ✅ Contexte chargé automatiquement
2. ✅ Continue selon plan Session Y
3. ✅ Progression continue

---

## Plan de Rollback

**Si problème détecté** :

### Niveau 1 - Correction dans session
- Problème mineur (test fail, bug)
- Je corrige immédiatement
- Re-test
- Continue

### Niveau 2 - Rollback module
- Problème majeur dans dernier module
- Git revert dernier commit
- Recommence module proprement
- Re-test
- Continue

### Niveau 3 - Rollback session
- Problème majeur affectant plusieurs modules
- Git reset vers checkpoint précédent
- Analyse root cause
- Recommence session avec corrections
- **VOUS décidez** si on continue ou stop

---

## Critères d'Arrêt d'Urgence

**J'arrête immédiatement si** :

1. ❌ **Tests régression échouent** (141 tests existants)
   - STOP - Demande décision
   - Rollback probable

2. ❌ **Stratégies refactorisées ≠ originales**
   - STOP - CRITIQUE
   - Rollback obligatoire

3. ❌ **Performance régression > 20%**
   - STOP - Analyse nécessaire
   - Optimisation ou rollback

4. ❌ **Tokens < 5k restants** en milieu de module
   - STOP proprement
   - Commit état actuel
   - Checkpoint session partielle

5. ❌ **Erreur bloquante non résolue** après 3 tentatives
   - STOP - Demande aide/décision

---

## Garanties de Qualité

**À la fin des 4 sessions, vous aurez** :

### Code
- ✅ 18 modules professionnels
- ✅ ~8000 lignes code production
- ✅ ~3000 lignes tests
- ✅ Type hints partout
- ✅ Docstrings complètes
- ✅ PEP8 compliant

### Tests
- ✅ 200+ tests unitaires
- ✅ 50+ tests intégration
- ✅ Coverage > 90%
- ✅ 141 tests existants passent
- ✅ Tests régression complets

### Fonctionnalités
- ✅ 200+ indicateurs techniques
- ✅ 60+ patterns chandeliers
- ✅ Chart patterns
- ✅ Trendlines auto
- ✅ Sentiment analysis
- ✅ Scanner DSL
- ✅ API REST (15+ endpoints)
- ✅ CLI complet
- ✅ Backtesting vectorisé

### Documentation
- ✅ README complet
- ✅ Architecture docs
- ✅ API reference
- ✅ CLI reference
- ✅ Scanner DSL docs
- ✅ 5+ notebooks exemples

### Production Ready
- ✅ Logging configuré
- ✅ Error handling robuste
- ✅ Performance optimisée
- ✅ Caching intelligent
- ✅ Rate limiting API
- ✅ Fallback strategies

---

## Prêt à Démarrer

**Quand vous dites "GO Session 1"**, je commence immédiatement :

1. Création module utils/
2. Tests utils/
3. Commit utils/
4. Création module data/
5. Tests data/
6. Commit data/
7. ... (continue jusqu'à fin session 1)
8. Push final
9. Rapport session 1
10. Attente validation

**Progression visible** :
- Commit après chaque module
- Messages clairs de progression
- Warnings si tokens faibles
- Checkpoint clair à la fin

---

## Votre Décision

**Êtes-vous prêt à lancer Session 1 ?**

Si oui, dites simplement : **"GO Session 1"**

Si vous avez des questions ou ajustements, dites-le maintenant.

**Après "GO Session 1", je travaille non-stop jusqu'à fin de Session 1 (~8h de travail, ~70k tokens).**
