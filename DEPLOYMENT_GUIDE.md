# 🚀 Guide de Déploiement - TradingSystemStack

Ce guide vous montre comment déployer et utiliser TradingSystemStack avec ses différentes interfaces.

---

## 📋 Table des Matières

1. [Installation](#installation)
2. [Déploiement API REST](#1-api-rest-fastapi)
3. [Interface Swagger UI](#2-swagger-ui-interface-web)
4. [CLI - Interface Ligne de Commande](#3-cli-interface-ligne-de-commande)
5. [Tests de Santé](#4-tests-de-santé)
6. [Exemples d'Utilisation](#5-exemples-dutilisation)

---

## Installation

### Prérequis
```bash
# Python 3.10+
python --version

# Installer les dépendances
cd TradingSystemStack
pip install -r requirements.txt

# TA-Lib (optionnel, pour certains indicateurs)
# Sur Ubuntu/Debian:
sudo apt-get install ta-lib

# Sur macOS:
brew install ta-lib

# Puis:
pip install TA-Lib
```

---

## 1. API REST (FastAPI)

### Démarrage Rapide

```bash
# Méthode 1: Via uvicorn (recommandé pour développement)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Méthode 2: Via Python directement
python -m src.api.main

# Méthode 3: Mode production avec Gunicorn
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Options de Configuration

```bash
# Développement avec auto-reload
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production (multi-workers)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Avec logs détaillés
uvicorn src.api.main:app --reload --log-level debug
```

### Accès à l'API

Une fois démarrée, l'API est accessible sur:
- **API Base**: http://localhost:8000
- **Documentation Swagger**: http://localhost:8000/docs
- **Documentation ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## 2. Swagger UI - Interface Web

**L'INTERFACE WEB PRINCIPALE** est Swagger UI, accessible après le démarrage de l'API.

### Accès Swagger UI

1. **Démarrez l'API**:
   ```bash
   uvicorn src.api.main:app --reload
   ```

2. **Ouvrez votre navigateur**:
   ```
   http://localhost:8000/docs
   ```

3. **Interface Swagger UI disponible!** 🎉

### Fonctionnalités Swagger UI

- ✅ **Exploration interactive** de tous les endpoints
- ✅ **Test en direct** des API calls
- ✅ **Documentation automatique** de tous les paramètres
- ✅ **Schémas de réponse** avec exemples
- ✅ **Try it out** - Exécution directe depuis le navigateur

### Endpoints Disponibles

#### 📊 Data Endpoints
- `GET /data/ohlcv` - Récupérer données OHLCV
- `GET /data/symbols` - Liste des symboles disponibles

#### 📈 Indicators Endpoints
- `POST /indicators/calculate` - Calculer n'importe quel indicateur
- `GET /indicators/list` - Liste tous les indicateurs disponibles

#### 🕯️ Candlesticks Endpoints
- `POST /candlesticks/detect` - Détecter patterns de chandelier

#### 📍 VWAP Endpoints
- `POST /vwap/calculate` - Calculer VWAP ancré

#### 🎯 Zones Endpoints
- `POST /zones/detect` - Détecter zones supply/demand

### Exemple d'Utilisation Swagger

1. **Naviguez vers** `http://localhost:8000/docs`
2. **Cliquez sur** un endpoint (ex: `POST /indicators/calculate`)
3. **Cliquez** "Try it out"
4. **Entrez** les paramètres:
   ```json
   {
     "symbol": "AAPL",
     "indicator": "RSI",
     "params": {
       "period": 14
     }
   }
   ```
5. **Cliquez** "Execute"
6. **Voyez** la réponse en temps réel!

---

## 3. CLI - Interface Ligne de Commande

### Commandes Disponibles

```bash
# Voir toutes les commandes
python -m src.cli --help

# Fetch data
python -m src.cli data-fetch --symbol AAPL --period 1y --interval 1d

# Calculer un indicateur
python -m src.cli indicator-run \
  --symbol AAPL \
  --indicator RSI \
  --params '{"period": 14}'

# Détecter patterns chandelier
python -m src.cli candlestick-detect \
  --symbol AAPL \
  --patterns DOJI,HAMMER

# Calculer VWAP
python -m src.cli vwap-calc \
  --symbol AAPL \
  --anchor-date 2024-01-01

# Détecter zones supply/demand
python -m src.cli zones-detect \
  --symbol AAPL \
  --lookback 100
```

### Exemples CLI Détaillés

#### Récupérer des Données
```bash
# Apple - 1 an de données daily
python -m src.cli data-fetch --symbol AAPL --period 1y --interval 1d

# Multiple symboles
python -m src.cli data-fetch --symbol "AAPL,MSFT,GOOGL" --period 6mo

# Crypto
python -m src.cli data-fetch --symbol BTC-USD --period 30d --interval 1h
```

#### Calculer Indicateurs
```bash
# RSI
python -m src.cli indicator-run \
  --symbol AAPL \
  --indicator RSI \
  --params '{"period": 14}'

# MACD
python -m src.cli indicator-run \
  --symbol AAPL \
  --indicator MACD \
  --params '{"fast": 12, "slow": 26, "signal": 9}'

# Bollinger Bands
python -m src.cli indicator-run \
  --symbol AAPL \
  --indicator BBANDS \
  --params '{"period": 20, "std": 2}'
```

---

## 4. Tests de Santé

### Vérifier que tout fonctionne

```bash
# Test 1: API Health Check
curl http://localhost:8000/health

# Réponse attendue:
# {"status": "healthy", "version": "2.0.0"}

# Test 2: Swagger UI accessible
# Ouvrir: http://localhost:8000/docs
# Vous devriez voir l'interface interactive

# Test 3: CLI fonctionnel
python -m src.cli --help

# Test 4: Récupérer des données via API
curl "http://localhost:8000/data/ohlcv?symbol=AAPL&period=1mo"
```

---

## 5. Exemples d'Utilisation

### Exemple 1: Workflow Complet via Swagger UI

1. **Démarrez l'API**:
   ```bash
   uvicorn src.api.main:app --reload
   ```

2. **Ouvrez** http://localhost:8000/docs

3. **Calculez RSI pour AAPL**:
   - Cliquez sur `POST /indicators/calculate`
   - Try it out
   - Body:
     ```json
     {
       "symbol": "AAPL",
       "indicator": "RSI",
       "params": {"period": 14}
     }
     ```
   - Execute

4. **Voyez les résultats** directement dans Swagger!

### Exemple 2: Workflow via API REST (Python)

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# 1. Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# 2. Calculer RSI
payload = {
    "symbol": "AAPL",
    "indicator": "RSI",
    "params": {"period": 14}
}
response = requests.post(f"{BASE_URL}/indicators/calculate", json=payload)
rsi_data = response.json()
print(f"RSI values: {rsi_data}")

# 3. Détecter patterns
payload = {
    "symbol": "AAPL",
    "patterns": ["DOJI", "HAMMER"]
}
response = requests.post(f"{BASE_URL}/candlesticks/detect", json=payload)
patterns = response.json()
print(f"Detected patterns: {patterns}")
```

### Exemple 3: Workflow via CLI

```bash
#!/bin/bash
# Script complet d'analyse

SYMBOL="AAPL"

# 1. Fetch data
echo "Fetching data for $SYMBOL..."
python -m src.cli data-fetch --symbol $SYMBOL --period 1y

# 2. Calculate indicators
echo "Calculating RSI..."
python -m src.cli indicator-run \
  --symbol $SYMBOL \
  --indicator RSI \
  --params '{"period": 14}'

# 3. Detect patterns
echo "Detecting candlestick patterns..."
python -m src.cli candlestick-detect \
  --symbol $SYMBOL \
  --patterns DOJI,HAMMER,ENGULFING

# 4. Calculate VWAP
echo "Calculating VWAP..."
python -m src.cli vwap-calc \
  --symbol $SYMBOL \
  --anchor-date 2024-01-01

echo "Analysis complete!"
```

---

## 6. Déploiement Production

### Option 1: Docker (Recommandé)

```dockerfile
# Créez un Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build
docker build -t trading-system-stack .

# Run
docker run -p 8000:8000 trading-system-stack

# Accès: http://localhost:8000/docs
```

### Option 2: Systemd Service (Linux)

```ini
# /etc/systemd/system/trading-api.service
[Unit]
Description=TradingSystemStack API
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/TradingSystemStack
ExecStart=/usr/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Activer le service
sudo systemctl enable trading-api
sudo systemctl start trading-api
sudo systemctl status trading-api
```

### Option 3: Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/trading-api
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 7. Troubleshooting

### Port déjà utilisé
```bash
# Trouver le processus sur le port 8000
lsof -i :8000

# Utiliser un autre port
uvicorn src.api.main:app --port 8001
```

### Module non trouvé
```bash
# Assurez-vous d'être à la racine du projet
cd TradingSystemStack

# Installer les dépendances
pip install -r requirements.txt

# Ajouter au PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Erreur TA-Lib
```bash
# Installer TA-Lib système
# Ubuntu:
sudo apt-get install ta-lib

# macOS:
brew install ta-lib

# Puis:
pip install TA-Lib
```

---

## 🎉 Résumé - Accès Rapide

**Pour démarrer MAINTENANT**:

```bash
# 1. Installer dépendances
pip install fastapi uvicorn typer pandas numpy

# 2. Démarrer l'API
uvicorn src.api.main:app --reload

# 3. Ouvrir dans le navigateur
# http://localhost:8000/docs

# C'est tout! Vous avez accès à l'interface web interactive! 🎉
```

**Interface Web**: http://localhost:8000/docs (Swagger UI)

**API Base**: http://localhost:8000

**CLI**: `python -m src.cli --help`

---

## 📚 Documentation Supplémentaire

- **API Reference**: Voir http://localhost:8000/docs après démarrage
- **Scanner DSL**: Voir `docs/SCANNER_DSL.md`
- **Tests**: `pytest tests/ -v`

---

**Besoin d'aide?** Consultez les logs ou ouvrez une issue sur GitHub!
