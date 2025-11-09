# ⚡ Quick Start - TradingSystemStack

Démarrage ultra-rapide en 3 étapes!

---

## 🚀 Démarrage en 30 Secondes

### Option 1: Script de Démarrage (Recommandé)

**Linux/macOS**:
```bash
./start_api.sh
```

**Windows**:
```cmd
start_api.bat
```

**Puis ouvrez**: http://localhost:8000/docs 🎉

---

### Option 2: Commande Directe

```bash
# Installer uvicorn si nécessaire
pip install uvicorn fastapi

# Démarrer l'API
uvicorn src.api.main:app --reload

# Ouvrir: http://localhost:8000/docs
```

---

## 🎯 Interface Utilisateur

Une fois l'API démarrée, vous avez accès à **Swagger UI** - l'interface web interactive complète!

### URLs Disponibles

| Interface | URL | Description |
|-----------|-----|-------------|
| **Swagger UI** | http://localhost:8000/docs | 🎨 Interface interactive principale |
| **ReDoc** | http://localhost:8000/redoc | 📚 Documentation alternative |
| **API Base** | http://localhost:8000 | 🔌 Endpoints API REST |
| **Health Check** | http://localhost:8000/health | ✅ Statut du système |

---

## 📊 Tester l'Interface

### Via Swagger UI (Interface Web)

1. **Ouvrez** http://localhost:8000/docs
2. **Cliquez** sur `POST /indicators/calculate`
3. **Cliquez** "Try it out"
4. **Collez** ce JSON:
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
6. **Voyez** le résultat immédiatement! 🎉

### Via cURL (Terminal)

```bash
# Health check
curl http://localhost:8000/health

# Calculer RSI
curl -X POST "http://localhost:8000/indicators/calculate" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "indicator": "RSI",
    "params": {"period": 14}
  }'
```

### Via CLI

```bash
# Voir les commandes disponibles
python -m src.cli --help

# Fetch data
python -m src.cli data-fetch --symbol AAPL --period 1y

# Calculer RSI
python -m src.cli indicator-run \
  --symbol AAPL \
  --indicator RSI \
  --params '{"period": 14}'
```

---

## 🎨 Fonctionnalités Swagger UI

L'interface Swagger UI vous permet de:

✅ **Explorer** tous les endpoints API
✅ **Tester** en temps réel sans code
✅ **Voir** la documentation complète
✅ **Exécuter** des requêtes directement
✅ **Visualiser** les schémas de réponse

---

## 📈 Endpoints Principaux

### 1. Indicateurs Techniques
- `POST /indicators/calculate` - Calculer n'importe quel indicateur
- `GET /indicators/list` - Liste des 200+ indicateurs

### 2. Données OHLCV
- `GET /data/ohlcv` - Récupérer données de prix
- `GET /data/symbols` - Symboles disponibles

### 3. Patterns de Chandelier
- `POST /candlesticks/detect` - Détecter patterns (Doji, Hammer, etc.)

### 4. VWAP
- `POST /vwap/calculate` - VWAP ancré

### 5. Zones Supply/Demand
- `POST /zones/detect` - Zones de support/résistance

---

## 🔧 Résolution de Problèmes

### "Port déjà utilisé"
```bash
# Utiliser un autre port
uvicorn src.api.main:app --reload --port 8001
```

### "Module not found"
```bash
# S'assurer d'être à la racine du projet
cd TradingSystemStack

# Installer dépendances
pip install -r requirements.txt
```

### "uvicorn not found"
```bash
pip install uvicorn fastapi
```

---

## 📚 Documentation Complète

- **Guide de Déploiement**: `DEPLOYMENT_GUIDE.md`
- **Scanner DSL**: `docs/SCANNER_DSL.md`
- **Architecture**: À venir

---

## 🎉 Succès!

Si vous voyez Swagger UI sur http://localhost:8000/docs, c'est un succès!

**Vous avez maintenant accès à l'interface web complète du TradingSystemStack!** 🚀

---

**Questions?** Consultez `DEPLOYMENT_GUIDE.md` pour plus de détails.
