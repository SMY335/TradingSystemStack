# 🌐 Guide d'Utilisation 100% En Ligne

**Vous avez des droits limités sur votre PC ? Pas de problème !**

Ce guide vous explique comment exécuter TradingSystemStack directement depuis votre navigateur, sans rien installer localement.

---

## 🎯 Options Disponibles

### ✅ Option 1 : GitHub Codespaces (RECOMMANDÉ)

**Avantages :**
- Intégré directement dans GitHub
- VS Code complet dans le navigateur
- Configuration automatique
- Ports forwarding automatique pour les dashboards
- **Gratuit : 60h/mois** (compte gratuit) ou **90h/mois** (GitHub Pro)

**Limitations :**
- Nécessite un compte GitHub
- Limites de temps d'utilisation

---

## 🚀 Démarrage avec GitHub Codespaces

### Étape 1 : Créer un Codespace

1. **Allez sur le dépôt GitHub** de votre projet
2. Cliquez sur le bouton **"Code"** (vert)
3. Sélectionnez l'onglet **"Codespaces"**
4. Cliquez sur **"Create codespace on main"** (ou votre branche)

![Création d'un Codespace](https://docs.github.com/assets/cb-138303/images/help/codespaces/new-codespace-button.png)

### Étape 2 : Attendre l'Installation

Le Codespace va :
- ✅ Créer un conteneur Python 3.11
- ✅ Installer toutes les dépendances (TA-Lib, vectorbt, etc.)
- ✅ Configurer l'environnement
- ⏱️ **Durée : 3-5 minutes** (première fois seulement)

Vous verrez un terminal avec :
```
🚀 Setting up TradingSystemStack...
📦 Installing system dependencies...
📊 Installing TA-Lib...
🐍 Installing Python packages...
✅ Setup complete!
```

### Étape 3 : Lancer les Dashboards

Une fois l'installation terminée, vous pouvez lancer :

#### Dashboard Backtesting
```bash
./run_dashboard.sh
```
- **Port : 8501**
- Une notification apparaîtra : cliquez sur **"Open in Browser"**

#### Dashboard Live Trading
```bash
./run_live_dashboard.sh
```
- **Port : 8502**

#### Dashboard Portfolio
```bash
./run_portfolio_dashboard.sh
```
- **Port : 8503**

### Étape 4 : Accéder aux Dashboards

Deux méthodes :

**Méthode A : Via la notification**
- Cliquez sur **"Open in Browser"** quand elle apparaît

**Méthode B : Via l'onglet Ports**
1. Allez dans l'onglet **"PORTS"** (en bas)
2. Trouvez le port 8501, 8502, ou 8503
3. Cliquez sur l'icône **globe** 🌐 pour ouvrir dans le navigateur

---

## 🎓 Utilisation Quotidienne

### Démarrer un Codespace Existant

1. Allez sur **github.com/codespaces**
2. Cliquez sur votre Codespace existant
3. Il redémarre en **~30 secondes**

### Arrêter un Codespace (Important !)

⚠️ **Pour économiser vos heures gratuites :**

1. **Arrêt automatique :** Le Codespace s'arrête après 30 min d'inactivité
2. **Arrêt manuel :**
   - Cliquez sur **"Codespaces"** (en bas à gauche)
   - Sélectionnez **"Stop Current Codespace"**

### Supprimer un Codespace

Si vous n'en avez plus besoin :
1. Allez sur **github.com/codespaces**
2. Cliquez sur les **trois points** ⋯ à côté du Codespace
3. Sélectionnez **"Delete"**

---

## 🌟 Option 2 : Gitpod (Alternative)

### Avantages
- Similaire à Codespaces
- **Gratuit : 50h/mois**
- Interface VS Code

### Démarrage Rapide

1. **Préfixez votre URL GitHub** avec `gitpod.io/#`

   Exemple :
   ```
   https://gitpod.io/#https://github.com/VOTRE-USERNAME/TradingSystemStack
   ```

2. **Connectez-vous** avec votre compte GitHub

3. **Attendez l'installation** (3-5 minutes la première fois)

4. **Lancez les dashboards** comme avec Codespaces

---

## 💡 Option 3 : Replit (Plus Simple)

### Avantages
- Interface très simple
- Pas besoin de configuration
- Gratuit avec limitations

### Configuration Manuelle

1. **Créez un compte** sur [replit.com](https://replit.com)
2. **Créez un nouveau Repl** → Import from GitHub
3. **Collez l'URL** de votre dépôt
4. **Installez manuellement** :
   ```bash
   pip install -r requirements_frameworks.txt
   pip install pandas numpy vectorbt streamlit plotly
   ```
5. **Lancez** `./run_dashboard.sh`

⚠️ **Limitations :**
- TA-Lib peut ne pas fonctionner (bibliothèque C)
- Performances limitées

---

## 🔧 Dépannage

### "Port already in use"

Si un dashboard ne démarre pas :
```bash
# Trouvez le processus
lsof -ti:8501

# Tuez-le
kill -9 $(lsof -ti:8501)

# Relancez
./run_dashboard.sh
```

### "Module not found"

Si un module manque :
```bash
pip install nom-du-module
```

### Streamlit ne charge pas

Essayez de redémarrer avec l'option `--server.headless true` :
```bash
streamlit run src/dashboard/app.py --server.headless true --server.port 8501
```

### Codespace trop lent

1. **Arrêtez** le Codespace actuel
2. **Recréez-en un nouveau** → parfois ça aide
3. **Utilisez une machine plus puissante** :
   - Paramètres → Machine type → 4-core

---

## 📊 Comparaison des Options

| Critère | GitHub Codespaces | Gitpod | Replit |
|---------|-------------------|--------|--------|
| **Gratuit** | 60-90h/mois | 50h/mois | Limité |
| **Configuration** | Automatique | Automatique | Manuelle |
| **Performance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Facilité** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **TA-Lib** | ✅ | ✅ | ❌ |
| **Recommandé** | ✅ **OUI** | ✅ Oui | ⚠️ Limité |

---

## 🎯 Workflow Recommandé

### Pour Backtesting (Analyse Historique)

1. **Démarrez un Codespace**
2. **Lancez** `./run_dashboard.sh`
3. **Testez vos stratégies** dans le navigateur
4. **Arrêtez le Codespace** quand vous avez fini

**Temps utilisé :** ~1-2h par session

### Pour Paper Trading (Tests Temps Réel)

⚠️ **Important :** Les Codespaces s'arrêtent après 30 min d'inactivité.

**Solution :**
- Utilisez un **serveur cloud permanent** (VPS) pour le paper trading 24/7
- Ou gardez votre Codespace actif en arrière-plan (coûte des heures gratuites)

**Alternative :**
```bash
# Lancez le bot en CLI (sans dashboard)
python run_paper_trading_bot.py --symbol BTC/USDT --timeframe 1h
```

---

## 💰 Coûts et Limites

### GitHub Codespaces (Plan Gratuit)

- **60 heures/mois** pour comptes gratuits
- **120 core-hours/mois** (2-core machine = 60h)
- **15 GB de stockage**

**Exemple d'utilisation :**
- 2h de backtesting par jour = **60h/mois** ✅
- 1h de paper trading par jour = **30h/mois** ✅

### GitHub Codespaces (GitHub Pro - $4/mois)

- **90 heures/mois**
- **180 core-hours/mois**
- **20 GB de stockage**

### Gitpod (Plan Gratuit)

- **50 heures/mois**
- Suffisant pour backtesting régulier

---

## 🔒 Sécurité et Données

### Vos Données Restent Privées

- ✅ Les Codespaces sont **privés** et isolés
- ✅ Seul **vous** avez accès à votre environnement
- ✅ Les données sont **chiffrées** au repos

### Clés API et Secrets

Pour le paper/live trading avec clés API :

1. **Utilisez les Secrets de Codespaces** :
   - Paramètres du dépôt → Secrets → Codespaces
   - Ajoutez vos clés API

2. **Dans le code** :
   ```python
   import os
   api_key = os.getenv('EXCHANGE_API_KEY')
   ```

⚠️ **JAMAIS** de clés API en dur dans le code !

---

## 🚀 Tips Avancés

### 1. Précharger les Dépendances

Pour démarrer plus vite, les dépendances sont déjà configurées via `.devcontainer/devcontainer.json`.

### 2. Persister les Données

Les données dans `/workspaces/TradingSystemStack` persistent entre les sessions.

### 3. Utiliser le Terminal

Codespaces = VS Code complet :
- Terminal intégré
- Éditeur de code
- Extensions Python
- Debugging

### 4. Collaborer en Temps Réel

Partagez votre Codespace avec d'autres (comme Google Docs) :
- Codespaces → Share → Copy link

---

## 📚 Ressources Supplémentaires

- [Documentation GitHub Codespaces](https://docs.github.com/en/codespaces)
- [Documentation Gitpod](https://www.gitpod.io/docs)
- [README.md principal](./README.md)
- [Guide Backtesting](./TRADING_BOT_GUIDE.md)
- [Guide Paper Trading](./PAPER_TRADING_GUIDE.md)

---

## ✅ Checklist de Démarrage

- [ ] Créer un compte GitHub (si pas déjà fait)
- [ ] Créer un Codespace sur votre dépôt
- [ ] Attendre l'installation (3-5 min)
- [ ] Lancer `./run_dashboard.sh`
- [ ] Tester une stratégie de backtesting
- [ ] Arrêter le Codespace quand terminé

---

## 🎉 Vous êtes Prêt !

Vous pouvez maintenant utiliser TradingSystemStack **100% en ligne**, sans rien installer sur votre ordinateur.

**Bon trading ! 📈🤖**

---

**Questions ? Problèmes ?**

Ouvrez une issue sur GitHub ou consultez la documentation complète.
