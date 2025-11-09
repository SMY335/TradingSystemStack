#!/bin/bash
# Script de démarrage rapide de l'API TradingSystemStack

echo "🚀 Démarrage de TradingSystemStack API..."
echo ""
echo "📍 L'API sera accessible sur:"
echo "   - API Base:     http://localhost:8000"
echo "   - Swagger UI:   http://localhost:8000/docs"
echo "   - ReDoc:        http://localhost:8000/redoc"
echo "   - Health Check: http://localhost:8000/health"
echo ""
echo "🔄 Démarrage en mode développement (auto-reload)..."
echo ""

# Démarrer l'API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
