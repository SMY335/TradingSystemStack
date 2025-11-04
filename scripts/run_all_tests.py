"""
Script pour exécuter tous les tests unitaires
"""
import subprocess
import sys
from pathlib import Path

# Tests à exécuter
TESTS = [
    'tests/unit/test_walk_forward.py',
    'tests/unit/test_backtest_engine.py',
    'tests/unit/test_metrics.py',
    'tests/unit/test_monte_carlo.py',
]

def run_test(test_path):
    """Exécute un test et retourne le résultat"""
    print(f"\n{'='*60}")
    print(f"▶️  Exécution: {test_path}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(
        [sys.executable, test_path],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    if result.returncode == 0:
        print(result.stdout)
        return True, None
    else:
        print(f"❌ ÉCHEC\n")
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        return False, result.stderr


def main():
    """Execute tous les tests"""
    print("\n" + "="*60)
    print("🧪 EXÉCUTION DE TOUS LES TESTS UNITAIRES")
    print("="*60)
    
    results = {}
    failed_tests = []
    
    for test in TESTS:
        success, error = run_test(test)
        results[test] = success
        
        if not success:
            failed_tests.append((test, error))
    
    # Résumé
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*60 + "\n")
    
    total = len(TESTS)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for test, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test}")
    
    print(f"\n{'='*60}")
    print(f"Total: {total} | Passés: {passed} | Échoués: {failed}")
    print(f"{'='*60}\n")
    
    if failed_tests:
        print("❌ TESTS ÉCHOUÉS:\n")
        for test, error in failed_tests:
            print(f"  • {test}")
            if error:
                # Afficher uniquement la première ligne d'erreur
                first_line = error.split('\n')[0] if error else "Unknown error"
                print(f"    {first_line}\n")
        
        sys.exit(1)
    else:
        print("✅ TOUS LES TESTS SONT PASSÉS!\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
