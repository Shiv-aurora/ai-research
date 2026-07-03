#!/usr/bin/env python3
"""
RIVE Research Audit - Master Runner
====================================
Runs all audit tests and produces a comprehensive report.

Usage:
    python "january tests/run_audit.py"

Author: External Audit
Date: January 2026
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header():
    """Print audit header."""
    print("\n" + "="*70)
    print("╔" + "═"*68 + "╗")
    print("║" + " "*20 + "RIVE RESEARCH AUDIT" + " "*29 + "║")
    print("║" + " "*15 + "External Verification Suite" + " "*25 + "║")
    print("╚" + "═"*68 + "╝")
    print("="*70)
    print(f"\n  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project: RIVE (Regime-Integrated Volatility Ensemble)")
    print(f"  Claimed Performance: 61% R² (Top 50), 22% R² (GICS-55)")
    print("="*70)


def run_test_suite():
    """Run all audit tests."""

    all_results = {}

    # =============================
    # TEST 1: Data Leakage
    # =============================
    print("\n\n" + "🔍 "*23)
    print("  RUNNING TEST 1: DATA LEAKAGE DETECTION")
    print("🔍 "*23 + "\n")

    try:
        from importlib import import_module
        test1 = import_module("01_data_leakage_audit")
        results1 = test1.run_all_tests()
        all_results["leakage"] = results1
    except Exception as e:
        print(f"  ✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_results["leakage"] = {"error": str(e)}

    # =============================
    # TEST 2: R² Verification
    # =============================
    print("\n\n" + "📊 "*23)
    print("  RUNNING TEST 2: R² METRIC VERIFICATION")
    print("📊 "*23 + "\n")

    try:
        test2 = import_module("02_r2_verification")
        results2 = test2.run_all_tests()
        all_results["r2_verification"] = results2
    except Exception as e:
        print(f"  ✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_results["r2_verification"] = {"error": str(e)}

    # =============================
    # TEST 3: Shuffle Test
    # =============================
    print("\n\n" + "🎲 "*23)
    print("  RUNNING TEST 3: SHUFFLE TEST (GOLD STANDARD)")
    print("🎲 "*23 + "\n")

    try:
        test3 = import_module("03_shuffle_test")
        results3 = test3.run_all_tests()
        all_results["shuffle"] = results3
    except Exception as e:
        print(f"  ✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_results["shuffle"] = {"error": str(e)}

    # =============================
    # TEST 4: Reproducibility
    # =============================
    print("\n\n" + "🔬 "*23)
    print("  RUNNING TEST 4: REPRODUCIBILITY TEST")
    print("🔬 "*23 + "\n")

    try:
        test4 = import_module("04_reproducibility_test")
        results4 = test4.run_all_tests()
        all_results["reproducibility"] = results4
    except Exception as e:
        print(f"  ✗ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_results["reproducibility"] = {"error": str(e)}

    return all_results


def print_final_report(all_results):
    """Print comprehensive audit report."""

    print("\n\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*20 + "FINAL AUDIT REPORT" + " "*30 + "║")
    print("╚" + "═"*68 + "╝")

    # Count passed tests
    total_tests = 0
    passed_tests = 0

    # 1. Leakage Tests
    print("\n  📋 TEST 1: DATA LEAKAGE DETECTION")
    print("  " + "-"*50)

    if "leakage" in all_results and "error" not in all_results["leakage"]:
        for test, result in all_results["leakage"].items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"      {test}: {status}")
            total_tests += 1
            if result:
                passed_tests += 1
    else:
        print("      ✗ TEST SUITE ERROR")

    # 2. R² Verification
    print("\n  📋 TEST 2: R² METRIC VERIFICATION")
    print("  " + "-"*50)

    if "r2_verification" in all_results and "error" not in all_results["r2_verification"]:
        for test, result in all_results["r2_verification"].items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"      {test}: {status}")
            total_tests += 1
            if result:
                passed_tests += 1
    else:
        print("      ✗ TEST SUITE ERROR")

    # 3. Shuffle Tests
    print("\n  📋 TEST 3: SHUFFLE TEST (GOLD STANDARD)")
    print("  " + "-"*50)

    if "shuffle" in all_results and "error" not in all_results["shuffle"]:
        for test, result in all_results["shuffle"].items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"      {test}: {status}")
            total_tests += 1
            if result:
                passed_tests += 1
    else:
        print("      ✗ TEST SUITE ERROR")

    # 4. Reproducibility
    print("\n  📋 TEST 4: REPRODUCIBILITY")
    print("  " + "-"*50)

    if "reproducibility" in all_results and "error" not in all_results["reproducibility"]:
        for test, (passed, value) in all_results["reproducibility"].items():
            if isinstance(value, float):
                print(f"      {test}: R² = {value:.4f} ({value*100:.2f}%)")
            elif isinstance(value, dict):
                print(f"      {test}: {len(value)} sectors analyzed")
            total_tests += 1
            if passed:
                passed_tests += 1
    else:
        print("      ✗ TEST SUITE ERROR")

    # Summary
    print("\n" + "="*70)
    print("  OVERALL SUMMARY")
    print("="*70)

    print(f"\n  Tests Passed: {passed_tests}/{total_tests}")

    pass_rate = passed_tests / total_tests if total_tests > 0 else 0

    if pass_rate >= 0.9:
        verdict = "🏆 AUDIT PASSED - RESEARCH IS SOUND"
        grade = "A"
    elif pass_rate >= 0.75:
        verdict = "✅ AUDIT PASSED - MINOR ISSUES"
        grade = "B"
    elif pass_rate >= 0.5:
        verdict = "⚠️ AUDIT MARGINAL - REVIEW RECOMMENDED"
        grade = "C"
    else:
        verdict = "❌ AUDIT FAILED - SIGNIFICANT ISSUES"
        grade = "F"

    print(f"\n  Verdict: {verdict}")
    print(f"  Grade: {grade}")

    # Key findings
    print("\n  KEY FINDINGS:")
    print("  " + "-"*50)

    # Check for leakage
    if "shuffle" in all_results and "error" not in all_results["shuffle"]:
        shuffle_passed = all(all_results["shuffle"].values())
        if shuffle_passed:
            print("  ✓ No data leakage detected (shuffle test passed)")
        else:
            print("  ⚠ Potential data leakage (shuffle test failed)")

    # Check reproducibility
    if "reproducibility" in all_results and "error" not in all_results["reproducibility"]:
        if "full_ensemble" in all_results["reproducibility"]:
            _, r2 = all_results["reproducibility"]["full_ensemble"]
            print(f"  ✓ Reproduced ensemble R²: {r2*100:.2f}%")

    print("\n" + "="*70)
    print(f"  Audit completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def main():
    """Main entry point."""

    print_header()

    print("\n  Starting comprehensive audit...")
    print("  This will run 4 test suites with multiple checks each.\n")

    all_results = run_test_suite()

    print_final_report(all_results)


if __name__ == "__main__":
    main()
