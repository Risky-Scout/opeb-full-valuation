                ${results.eoy_tol:>12,.0f}")
    print()
    print("Sensitivities:")
    print(f"  Discount +1%:           ${results.sensitivity_disc_plus:>12,.0f}")
    print(f"  Discount -1%:           ${results.sensitivity_disc_minus:>12,.0f}")
    print(f"  Trend +1%:              ${results.sensitivity_trend_plus:>12,.0f}")
    print(f"  Trend -1%:              ${results.sensitivity_trend_minus:>12,.0f}")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("GASB 75 Full Valuation Excel Updater")
    print("=" * 50)
    print()
    print("Usage:")
    print("  from opeb_valuation.excel_updater import update_full_valuation_excel")
    print()
    print("  update_full_valuation_excel(")
    print("      input_path='prior_year.xlsx',")
    print("      output_path='current_year.xlsx',")
    print("      inputs=FullValuationInputs(...),")
    print("      results=FullValuationResults(...),")
    print("  )")
