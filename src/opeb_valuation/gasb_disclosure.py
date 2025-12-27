from opeb_valuation.gasb_disclosure import populate_gasb75_disclosure, ValuationResults

results = ValuationResults(
    client_name="West Florida Planning",
    measurement_date=date(2025, 9, 30),
    tol_boy_old_rate=24010,
    tol_eoy_baseline=23696.43,
    # ... etc
)

populate_gasb75_disclosure(
    template_path="template.xlsx",
    output_path="output.xlsx",
    results=results,
    prior_year_path="prior_year.xlsx",
    remove_yellow=True,
    remove_notes=True
)
