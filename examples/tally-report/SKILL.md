---
name: tally-report
description: Summarize a CSV of line items into a small JSON report.
---

# Tally report

Use this when asked to tally, summarize, or total a CSV of line items.

## Steps

1. Read the CSV file named in the request.
2. Work out how many line items it contains, and the total of the amount column.
3. Write the results to `report.json` in the working directory.

Keep the report small — just the count and the total.
