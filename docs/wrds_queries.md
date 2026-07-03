# WRDS Query Log

Every query issued against WRDS for this project is recorded here **verbatim**, with date, purpose, and output location, so the data pipeline is exactly reproducible by any WRDS subscriber.

Format per entry:

```
## <date> — <purpose>
Library/table: <e.g. taqm_2020.ctm_2020, crsp.msf>
Output: <data/raw/... path>
SQL/API call:
<verbatim query or wrds-py call>
Row count returned: <n>
```

_No queries issued yet. First entries will come from Phase 1 (universe construction from CRSP, then TAQ extraction)._
