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

## 2026-07-11 — PIT universe formation 2005: December 2004 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2004-12-01', 'date__lte': '2004-12-31'}
Row count returned: 6971

## 2026-07-11 — PIT universe formation 2005: eligible common-stock names as of 2004-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2004-12-31', 'nameendt__gte': '2004-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 74892

## 2026-07-11 — PIT universe formation 2006: December 2005 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2005-12-01', 'date__lte': '2005-12-31'}
Row count returned: 6970

## 2026-07-11 — PIT universe formation 2006: eligible common-stock names as of 2005-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2005-12-31', 'nameendt__gte': '2005-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 76587

## 2026-07-11 — PIT universe formation 2007: December 2006 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2006-12-01', 'date__lte': '2006-12-31'}
Row count returned: 7081

## 2026-07-11 — PIT universe formation 2007: eligible common-stock names as of 2006-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2006-12-31', 'nameendt__gte': '2006-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 80626

## 2026-07-11 — PIT universe formation 2008: December 2007 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2007-12-01', 'date__lte': '2007-12-31'}
Row count returned: 7183

## 2026-07-11 — PIT universe formation 2008: eligible common-stock names as of 2007-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2007-12-31', 'nameendt__gte': '2007-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 82346

## 2026-07-11 — PIT universe formation 2009: December 2008 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2008-12-01', 'date__lte': '2008-12-31'}
Row count returned: 6961

## 2026-07-11 — PIT universe formation 2009: eligible common-stock names as of 2008-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2008-12-31', 'nameendt__gte': '2008-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 84250

## 2026-07-11 — PIT universe formation 2010: December 2009 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2009-12-01', 'date__lte': '2009-12-31'}
Row count returned: 6760

## 2026-07-11 — PIT universe formation 2010: eligible common-stock names as of 2009-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2009-12-31', 'nameendt__gte': '2009-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 85114

## 2026-07-11 — PIT universe formation 2011: December 2010 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2010-12-01', 'date__lte': '2010-12-31'}
Row count returned: 6790

## 2026-07-11 — PIT universe formation 2011: eligible common-stock names as of 2010-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2010-12-31', 'nameendt__gte': '2010-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 86246

## 2026-07-11 — PIT universe formation 2012: December 2011 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2011-12-01', 'date__lte': '2011-12-31'}
Row count returned: 6866

## 2026-07-11 — PIT universe formation 2012: eligible common-stock names as of 2011-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2011-12-31', 'nameendt__gte': '2011-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 87843

## 2026-07-11 — PIT universe formation 2013: December 2012 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2012-12-01', 'date__lte': '2012-12-31'}
Row count returned: 6797

## 2026-07-11 — PIT universe formation 2013: eligible common-stock names as of 2012-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2012-12-31', 'nameendt__gte': '2012-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 89245

## 2026-07-11 — PIT universe formation 2014: December 2013 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2013-12-01', 'date__lte': '2013-12-31'}
Row count returned: 6947

## 2026-07-11 — PIT universe formation 2014: eligible common-stock names as of 2013-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2013-12-31', 'nameendt__gte': '2013-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 90245

## 2026-07-11 — PIT universe formation 2015: December 2014 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2014-12-01', 'date__lte': '2014-12-31'}
Row count returned: 7209

## 2026-07-11 — PIT universe formation 2015: eligible common-stock names as of 2014-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2014-12-31', 'nameendt__gte': '2014-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 91830

## 2026-07-11 — PIT universe formation 2016: December 2015 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2015-12-01', 'date__lte': '2015-12-31'}
Row count returned: 7371

## 2026-07-11 — PIT universe formation 2016: eligible common-stock names as of 2015-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2015-12-31', 'nameendt__gte': '2015-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 93010

## 2026-07-11 — PIT universe formation 2017: December 2016 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2016-12-01', 'date__lte': '2016-12-31'}
Row count returned: 7315

## 2026-07-11 — PIT universe formation 2017: eligible common-stock names as of 2016-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2016-12-31', 'nameendt__gte': '2016-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 94532

## 2026-07-11 — PIT universe formation 2018: December 2017 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2017-12-01', 'date__lte': '2017-12-31'}
Row count returned: 7427

## 2026-07-11 — PIT universe formation 2018: eligible common-stock names as of 2017-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2017-12-31', 'nameendt__gte': '2017-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 97061

## 2026-07-11 — PIT universe formation 2019: December 2018 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2018-12-01', 'date__lte': '2018-12-31'}
Row count returned: 7625

## 2026-07-11 — PIT universe formation 2019: eligible common-stock names as of 2018-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2018-12-31', 'nameendt__gte': '2018-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 98628

## 2026-07-11 — PIT universe formation 2020: December 2019 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2019-12-01', 'date__lte': '2019-12-31'}
Row count returned: 7725

## 2026-07-11 — PIT universe formation 2020: eligible common-stock names as of 2019-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2019-12-31', 'nameendt__gte': '2019-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 102696

## 2026-07-11 — PIT universe formation 2021: December 2020 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2020-12-01', 'date__lte': '2020-12-31'}
Row count returned: 8125

## 2026-07-11 — PIT universe formation 2021: eligible common-stock names as of 2020-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2020-12-31', 'nameendt__gte': '2020-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 105820

## 2026-07-11 — PIT universe formation 2022: December 2021 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2021-12-01', 'date__lte': '2021-12-31'}
Row count returned: 9451

## 2026-07-11 — PIT universe formation 2022: eligible common-stock names as of 2021-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2021-12-31', 'nameendt__gte': '2021-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 109434

## 2026-07-11 — PIT universe formation 2023: December 2022 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2022-12-01', 'date__lte': '2022-12-31'}
Row count returned: 9661

## 2026-07-11 — PIT universe formation 2023: eligible common-stock names as of 2022-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2022-12-31', 'nameendt__gte': '2022-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 111427

## 2026-07-11 — PIT universe formation 2024: December 2023 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2023-12-01', 'date__lte': '2023-12-31'}
Row count returned: 9591

## 2026-07-11 — PIT universe formation 2024: eligible common-stock names as of 2023-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2023-12-31', 'nameendt__gte': '2023-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 113734

## 2026-07-11 — PIT universe formation 2025: December 2024 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2024-12-01', 'date__lte': '2024-12-31'}
Row count returned: 9813

## 2026-07-11 — PIT universe formation 2025: eligible common-stock names as of 2024-12-31
Library/table: crsp.msenames
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={'namedt__lte': '2024-12-31', 'nameendt__gte': '2024-12-31', 'shrcd__in': '10,11', 'exchcd__in': '1,2,3'}
Row count returned: 117830

## 2026-07-11 — PIT universe: delisting records for all selected permnos
Library/table: crsp.msedelist
Output: data/raw/wrds/universe_pit_delist.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msedelist/ params={'permno__in': '10104,10107,10145,10147,10696,11081,11308,11552,11703,11762,11850,11990,12060,12305,12308,12369,12490,12558,12972,13033,13407,13447,13613,13721,13788,13856,13901,13928,14008,14277,14541,14542,14593,14702,14714,14888,15054,15069,15358,15408,15488,15579,15667,16424,16851,17005,17478,17746,17750,17778,17830,18143,18163,18411,18542,18576,18729,19350,19387,19393,19502,19561,19788,20482,20626,21178,21378,21776,21936,22103,22111,22592,22752,22779,23819,24205,25013,25267,25785,26403,27828,27887,27959,28310,29890,33099,34833,36469,38703,39087,39490,40539,43449,44644,45356,47896,48725,49154,49656,49680,50227,50876,51043,52919,55976,57665,59176,59184,59328,59408,59459,60097,60442,60628,61241,61399,62092,64186,64390,65875,65883,66093,66157,66181,66800,69032,69892,70332,70519,71298,73139,75154,75186,75333,75510,75646,75652,75789,75825,75844,76076,76226,76557,76614,76841,76932,77178,77274,77284,77418,77546,77605,77668,77702,78916,78975,79057,79237,79323,80070,80100,80599,81284,81593,81774,82654,82800,83225,83435,83443,83835,84181,84398,84788,85261,85442,85592,86111,86223,86356,86381,86451,86454,86455,86457,86580,86745,86755,86783,86868,86946,87031,87055,87128,87137,87212,87267,87447,87842,88215,88239,88352,88391,88396,88490,88668,88885,89003,89006,89071,89129,89130,89134,89179,89199,89258,89350,89393,89428,89525,89626,89730,89954,89996,90215,90319,90386,90441,90448,90505,90794,90857,90878,91001,91010,91130,91233,91321,91883,91937,91952,92027,92108,92187,92602,92611,92655,92922,93002,93112,93114,93116,93142,93144,93436'}
Row count returned: 255

## 2026-07-11 — PIT universe: full msenames history (filtered locally; API ignores shrcd/nameendt filters)
Library/table: crsp.msenames
Output: data/raw/wrds/msenames_full.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msenames/ params={}
Row count returned: 117830

## 2026-07-11 — PIT universe formation 2005: December 2004 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2004-12-01', 'date__lte': '2004-12-31'}
Row count returned: 6971

## 2026-07-11 — PIT universe formation 2006: December 2005 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2005-12-01', 'date__lte': '2005-12-31'}
Row count returned: 6970

## 2026-07-11 — PIT universe formation 2007: December 2006 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2006-12-01', 'date__lte': '2006-12-31'}
Row count returned: 7081

## 2026-07-11 — PIT universe formation 2008: December 2007 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2007-12-01', 'date__lte': '2007-12-31'}
Row count returned: 7183

## 2026-07-11 — PIT universe formation 2009: December 2008 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2008-12-01', 'date__lte': '2008-12-31'}
Row count returned: 6961

## 2026-07-11 — PIT universe formation 2010: December 2009 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2009-12-01', 'date__lte': '2009-12-31'}
Row count returned: 6760

## 2026-07-11 — PIT universe formation 2011: December 2010 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2010-12-01', 'date__lte': '2010-12-31'}
Row count returned: 6790

## 2026-07-11 — PIT universe formation 2012: December 2011 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2011-12-01', 'date__lte': '2011-12-31'}
Row count returned: 6866

## 2026-07-11 — PIT universe formation 2013: December 2012 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2012-12-01', 'date__lte': '2012-12-31'}
Row count returned: 6797

## 2026-07-11 — PIT universe formation 2014: December 2013 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2013-12-01', 'date__lte': '2013-12-31'}
Row count returned: 6947

## 2026-07-11 — PIT universe formation 2015: December 2014 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2014-12-01', 'date__lte': '2014-12-31'}
Row count returned: 7209

## 2026-07-11 — PIT universe formation 2016: December 2015 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2015-12-01', 'date__lte': '2015-12-31'}
Row count returned: 7371

## 2026-07-11 — PIT universe formation 2017: December 2016 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2016-12-01', 'date__lte': '2016-12-31'}
Row count returned: 7315

## 2026-07-11 — PIT universe formation 2018: December 2017 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2017-12-01', 'date__lte': '2017-12-31'}
Row count returned: 7427

## 2026-07-11 — PIT universe formation 2019: December 2018 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2018-12-01', 'date__lte': '2018-12-31'}
Row count returned: 7625

## 2026-07-11 — PIT universe formation 2020: December 2019 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2019-12-01', 'date__lte': '2019-12-31'}
Row count returned: 7725

## 2026-07-11 — PIT universe formation 2021: December 2020 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2020-12-01', 'date__lte': '2020-12-31'}
Row count returned: 8125

## 2026-07-11 — PIT universe formation 2022: December 2021 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2021-12-01', 'date__lte': '2021-12-31'}
Row count returned: 9451

## 2026-07-11 — PIT universe formation 2023: December 2022 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2022-12-01', 'date__lte': '2022-12-31'}
Row count returned: 9661

## 2026-07-11 — PIT universe formation 2024: December 2023 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2023-12-01', 'date__lte': '2023-12-31'}
Row count returned: 9591

## 2026-07-11 — PIT universe formation 2025: December 2024 month-end prices/shares
Library/table: crsp.msf
Output: data/raw/wrds/universe_pit_membership.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msf/ params={'date__gte': '2024-12-01', 'date__lte': '2024-12-31'}
Row count returned: 9813

## 2026-07-11 — PIT universe: delisting records for all selected permnos
Library/table: crsp.msedelist
Output: data/raw/wrds/universe_pit_delist.parquet
SQL/API call:
GET https://wrds-api.wharton.upenn.edu/data/crsp.msedelist/ params={'permno__in': '10104,10107,10145,10147,10696,11081,11308,11552,11703,11850,11990,12052,12060,12308,12369,12490,12558,12591,13356,13407,13447,13511,13721,13788,13856,13901,13928,14008,14040,14541,14542,14593,14702,14714,15069,15408,15488,15579,15667,15826,16424,16851,17005,17144,17478,17750,17778,17830,18163,18312,18411,18542,18576,18729,19350,19393,19502,19561,19654,19788,20482,20626,20894,21178,21207,21371,21776,21936,22103,22111,22265,22293,22592,22752,22779,23819,24205,24643,24766,24878,24942,25013,25785,26403,26710,27828,27887,27959,28222,34746,34833,36468,36469,38703,39087,39490,39642,40539,43449,44644,45751,47896,48486,48725,49154,49656,49680,50227,50876,51043,52919,53613,55976,56573,57665,57904,59176,59184,59328,59408,59459,60097,60442,60628,60871,61241,61399,62092,62148,64186,64390,64936,65875,65883,66093,66157,66181,66800,68144,69032,70332,70519,71563,73139,75186,75333,75510,75646,75789,75825,76076,76226,76557,76614,76744,76841,77178,77274,77284,77418,77546,77605,77668,77702,77768,78975,79212,79323,79678,80599,81055,81061,81593,81774,82642,82775,82800,83435,83443,84032,84788,85269,85348,86356,86580,86783,86868,86946,87031,87055,87137,87267,87447,87842,88352,88360,88668,88845,89003,89006,89130,89179,89258,89393,89525,89626,89813,89954,90215,90319,90386,90441,90505,91233,91883,91937,92108,92221,92602,92611,92655,93002,93436'}
Row count returned: 223
