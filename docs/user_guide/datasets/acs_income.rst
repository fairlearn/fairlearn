.. _acsincome_data:

ACSIncome
---------


Introduction
^^^^^^^^^^^^

The ACSIncome dataset is one of five datasets created by
Ding et al. :footcite:`ding2021retiring`
as an improved alternative to the popular
UCI Adult dataset. :footcite:`kohavi1996adult`
Briefly, the UCI Adult dataset is commonly used as a benchmark dataset 
when comparing different algorithmic fairness interventions. ACSIncome offers 
a few improvements, such as providing more datapoints (1,664,500 vs. 48,842) 
and more recent data (2018 vs. 1994). Further, the binary labels in the UCI 
Adult dataset indicate whether an individual earned more than $50k US dollars 
in that year. Ding et al. show that the choice of threshold impacts the 
amount of disparity in proportion of positives, so they allow users to 
define any threshold rather than fixing it at $50k.

Ding et al. compiled data from the American Community Survey (ACS) Public 
Use Microdata Sample (PUMS). Note that this is a different source than the 
Annual Social and Economic Supplement (ASEC) of the Current Population 
Survey (CPS) used to construct the original UCI Adult dataset. Ding et al. 
filtered the data such that ACSIncome only includes individuals above 16 
years old who worked at least 1 hour per week in the past year and had an 
income of at least $100.


.. _acsincome_dataset_description:

Dataset Description
^^^^^^^^^^^^^^^^^^^
Ding et al. provide data from 2014-2018 for all 50 states and Puerto Rico.
Note that Puerto Rico is the only US territory included in this dataset.
We uploaded the 2018 data to `OpenML <https://www.openml.org/d/43141>`_.
The dataset contains 1,664,500 rows. Each row describes a person and contains 
10 features, which we describe below:

.. list-table::
   :header-rows: 1
   :widths: 7 30
   :stub-columns: 1

   *  - Column name
      - Description

   *  - AGEP
      - Age as an integer from 0 to 99

   *  - COW
      - Class of worker:
         1. Employee of a private for-profit company or business, or of an individual, for wages, salary, or commissions 
         2. Employee of a private not-for-profit, tax-exempt, or charitable organization 
         3. Local government employee (city, county, etc.) 
         4. State government employee 
         5. Federal government employee 
         6. Self-employed in own not incorporated business, professional practice, or farm 
         7. Self-employed in own incorporated business, professional practice or farm 
         8. Working without pay in family business or farm 
         9. Unemployed and last worked 5 years ago or earlier or never worked

   *  - SCHL
      - Educational attainment:
         1. No schooling completed
         2. Nursery school or preschool
         3. Kindergarten
         4. Grade 1
         5. Grade 2
         6. Grade 3
         7. Grade 4
         8. Grade 5
         9. Grade 6
         10. Grade 7
         11. Grade 8
         12. Grade 9
         13. Grade 10
         14. Grade 11
         15. Grade 12 (no diploma)
         16. Regular high school diploma
         17. GED or alternative credential
         18. Some college but less than 1 year
         19. 1 or more years of college credit but no degree
         20. Associate's degree
         21. Bachelor's degree
         22. Master's degree
         23. Professional degree beyond a bachelor's degree
         24. Doctorate degree

   *  - MAR
      - Marital status:
         1. Married
         2. Widowed
         3. Divorced
         4. Separated
         5. Never married or under 15 years old

   *  - OCCP
      - Occupation. There are over 500 categories. Please see data dictionary at ACS PUMS documentation :footcite:`census2019pums` for the full list of occupation codes.

   *  - POBP
      - Place of birth. There are over 200 categories, including the 50 US states and several countries. Please see the data dictionary at ACS PUMS documentation :footcite:`census2019pums` for the full list.

   *  - RELP
      - Relationship to householder:
         0. Reference person
         1. Husband or wife
         2. Biological son or daughter
         3. Adopted son or daughter
         4. Stepson or stepdaughter
         5. Brother or sister
         6. Father or mother
         7. Grandchild
         8. Parent-in-law
         9. Son-in-law or daughter-in-law
         10. Other relative
         11. Roomer or boarder
         12. Housemate or roommate
         13. Unmarried partner
         14. Foster child
         15. Other nonrelative
         16. Institutionalized group quarters population. Includes correctional facilities, nursing homes, and mental hospitals. :footcite:`census2023group`
         17. Noninstitutionalized group quarters population. Includes college dormitories, military barracks, group homes, missions, and shelters. :footcite:`census2023group`

   *  - WKHP
      - Usual hours worked per week in the past 12 months. Values are an integer from 1 to 99. Any hours above 99 are rounded down to 99

   *  - SEX
      - Sex code:
         1. Male
         2. Female

   *  - RAC1P
      - Race code
         1. White alone
         2. Black or African American alone
         3. American Indian alone
         4. Alaska Native alone
         5. American Indian and Alaska native tribes specified; or American Indian or Alaska Native, not specified and no other races
         6. Asian alone
         7. Native Hawaiian and Other Pacific Islander alone
         8. Some Other Race alone
         9. Two or More races


The target label is given by PINCP. For generalizability, the integer value 
is provided. A threshold can be applied to PINCP to frame this as a binary 
classification task.

.. list-table::
   :header-rows: 1
   :widths: 7 30
   :stub-columns: 1

   *  - Column name
      - Description

   *  - PINCP
      - Total annual income per person, denoted as an integer ranging from 104 to 1,423,000.


.. topic:: References:

   .. footbibliography::
