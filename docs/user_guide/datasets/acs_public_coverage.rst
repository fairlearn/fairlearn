.. _acs_public_coverage:
ACSPublicCoverage
-------------------------

Introduction
^^^^^^^^^^^^^^^^^

The ACSPublicCoverage dataset is one of five datasets created by Ding et al. [1]_ 
as an improved alternative to the popular UCI Adult dataset [2]_.
Briefly, the UCI Adult dataset is commonly used as a benchmark dataset when comparing
different algorithmic fairness interventions. While UCI Adult is used to predict 
whether an individual's annual income is above $50,000, the authors created 
ACSPublicCoverage to predict whether a low-income individual who is not eligible for Medicare 
is covered by public health insurance, such as Medicaid or Medicare.
The authors compiled data from the American Community Survey (ACS) Public Use Microdata Sample (PUMS). 
Note that this is a different source than the Annual Social and Economic Supplement (ASEC) 
of the Current Population Survey (CPS) used to construct the original UCI Adult dataset.
Ding et al. [#0]_ filtered the data such that ACSPublicCoverage only includes individuals under 65 years old 
had an income of less than $30,000. Although usually only people over age 65 are eligible for Medicare,
people under age 65 may qualify if they have certain disabilities. 


.. _acs_public_coverage_dataset_description:

Dataset Description
^^^^^^^^^^^^^^^^^^^
The authors provide 2018 data for all 50 states and Puerto Rico.
Note that Puerto Rico is the only US territory included in this dataset.
The dataset contains 1,138,289 rows and 19 features, which we describe below:

.. list-table::
   :header-rows: 1
   :widths: 7 30
   :stub-columns: 1

   *  - Column name
      - Description

   *  - AGEP
      - Age as an integer from 0 to 99

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

   *  - SEX
      - Sex code:
         1. Male
         2. Female

   *  - DIS
      - Disability recode:
         1. With a disability
         2. Without a disability

   *  - ESP
      - Employment status of parents:
         1. Living with two parents: both parents in labor force
         2. Living with two parents: Father only in labor force
         3. Living with two parents: Mother only in labor force
         4. Living with two parents: Neither parent in labor force
         5. Living with father: Father in the labor force
         6. Living with father: Father not in labor force
         7. Living with mother: Mother in the labor force
         8. Living with mother: Mother not in labor force

   *  - CIT
      - Citizenship status:
         1. Born in the U.S.
         2. Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas
         3. Born abroad of American parent(s)
         4. U.S. citizen by naturalization
         5. Not a citizen of the U.S.

   *  - MIG
      - Mobility status (lived here 1 year ago):
         1. Yes, same house (nonmovers)
         2. No, outside US and Puerto Rico
         3. No, different house in US or Puerto Rico
         
   *  - MIL
      - Military service:
         1. Now on active duty
         2. On active duty in the past, but not now
         3. Only on active duty for training in Reserves/National Guard
         4. Never served in the military
         
   *  - ANC
      - Ancestry recode:
         1. Single
         2. Multiple
         3. Unclassified
         4. Not reported
         8. Suppressed for data year 2018 for select PUMAs

   *  - NATIVITY
      - Nativity:
         1. Native
         2. Foreign born
         
   *  - DEAR
      - Hearing difficulty:
         1. Yes
         2. No
         
   *  - DEYE
      - Vision difficulty:
         1. Yes
         2. No
         
   *  - DREM
      - Cognitive difficulty:
         1. Yes
         2. No
         
   *  - PINCP
      - Total annual income per person as an integer between -19997 and 4209995 US dollars. Loss of $19998 or more is coded as -19998. Income of $4209995 or more is coded as 4209995.

   *  - ESR
      - Employment status recode:
         1. Civilian employed, at work
         2. Civilian employed, with a job but not at work
         3. Unemployed
         4. Armed forces, at work
         5. Armed forces, with a job but not at work
         6. Not in labor force
         
   *  - ST
      - State code:
         Please see data dictionary at `ACS PUMS documentation <https://www.census.gov/programs-surveys/acs/microdata/documentation.2018.html>`_ for the full list of state codes.  

   *  - FER
      - Gave birth to child within the past 12 months:
         1. Yes
         2. No

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


The target label is given by PUBCOV, which can be used for a binary classification task.

.. list-table::
   :header-rows: 1
   :widths: 7 30
   :stub-columns: 1

   *  - Column name
      - Description

   *  - PUBCOV
      - Public health coverage, with PUBCOV == 1 if the individual has public health coverage, else 0

.. topic:: References:

  .. [1] Frances Ding, Moritz Hardt, John Miller, Ludwig Schmidt `"Retiring Adult: New Datasets for Fair Machine Learning" <https://arxiv.org/pdf/2108.04884.pdf>`_,
      Advances in Neural Information Processing Systems 34, 2021.

  .. [2] R. Kohavi and B. Becker. "UCI Adult Data Set." UCI Machine Learning Repository, 5, 1996.

