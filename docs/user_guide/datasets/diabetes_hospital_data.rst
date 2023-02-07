.. _diabetes_hospital_data:
Diabetes 130-Hospitals Dataset
------------------------------

Introduction
^^^^^^^^^^^^

The Diabetes 130-Hospitals Dataset consists of 10 years worth of clinical care data
at 130 US hospitals and integrated delivery networks :footcite:`strack2014impact`.
Each record represents the hospital admission record for a patient diagnosed with
diabetes whose stay lasted between one to fourteen days. Also, laboratory tests were
performed and medications were administered during the encounter. The features
describing each encounter include demographics, diagnoses, diabetic medications, number
of visits in the year preceding  the encounter, and payer information, as well as
whether the patient was readmitted after release, and whether the readmission occurred
within 30 days of the release.

Strack et al. used the data to investigate the impact of HbA1c measurement
on hospital readmission rates. The data was collected from the Health Facts
database, which is a national data warehouse in the United States consisting of
clinical records from hospitals throughout the United States. Once Strack et al.
completed their research, the dataset was submitted to the UCI Machine Learning
Repository such that it became available for later use.

.. _diabetes_hospital_dataset_description:

Dataset Description
^^^^^^^^^^^^^^^^^^^

The original data can be found in the UCI Repository :footcite:`strack2014diabetes`.
This version of the dataset was derived by the Fairlearn team for the SciPy 2021
tutorial "Fairness in AI Systems: From social context to practice using Fairlearn".
In this version, the target variable "readmitted" is binarized into whether the
patient was readmitted within thirty days. The full dataset pre-processing script
can be found on `GitHub <https://github.com/fairlearn/talks/blob/main/2021_scipy_tutorial/preprocess.py>`_.
The dataset contains 101,766 rows. Each row describes a patient encounter and
contains 25 features, which we describe below:

.. list-table::
   :header-rows: 1
   :widths: 7 30
   :stub-columns: 1

   *  - Column name
      - Description

   *  - race
      - Race of the patient:
         - African American
         - Asian
         - Caucasian
         - Hispanic
         - Other
         - Unknown

   *  - gender
      - Gender of patient:
         - Female
         - Male
         - Unknown/Invalid

   *  - age
      - Age of patient:
         - 30 years or younger
         - 30-60 years
         - Over 60 years

   *  - discharge_disposition_id
      - The place the patient was discharged to:
         - Discharged to Home
         - Other

   *  - admission_source_id
      - Means of admission into the hospital:
         - Emergency
         - Other
         - Referral

   *  - time_in_hospital
      - Integer number of days between admission and discharge.

   *  - medical_specialty
      - Specialty of the admitting physician:
         - Cardiology
         - Emergency/Trauma
         - Family/GeneralPractice
         - InternalMedicine
         - Missing
         - Other

   *  - num_lab_procedures
      - Integer number of lab tests performed during the encounter

   *  - num_procedures
      - Integer number of procedures (other than lab tests) performed during the
        encounter

   *  - num_medications
      - Integer number of distinct generic names administered during the encounter

   *  - primary_diagnosis
      - The primary (first) diagnosis:
         - Diabetes
         - Genitourinary Issues
         - Musculoskeletal Issues
         - Respiratory Issues
         - Other

   *  - number_diagnoses
      - Integer number of diagnoses.

   *  - max_glu_serum
      - Indicates the range of the result in mg/dL or if the Glucose serum test was not taken:
         - >200
         - >300
         - Norm (indicating normal)
         - None

   *  - A1Cresult
      - Indicates the range of the result in percentages or if the A1c test was
        not taken:
         - >7 (greater than 7%, but less than 8%)
         - >8 (greater than 8%)
         - Norm (indicating normal, which is less than 7%)
         - None

   *  - insulin
      - Indicates whether the drug was prescribed or there was a change in the dosage:
         - Down
         - Steady
         - Up
         - No

   *  - change
      - Indicates if there was a change in diabetic medications:
         - Ch (Change)
         - No (no change)

   *  - diabetesMed
      - Binary attribute indicating whether there was any diabetic medication
        prescribed.

   *  - medicare
      - Binary attribute indicating whether the patient had medicare as insurance.

   *  - medicaid
      - Binary attribute indicating whether the patient had medicaid as insurance.

   *  - had_emergency
      - Binary attribute indicating whether the patient had an emergency in the prior
        year.

   *  - had_inpatient_days
      - Binary attribute indicating whether the patient had inpatient days in the prior
        year.

   *  - had_outpatient_days
      - Binary attribute indicating whether the patient had outpatient days in the
        prior year.

   *  - readmitted
      - Attribute indicating whether the patient was readmitted and when. Can also be used as a target variable:
         - <30 (readmitted in less than 30 days)
         - >30 (readmitted in more than 30 days)
         - NO (not readmitted)

   *  - readmit_binary
      - Binary attribute indicating whether the patient was readmitted. Can also be
        used as a target variable.


The default target label is given by readmit_30_days. However, the "readmitted" or
"readmit_binary" attributes can also be used as a target, depending on what you
are interested in.

.. list-table::
   :header-rows: 1
   :widths: 7 30
   :stub-columns: 1

   *  - Column name
      - Description

   *  - readmit_30_days
      - Binary attribute indicating whether the patient was readmitted within 30 days.


.. _using_diabetes_hospital_dataset:

Using the dataset
^^^^^^^^^^^^^^^^^
The dataset can be loaded via the :func:`fairlearn.datasets.fetch_diabetes_hospital`
function. By default, the dataset is returned as a :class:`pandas.DataFrame`.

.. topic:: References:

    .. footbibliography::
