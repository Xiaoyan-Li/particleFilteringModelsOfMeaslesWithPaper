# particle Filtering Models Of Measles With Paper

## Software used to build the models

we employ the software of Anylogic to build these models with the version of 8.1.0.
Readers interested in reproducing the models, please download the [Anylogic PLE version of 8.1.0](https://www.anylogic.com/files/anylogic-ple-8.1.0.x86_64.exe) for free to open the models in the folder of "models". Then the models could be reproduced by running them with the software of Anylogic PLE version of 8.1.0.

## Folder contents:
  - code of classification: include the Python code of analyzing the next month outbreak classification; the empirical monthly measles reported cases and the model result of the monthly measles reported cases of each particle from the year 1921 to 1956 (the minimum discrepancy model -- age structured particle filtering model where the children are up the end of 14 years old, and incorporating both the monthly and yearly empirical dataset).
  - data: the input and output data of all models
    - input data: the monthly and yearly measles reported cases normalized to the average population employed in the models in this research. The monthly data are from 1921 to 1956, while the yearly data are available from 1925 to 1956. It is notable that the yearly empirical data are age-structured. Both the yearly empirical of four age groups are included: less than 5 year, equal and greater than 5 years, less than 15 years, equal and greater than 15 years.
    - result data: include the discrepancy data of all models.
  - models: include all the models in this research. Readers interested could run these models with the software of [Anylogic PLE version of 8.1.0](https://www.anylogic.com/files/anylogic-ple-8.1.0.x86_64.exe).

  