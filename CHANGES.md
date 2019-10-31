# Changes

### v0.3.0, 2019-11-01

* Major changes to the API. In particular the `expgrad` function is now implemented by the `ExponentiatedGradient` class. Please refer to the [ReadMe](readme.md) file for information on how to upgrade

* Added new algorithms
  * Threshold Optimization
  * Grid Search
  
* Added grouped metrics


### v0.2.0, 2018-06-20

* registered the project at [PyPI](https://pypi.org/)

* changed how the fairness constraints are initialized (in `fairlearn.moments`), and how they are passed to the fair learning reduction `fairlearn.classred.expgrad`

### v0.1, 2018-05-14

* initial release
