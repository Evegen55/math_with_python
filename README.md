
## Examples for tensorflow 
### [1_notmnist.ipynb](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb)

  Has solution in files like the src/spyder/tensorflow_examples_udacity_1_notmnist_soved_for_****k_set.ipynb
   
  It used to solve classification problem with 
  [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) - 
  [see docs](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) - 
  and
  
  gets next accuracy of predicting:
  - 0.804 for 10000 samples
  - 0.8273 for 200000 samples
  - 0.8273 for 500000 samples
  
  Trained models are in files
  - resources/trained_nn/finalized_model_log_regr_200K_samples.sav
  - resources/trained_nn/finalized_model_log_regr_500K_samples.sav
  
  You can use this models by using code like this:
  
    import numpy as np
    from six.moves import cPickle as pickle
    from sklearn.linear_model import LogisticRegression
    ...
    loaded_model = pickle.load(open(filename_for_log_regr, 'rb')) #here is a loaded LogisticRegression
    ...
  
## NOTE

Be careful to point the `data_root` just because code loads and unpacks LOTS OF data  

### [mnist_with_python.ipynb](https://github.com/Evegen55/math_with_python/blob/master/src/jupiter/mnist_with_python.ipynb)

**Result of prediction for 7:**
![**Result for 7**](https://raw.githubusercontent.com/Evegen55/math_with_python/master/resources/solved/tremendous_7_my_mnist.PNG)

Solved for MNIST dataset with LogisticRegression(solver='sag', multi_class='ovr', n_jobs=1) with score = 0.9182

[Another example from docs](http://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html#sphx-glr-auto-examples-linear-model-plot-sparse-logistic-regression-mnist-py)
shows a classification vectors for each of classes we have:

![classes](http://scikit-learn.org/stable/_images/sphx_glr_plot_sparse_logistic_regression_mnist_001.png)

## License

Copyright (C) 2017 - 2017 [Evgenii Lartcev](https://github.com/Evegen55/) and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.