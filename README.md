
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

### [mnist_with_python.ipynb](https://raw.githubusercontent.com/Evegen55/math_with_python/master/src/jupiter/mnist_with_python.ipynb)

Solved for MNIST dataset with LogisticRegression(solver='sag', multi_class='ovr', n_jobs=1) with score = 0.9182

## License

Copyright (C) 2017 - 2017 [Evgenii Lartcev](https://github.com/Evegen55/) and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.