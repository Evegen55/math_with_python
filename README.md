
## Examples for tensorflow 
### [1_notmnist.pynn](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb)

  Has solution in files like the spyder/tensorflow_examples_udacity_1_notmnist_soved_for_****k_set.ipynb
   
  It used to solve classification problem with 
  [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) - 
  [see docs](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) - 
  and
  
  gets next accuracy of predicting:
  - 0.804 for 10000 samples
  - 0.8273 for 200000 samples
  - 0.8273 for 500000 samples
  
  Trained models are in files
  - trained_nn/finalized_model_log_regr_200K_samples.sav
  - trained_nn/finalized_model_log_regr_500K_samples.sav
  
  You can use this models by using code like this:
  
    import numpy as np
    from six.moves import cPickle as pickle
    from sklearn.linear_model import LogisticRegression
    ...
    loaded_model = pickle.load(open(filename_for_log_regr, 'rb')) #here is a loaded LogisticRegression
    ...
  
## NOTE

Be careful to point the `data_root` just because code loads and unpacks LOTS OF data