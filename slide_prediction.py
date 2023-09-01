import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, svm, neural_network, neighbors, decomposition
import misc

class SlidePrediction():
  def __init__(self, subdir, patient_split, slide_to_label):
    self.fv_len = 512
    self.output_dict = misc.load_pkl(os.path.join(self.predictions_dir,'full_output_dict.pkl'))

  def get_top_feature_vectors(self, topN=50,
                              dimensionality_reduction=True):
    self.fvs_dict = {}
    self.labels_dict = {}
    for mode in ['train', 'test']:
      #fvs = np.empty((0, topN*self.fv_len))
      labels = np.empty(0)

      patient_list = list(self.output_dict[mode].keys())
      for patient in patient_list:
        print(patient)
        try:
          label = np.array(label_to_value[get_label(patient)]).reshape(1)
          labels = np.concatenate((labels, label))
          softmax = self.output_dict[mode][patient]['softmax'][:,1]
          softmax = softmax.reshape((-1,1))
          fv = self.output_dict[mode][patient]['fv']

          # zip softmax and feature vectors together
          dict = {np.array_str(softmax[i])[1:-1]: fv[i]
                  for i in range(len(softmax))}
          z = np.array(list(zip(softmax,fv)))
          # rank feature vectors by softmax value and unzip
          z = z[z[:,0].argsort()]
          softmax, full_fv = zip(*(tuple(z)))

          # verify that softmax and fv are still properly matched up
          for i in random.sample(range(0, len(z)),100):
            k = np_to_str(z[i][0])
            v = z[i][1]
            #if not (dict.get(k) == v).all(): print(k)
            #if dict.get(k) is not None:
            #  assert (dict.get(k) == v).all(), print( (dict.get(k) == v))

          # take top 100 feature vectors
          full_fv = np.array(full_fv)
          fv = full_fv[-topN:, :]
          if dimensionality_reduction:
            pca = decomposition.PCA(n_components=10)
            fv = pca.fit_transform(fv)
          # reshape and concatenate full fv array
          fv = fv.ravel()
          fv = np.expand_dims(fv, 0)
          try:
            self.fvs_dict[mode] = np.concatenate((self.fvs_dict[mode], fv))
          except (NameError, KeyError):
            self.fvs_dict[mode] = fv
        except (IndexError, KeyError) as e:
          print('*** {} ***'.format(e))
          print(self.output_dict[mode][patient])
          self.output_dict[mode].pop(patient)
      #self.fvs_dict[mode] = fvs
      self.labels_dict[mode] = labels.reshape((-1,1))
    misc.save_pkl(self.fvs_dict, os.path.join(self.subdir, 'fvs_dict.pkl'))
    misc.save_pkl(self.labels_dict, os.path.join(self.subdir,
                                                 'labels_dict.pkl'))

  def KNN_prediction(self):
    self.predictions_dict = {}
    clf = neighbors.KNeighborsClassifier()
    clf.fit(self.fvs_dict['train'], self.labels_dict['train'])
    preds = clf.predict_proba(self.fvs_dict['test'])
    self.predictions_dict['test'] = preds[:,1]
    return self.predictions_dict, self.labels_dict

  def SVM_prediction(self):
    self.predictions_dict = {}
    clf = svm.SVC(probability=True)
    # use train feature vectors to train classifier
    clf.fit(self.fvs_dict['train'], self.labels_dict['train'])
    # evaluate on test feature vectors and output predictions
    preds = clf.predict_proba(self.fvs_dict['test'])
    self.predictions_dict['test'] = preds[:,1]
    return self.predictions_dict, self.labels_dict


  def MLP_prediction(self):
    pass

  def RNN_prediction(self):
    pass


# ——————————————————————————————————————————————————————————————————————

def evaluate_crossval_run(subdir, patient_split, prediction_types,
                          slide_to_label):
  predictor = SlidePrediction(subdir, patient_split,
                              slide_to_label)

  modes = ['train', 'test']
  if ('svm' or 'knn') in prediction_types:
    predictor.load_feature_vectors(topN=50,
                                   dimensionality_reduction=False)

  prediction_methods = {'avg_pool': predictor.avg_pool_prediction,
                        'max_pool': predictor.max_pool_prediction,
                        'svm': predictor.SVM_prediction,
                        'knn': predictor.KNN_prediction}

  test_metrics = {}
  for pred_type in prediction_types:
    print('\n{}'.format(pred_type))
    preds, labels = prediction_methods[pred_type]()
    if pred_type not in ['avg_pool', 'max_pool']:
      modes = ['test']

    for mode in modes:
      print(mode)
      metrics = evaluate_predictions(preds[mode], labels[mode], 0.99)
      if mode == 'test':
        test_metrics[pred_type] = metrics
  return test_metrics

def main(FLAGS):
  run_dir = os.path.join(FLAGS.runs_main, FLAGS.run_id)
  misc.init_output_logging(os.path.join(run_dir, 'prediction_logs.txt'))

  # load slide split
  patient_split = misc.load_pkl(os.path.join(run_dir, 'patient_split.pkl'))
  slide_to_label = misc.load_pkl(os.path.join(run_dir, 'slide_to_label.pkl'))

  run_subdirs = glob.glob(os.path.join(run_dir, 'run_*'))
  prediction_types = FLAGS.prediction_type.split(',')
  full_metrics = {}

  if FLAGS.cv_run == None:
    for i in range(1, len(run_subdirs)+1):
      print('\n\nCROSS VALIDATION RUN {}'.format(i))
      subdir = run_subdirs[i-1]
      full_metrics[i] = evaluate_crossval_run(subdir, patient_split[str(i)],
                                              prediction_types, slide_to_label)
    print('\nAverage test metrics values:')
    for pred_type in prediction_types:
      for cv_run in full_metrics.keys():
        run_metrics = np.array(full_metrics[cv_run][pred_type]).reshape((-1,1))
        try:
          metrics = np.concatenate((metrics, run_metrics), 1)
        except:
          metrics = run_metrics
      avg_metrics = [round(x, 3) for x in np.mean(metrics, 1)]
      print(pred_type, avg_metrics)
