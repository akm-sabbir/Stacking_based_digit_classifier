#! usr/bin/env python
#! -*- coding: utf-8 -*- 
from sklearn.datasets import fetch_mldata
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import sklearn
import numpy as np
import scipy as sp
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import Scaler
from sklearn.metrics import classification_report
from nolearn.dbn import DBN
from joblib import Parallel,delayed
from multiprocessing import Pool
from multiprocessing import Lock,Manager
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import tree
from train_test_data_read import test_op
from train_test_data_read import mnist_data
from train_test_data_read import mnist_data_read
from sklearn.metrics import confusion_matrix
import csv
import pickle
class EnsembleClassifier(object):
	def __init__(self,baseEstimator = None):
		self.estimator = baseEstimator
		return
	def fit(self,X,y):
		if(self.estimator == None):
			print('there is no estimator to train')
			return
		for ind,each_estimator in enumerate(self.estimator):
			if(ind == 4 ):
				print('train svm')
				new_x,new_y = X,y#shuffle(X,y)
				train_new_x,test_new_x,train_new_y,test_new_y = train_test_split(new_x,new_y,test_size = 0.0,random_state = 42)
				each_estimator.fit(train_new_x,train_new_y)
				print('end of svm training')
			else:
				each_estimator.fit(X,y)
		self.classes = self.estimator[0].classes_
		print('train done on each estimator')
		return
	def get_class_list():
		return
	def predict_proba(self,X_test):
		self.prob_list = []
		for each_estimator in self.estimator:
			self.prob_list.append(each_estimator.predict_proba(X_test))
		return (np.mean(self.prob_list,axis = 0),self.prob_list)
	def build_confusion_mat(self,prob_mat):
		class_list = self.estimator[0].classes_
		y_predict = []
		for each in prob_mat:
			y_predict.append(class_list[np.argmax(each)])
			
		return y_predict
	def predict_mat(self,list_mat):
		data_mat = np.zeros((np.shape(list_mat[0])[0],len(self.estimator)))
		'''
		for ind, each in enumerate(list_mat):
			for sub_ind,sub_each in enumerate(each):
				data_mat[sub_ind][ind] = self.classes[np.argmax(sub_each)]
		'''
		data_mat = np.hstack(list_mat)
		return data_mat
	def get_estimator_results(self):
		return self.prob_list
	def get_stacked_result(self):
		if(self.prob_list == None ):
			print('there is some problem with probability results check it in')
		if(self.stack_mat == None):
			self.stack_mat = np.vstack(self.prob_list).T
		return self.stack_mat
class Stacking(object):
	def __init__(self,base_estimators = None,final_estimator = None):
		self.final_estimator = final_estimator
		self.estimator = base_estimators
		return
	def form_final_mat(self,X_train = None,X_test = None, y_train = None,y_test = None):
		mat_train = self.estimator.predict_proba(X_train)
		self.train_predict = self.estimator.predict_mat(mat_train[1])
		mat_test = self.estimator.predict_proba(X_test)
		self.test_predict = self.estimator.predict_mat(mat_test[1])
		#self.data_mat = np.hstack((X,self.estimator.get_stacked_result))
		self.y_train = y_train
		self.y_test = y_test
		return
	def fit(self,X = None,y = None):
		if(self.final_estimator == None):
			print('something is wrong with final estimator it may not be initialized')
			return
		self.final_estimator.fit(self.train_predict,self.y_train)
		return
	def perform_grid_search(self):
		parameter_sets = get_parameter_sets()
		perform_grid_search(self.train_predict,self.y_train,self.test_predict,self.y_test,parameter_sets,file_name = 'stacking_grid_01.txt')
		return
	def predict_proba(self,X_test):
		return self.final_estimator.predict_proba(self.test_predict)
	def predict(self,X_test = None ,y_test = None):
		y_predict = self.final_estimator.predict(self.test_predict)
		with open('prediction/stack_prediction.txt','w+') as result_writer:
			for each in y_predict:
				result_writer.write(str(each)+'\n')
		#self.correct_predict = np.sum(y_predict == self.y_test)
		return #(self.correct_predict,np.shape(self.y_test)[0] - self.correct_predict)
class Boosting(object):
	def __init__(self,base_estimator = AdaBoostClassifier()):
		self.estimator = base_estimator
		return
	def fit(self,X,y):
		self.estimator.fit(X,y,sample_weight = None)
		return
	def predict(self,X):
		return self.estimator.predict(X)
	def scoring(self,data,y):
		y_pred = self.predict(data)
		correct_result = np.sum(y == y_pred)
		return (correct_result,np.shape(y)[0] - correct_result)

def do_operation_boost(X_train,X_test,y_train,y_test,C = 20.,gamma = 0.01,learn_r = 1.):
		clf = AdaBoostClassifier(base_estimator = SVC(C = C,gamma = gamma,kernel='rbf',probability = True),n_estimators = 30,learning_rate = learn_r )
		clf.fit(X_train,y_train)
		y_test,y_pred = y_test, clf.predict(X_test)
		result = np.sum(y_test == y_pred)
		return (result,C,gamma)

def train_adaboost(X_train,X_test,y_train,y_test,parameter_set,learn_opt = False):
		C_set = parameter_set['C']
		gamma_set = parameter_set['gamma']
		if(learn_opt == False):
			print('start core operations on C and gamma')
			get_result = Parallel(n_jobs = 25)(delayed(do_operation_boost)(X_train,X_test,y_train,y_test,c_val,gamma_val) for c_val in C_set for gamma_val in gamma_set)
		else:
			print('start core testing on learning rate')
			learn_rate = [0.001,0.1,0.5,1.,2.,5.,8.,10.,15.,20.,30.,50.]
			get_result = Parallel(n_jobs = 15)(delayed(do_operation_boost)(X_train,X_test,y_train,y_test,learn_r = l_r) for l_r in learn_rate )
		max_val = -10
		max_ob = None
		for each in get_result:
			if(each[0] > max_val):
				max_ob = each
				max_val = each[0]
		print('total test size: ' + str(np.shape(y_test)))
		print('total correct classification: ' + str(max_ob[0]))
		print('model parameters are: C : ' + str(max_ob[1]) + ' gamma : ' + str(max_ob[2]))
		return

def perform_shuffle(dataset,target):
	data_set,tar_get = shuffle(dataset,target,random_state = 0)
	return (data_set,tar_get)
def perform_pca_operation(dataset,number_of_com = 10):
	pca = PCA(n_components = number_of_com)
	dataset = pca.fit_transform(dataset)
	return (pca,dataset)
def perform_dimension_reduction(pca ,data_set):
	reduced_dimension = pca.fit_transform(data_set)
	return reduced_dimension
def do_operation_(X_train,X_test,y_train,y_test,l_r,d_r):
	clf = DBN([np.shape(X_train)[1],300,10],learn_rates = l_r,learn_rate_decays = d_r,epochs = 30,verbose = 1 )
	clf.fit(X_train,y_train)
	y_test,y_pred = y_test, clf.predict(X_test)
	result = np.sum(y_test == y_pred)
	return (result,l_r,d_r)
def do_operations(args):
	return do_operation(*args)
def train_DBN(X_train,X_test,y_train,y_test):
	learning_rate = np.arange(0.1,1,0.1)
	decaying_rate = np.arange(0.6,1,0.1)
	get_result = Parallel(n_jobs = 20)(delayed(do_operation_)(X_train,X_test,y_train,y_test,learn_rate,decay_rate) for learn_rate in learning_rate for decay_rate in decaying_rate)
	max_val = -10
	max_ob = None
	for each in get_result:
		if(each[0] > max_val):
			max_ob = each
			max_val = each[0]
	print('total test size: ' + str(np.shape(y_test)))
	print('total correct classification: ' + str(max_ob[0]))
	print('model parameters are: l_r: ' + str(max_ob[1]) + ' d_r: ' + str(max_ob[2]))

	return

def load_data(path_name = None):
	data_sets = fetch_mldata('MNIST original')
	X = np.array(data_sets.data,np.float64)
	y = np.array(data_sets.target) # previously it was not numpy
	return (X,y)
def get_splitted_data(X,y,test_size = 0.25,random_state = 42):
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = test_size, random_state= 42)
	return (X_train,X_test,y_train,y_test)
#def do_operation(X_train,X_test,y_train,y_test,):
#	return
def get_parameter_sets():
	gamma_r = np.array([0.001,0.01,0.1,1,5,10,20])
	c_range = np.array([0.01,0.1,1,5,10,15,20,25])
	degree_r = np.arange(1,11)
	param_rbf = dict(gamma = gamma_r,C = c_range,kernel = ['rbf'])
	param_poly = dict(degree = degree_r,C = c_range,kernel = ['poly'])
	parameter_sets = [param_rbf,param_poly]
	return parameter_sets
def perform_grid_search(X_train,y_train,X_test,y_test,parameter_sets,file_name = None,model = SVC()):
	with open(file_name,'w') as result_writer,open('confusion_mat.txt','w+') as confusion_writer:	
		grid = GridSearchCV(model,parameter_sets,cv = StratifiedKFold(y = y_train,n_folds = 4 ),n_jobs = 25)
		print('Start training on grid search: ')
		grid.fit(X_train,y_train)
		print('writing down best estimator')
		result_writer.write('best paramter set is: '+ str(grid.best_estimator_) )
		result_writer.write('\n')
		y_test,y_predict = y_test, grid.predict(X_test)
		'''
		with open('prediction/stack_prediction.txt','w+') as result_writer:
			for each in y_predict:
				result_writer.write(str(each)+'\n')
		'''
		result_writer.write(classification_report(y_test,y_predict))
		arr = confusion_matrix(y_test,y_predict)
		for each in arr:
			for sub_each in each:
				confusion_writer.write(str(sub_each) + ' ')
			confusion_writer.write('\n')
		#confusion_writer.write(str(confusion_matrix(y_test,y_predict)))
		result_writer.write('\n')
		correct = np.sum(y_test == y_predict)
		print('total correct: ' + str(correct)+ ' total size:' + str(np.shape(y_test)[0]))
	
	return
def build_random_forest(n_estimator = 10,_njobs = 2):
	rand_ob = RandomForestClassifier(n_estimators = n_estimator, n_jobs = _njobs)
	return rand_ob
def build_extreme_random_forest(n_estimator = 10,_njobs = 2):
	rand_ob = ExtraTreesClassifier(n_estimators = n_estimator,n_jobs = _njobs)
	return rand_ob
def build_adaboost(base_estimator = tree.DecisionTreeClassifier(),n_estimators = 50,learning_rate = 1.0):
	rand_ob = AdaBoostClassifier(base_estimator = base_estimator,n_estimators = n_estimators,learning_rate = learning_rate)
	return rand_ob
def build_nonlinear_svm(c_value = 1., gamma = 0.):
	rand_ob = SVC(C= c_value,gamma = gamma,probability = True)
	return rand_ob
def build_linear_svc():
	rand_ob = LogisticRegression(C = 0.9)
	return rand_ob
def perform_train_test(X_train,X_test,y_train,y_test):
	svm_classifier = SVC(C = 10.0,kernel = 'rbf',probability = True,gamma = 0.001)
	print('Start training dataset:')
	svm_classifier.fit(X_train,y_train)
	print('End of training dataset')
	print('start testing dataset')
	y_predict = svm_classifier.predict(X_test)
	with open('prediction/caltechPredictLabel.dat','w+') as file_writer:
		for each in y_predict:
			file_writer.write(str(each)+'\n')
	print('end of Testing dataset')
	'''
	total_result = np.sum(y_test == y_predict)
	print('total data size is: ' + str(np.shape(y_predict)[0]))
	print('total corrected prediction: ' + str(total_result))
	'''
	return
def get_reduced_dim(X,pca_mod):
	X = pca_mod.transform(X)

def read_data_file(path_name = None,target = 0):
	with open(path_name,'rU') as files:
		data = csv.reader(files,delimiter = ',')
		X = []
		y = []
		count = 0
		for rows in data:
			if(count == 0):
				count += 1
				continue
			X.append(np.array(rows[target + 1:]))
			y.append(rows[target])
	print(str(np.shape(X)))
	print(str(len(y)))
	return (np.vstack(X),np.array(y,dtype = np.uint8))

def read_data_hw4():
	with open('model/caltechTrainData.dat','r') as data_reader, open('model/caltechTrainLabel.dat') as label_reader,open('model/caltechTestData.dat') as test_reader:
		X = data_reader.readlines()
		X_test = test_reader.readlines()
		list_data = []
		for each in X:
			list_data.append(each.split())
		test_list_data = []
		for each in X_test:
			test_list_data.append(each.split())

		x_data = np.vstack(list_data)
		test_list_data = np.vstack(test_list_data)
		print(str(np.shape(x_data))+ ' ' + str(type(x_data)))
		x_data = np.asarray(x_data,dtype= np.float32)
		test_list_data = np.asarray(test_list_data,dtype = np.float32)
		y = label_reader.readlines()
		Y = np.array(y,dtype = np.uint8)
		x_data = x_data/255.
		test_list_data = test_list_data/255.
	return (x_data,Y,test_list_data)
def main_operation(ensemble_set = 0,grid_set = 0,stacking = 0,_svm = 0,_dbn = 0,boosting = 0):
	#X,y = load_data()
	#X = X/255.
	#moddata = perform_pca_operation(X,number_of_com = 35)
	#X,y = moddata[1],y #perform_shuffle(X,y)
	#X,y = tup[0],tup[1]
	'''
	tup = mnist_data_read()
	X_train_r,X_test_r,y = tup[0],tup[1],tup[1]
	print(str(np.shape(X_train_r)))
	print(str(np.shape(X_test_r)))
	print(str(np.shape(y)))
	'''
	tup = read_data_hw4()
	X_train_r,y,X_test_r = tup[0],tup[1],tup[2]
	#X,y = read_data_file(path_name = 'train.csv',target = 0)
	#X,y = test_op()[0],y
	X_train,X_test,y_train,y_test = get_splitted_data(X_train_r, y, test_size = 0.0)
	if(boosting == 1):
		X_train_1,X_test_1,y_train_1,y_test_1 = get_splitted_data(X_train,y_train,test_size = 0.4,random_state = 42)
		print('kick off the boosting')
		boost = Boosting(base_estimator = AdaBoostClassifier(base_estimator = SVC(C = 20.,gamma = 0.01,probability = True),algorithm = 'SAMME.R'),n_estimators = 20)
		if(grid_set == 1):
			print('start grid search on Adaboost')
			parameter_set = get_parameter_sets()
			train_adaboost(X_train_1,X_test_1,y_train_1,y_test_1,parameter_set[0])
		else:
			print('Start of training')
			boost.fit(X_train_1,y_train_1)
			print('Start of testing')
			boost.scoring(X_test_1,y_test_1)
		return
	if(ensemble_set == 1):
		print('kick off of bagging')
		rand_forest = build_random_forest(n_estimator = 50,_njobs = 32)
		rand_extreme_forest = build_extreme_random_forest(n_estimator = 50, _njobs = 32)
		rand_adaboost = build_adaboost()
		rand_LinearSVC = build_linear_svc()
		rand_non_svm = build_nonlinear_svm(c_value = 20.,gamma = 0.01)
		ensemble_classifier = EnsembleClassifier([rand_forest,rand_extreme_forest,rand_adaboost,rand_LinearSVC,rand_non_svm])
		ensemble_classifier.fit(X_train,y_train)
		#output_mat = ensemble_classifier.predict_proba(X_test)
		if(stacking  == 0):
			output_mat = ensemble_classifier.predict_proba(X_test)
			y_predict = ensemble_classifier.build_confusion_mat(output_mat[0])
			print('Total correct prediction: ' + str(np.sum(y_test == y_predict)))
		if(stacking == 1 ):
			print('kick off of stacking')
			X_train_2,X_test_2,y_train_2,y_test_2 = get_splitted_data(X_test,y_test,test_size = .20)
			stacked = Stacking(base_estimators = ensemble_classifier, final_estimator = SVC(C = 20,gamma = 0.1,kernel='rbf'))
			stacked.form_final_mat(X_train = X_train_2,X_test = X_test_2,y_train = y_train_2,y_test = y_test_2)
			if(True):
				stacked.perform_grid_search()
############################## with out grid search ###########################
			else:
				stacked.fit()
				result = stacked.predict()
				print('predicted result: ' + str(result[0]) +' wrong prediction: '+ str(result[1]))
###################################################################################

	#parameter_sets = get_parameter_sets()
	if(_svm == 1):
		perform_train_test(X_train,X_test_r,y_train,y_test)
	if(_dbn == 1):
		train_DBN(X_train,X_test,y_train,y_test)
	#print(np.shape(X_train))
	#print(np.shape(X_test))
	#print(np.shape(y_train))
	#print(np.shape(y_test))
	##### perform grid search ###############	
	if(grid_set == 1):
		parameter_sets = get_parameter_sets()
		X_train_1,X_test_1,y_train_1,y_test_1 = train_test_split(X_train,y_train,test_size = 0.2,random_state = 42)
		perform_grid_search(X_train_1,y_train_1,X_test_1,y_test_1,parameter_sets,file_name = 'resultant_file_grid_hw4.txt')
	###################################################################
	return
main_operation(ensemble_set = 0 ,stacking = 0,grid_set = 0, _svm = 1,_dbn = 0,boosting = 0)
