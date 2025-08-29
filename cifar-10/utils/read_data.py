import numpy as np

def get_train_label() -> np.ndarray: return np.load('data/train_label.npy')
def get_train_pic() -> np.ndarray: return np.load('data/train_pic.npy')
def get_train_label_test() -> np.ndarray: return np.load('data/train_label_test.npy')
def get_train_pic_test() -> np.ndarray: return np.load('data/train_pic_test.npy')
def get_train_label_train() -> np.ndarray: return np.load('data/train_label_train.npy')
def get_train_pic_train() -> np.ndarray: return np.load('data/train_pic_train.npy')

def get_test_pic() -> np.ndarray: return np.load('data/test_pic.npy')
