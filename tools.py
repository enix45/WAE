import pickle

def load_pickle(file_name):
	with open(file_name, 'rb') as f:
		data = pickle.load(f)
	return data

def save_pickle(data, file_name):
	with open(file_name, 'wb') as f:
		pickle.dump(data, f)


