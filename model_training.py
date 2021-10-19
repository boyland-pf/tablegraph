import random
import torch
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from torch_geometric.loader import DataLoader
from data_creation import *

config = {'batch_size':100,'test_fraction':0.1}

def train(model,optimizer,train_data,device):
	model.train()
	total_loss = 0
	train_loader = DataLoader(train_data,batch_size=config['batch_size'],shuffle=True)
	pbar = tqdm(train_loader,ncols=70)
	for data_batch in pbar:
		data_batch = data_batch.to(device)
		optimizer.zero_grad()
		logits = model(data_batch.z,data_batch.edge_index,data_batch.batch)
		loss = BCEWithLogitsLoss()(logits.view(-1),data_batch.y.to(torch.float))
		loss.backward()
		optimizer.step()
		total_loss += loss.item() * data_batch.num_graphs
	return total_loss / len(train_data)

def skip_training(model,optimizer,saved_folder):
	model.load_state_dict(torch.load(saved_folder+"model.pth"))
	optimizer.load_state_dict(torch.load(saved_folder+"optimizer.pth"))

def four_results(pred,y_true):
	if y_true == 1:
		if pred == 1:
			return "true_pos"
		else:
			return "false_neg"
	else:
		if pred == 1:
			return "false_pos"
		else:
			return "true_neg"

@torch.no_grad()
def test(model,test_data,device):
	model.eval()
	test_loader = DataLoader(test_data,batch_size=1,shuffle=True)

	y_pred, y_true = [], []
	for data_batch in tqdm(test_loader,ncols=70):
		data_batch = data_batch.to(device)
		logits = model(data_batch.z, data_batch.edge_index, data_batch.batch)
		y_pred.append(logits.view(-1).cpu())
		y_true.append(data_batch.y.view(-1).cpu().to(torch.float))
		# print("logits",y_pred[-1])
	preds = [1 if logit > 0 else 0 for logit in y_pred]
	fourway = [four_results(a,b) for (a,b) in zip(preds,y_true)]
	r = {cat:fourway.count(cat) for cat in ["true_pos","false_neg","true_neg","false_pos"]}
	total_tested = r["true_pos"]+r["false_neg"]+r["true_neg"]+r["false_pos"]
	print("total_tested",total_tested)
	acc = (r["true_pos"]+r["true_neg"])/float(total_tested)
	print("acc",acc)
	print(r)

@torch.no_grad()
def predict(model,predict_dataset,device):
	model.eval()
	y_pred = []
	y_true = []
	for data in predict_dataset:
		data = data.to(device)
		logits = model(data.z, data.edge_index, data.batch)
		y_pred.append(logits.view(-1).cpu())
	if len(y_pred) == 1:
		prediction = y_pred[0].item()
		if prediction < 0:
			print("edge")
		else:
			print("no edge")
		return prediction
	y_pred = torch.stack(y_pred)
	y_pred[y_pred>0] = 1
	y_pred[y_pred<0] = 0
	print("predicted edges are with cells with #s",[i-2 for (i,x) in enumerate(y_pred) if x.item() == 1 and i-2>0])
	print("in total:",len([i-2 for (i,x) in enumerate(y_pred) if x.item() == 1 and i-2>0]))
	return y_pred

def split_train_test(table_datasets):
	total_num_data = sum([len(dataset) for dataset in table_datasets])
	random.shuffle(table_datasets)
	test_dataset = []
	train_dataset = []
	tests_to_get = total_num_data*config['test_fraction']
	test = True
	for dataset in table_datasets:
		if not test:
			train_dataset += dataset
			continue
		test_dataset += dataset
		if len(test_dataset)>=tests_to_get:
			test = False
	return train_dataset, test_dataset

