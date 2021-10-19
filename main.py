from data_creation import *
import torch
from torch.utils.data import DataLoader
from model import *
from model_training import *
import os
from gencsvs import convert_graphs

num_epochs = 5
eval_every_n_epochs = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DGCNN()
optimizer = torch.optim.Adam(params=list(model.parameters()),lr=model.lr)

# datasets = create_dataset()
# train_dataset, test_dataset = split_train_test(datasets)
# print(train_dataset[0].to_dict())
# for epoch in range(num_epochs):
# 	loss = train(model,optimizer,train_dataset,device)
# 	print(loss)
# 	if (epoch + 1) % eval_every_n_epochs == 0:
# 		results = test(model,test_dataset,device)
# 		print(results)
# 	torch.save(model.state_dict(), "saved_models/model.pth")
# 	torch.save(optimizer.state_dict(), "saved_models/optimizer.pth")
skip_training(model,optimizer,"saved_models/")
print("done training")

while True:
	folder = input("pass in example name with input and output csvs\n")
	folder = "raw_data/"+folder+"/"
	convert_graphs(folder)
	x = int(input("type in the particular node whose link you want to know, or 0 for finding all predicted edges\n"))
	if x == 0:
		predict_data = get_predict_data(folder)
		predict(model,predict_data,device)
	elif x < 0:
		predict_data = get_negatives(folder)
		predict(model,predict_data,device)
	else:
		predict_data = get_positives(folder)
		predict(model,predict_data,device)