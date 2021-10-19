import random
from random import choice 
import csv
import os

#make a long list of full graphs
#each of them is disconnected from the others

def isnum(x):
	try:
		float(x)
		return 1
	except:
		return 0

def sfl(x):
	return str(float(x))

# node features are (main cell, column, endofrow, endoftable, input/output indicator, isnum)
def convert_graphs(trainfolder):
	resnodes = []
	resedges = []

	inputtable = []
	outputtable = []

	print(trainfolder)

	# we read in the csvs as 2d arrays
	with open(trainfolder+"input.csv",'r') as inputcsvfile:
		inputcsv = csv.reader(inputcsvfile)
		for row in inputcsv:
			inputtable.append(list(row))
	with open(trainfolder+"output.csv",'r') as outputcsvfile:
		outputcsv = csv.reader(outputcsvfile)
		for row in outputcsv:
			outputtable.append(list(row))

	# no check for file validity 
	numcolsinput = len(inputtable[0])
	numrowsinput = len(inputtable)
	numcolsoutput = len(outputtable[0])
	numrowsoutput = len(outputtable)

	#make a list of all the nodes, with their numbers, values, whether they are in the input or output, and features
	counter = 0
	#before making most nodes, we get the nodes representing the end
	endtabnode = 0
	resnodes.append((1,False,False,(0,0,0,1,0,0)))
	counter+=1
	endrownode = 1
	resnodes.append((0,False,False,(0,0,1,0,0,0)))
	counter+=1
	endtabnode = 2
	resnodes.append((1,False,False,(0,0,0,1,0,0)))
	counter+=1

	for val in inputtable[0]: #column names of the input
		ifnum1 = isnum(val)
		resnodes.append((counter,val,True,(0,1,0,0,0,ifnum1)))
		counter+=1
	for row in inputtable[1:]: #rest of the cells of the input
		for val in row:
			ifnum1 = isnum(val)
			resnodes.append((counter,val,True,(0,1,0,0,0,ifnum1)))
			counter += 1
	numnodesininput = counter #column names of the output
	for val in outputtable[0]:
		ifnum1 = isnum(val)
		resnodes.append((counter,val,False,(0,1,0,0,0,ifnum1)))
		counter += 1
	for row in outputtable[1:]: #rest of the cells of the output
		for val in row:
			ifnum1 = isnum(val)
			resnodes.append((counter,val,False,(0,1,0,0,0,ifnum1)))
			counter +=1

	#for each node, we look at the nodes previous to it, find all matches
	#edge features are (equality, horizontal-adjacency, vertial-adjacency, other)
	#the universal nodes are end of row and end of column which are 0 and 1, we want to ignore them
	tmpnodes = resnodes
	resnodes = resnodes[3:]
	for i in range(len(resnodes)):
		(nodeid,nodeval,inputq,nodefeatures) = resnodes[i]
		newedges = []
		for prevnode in resnodes[:i]:
			(nid,nval,ninputq,nfeat) = prevnode
			#check for equality, which is the first kind of edge
			if (nval == nodeval):
				newedges.append((nodeid,nid,(1,0,0,0)))
		nrows = numrowsinput if inputq else numrowsoutput
		ncols = numcolsinput if inputq else numcolsoutput
		if i > ncols:
			#check for the cell directly above this one 
			(nid,nval,ninputq,nfeat) = resnodes[i-ncols]
			if ninputq == inputq:
				newedges.append((nodeid,nid,(0,0,1,0)))
		if inputq:
			#check for the input table end
			if i == numnodesininput:
				newedges.append((nodeid,endtabnode,(0,0,1,1)))
			#check for ends of rows
			elif (i+1) % ncols == 0:
				newedges.append((nodeid,endrownode,(0,0,1,1)))
			#find the cell to the left if it exists
			if i > 0 and i % ncols != 0:
				(nid,nval,ninputq,nfeat) = resnodes[i-1]
				if ninputq == inputq:
					newedges.append((nodeid,nid,(0,1,0,0)))
		else:
			#for the purpose of checking column alignment we need to ignore the input table which could have a different column size
			adjustedi = i-numnodesininput
			#check for the final node of the output
			if i == len(resnodes)-1:
				newedges.append((nodeid,endtabnode,(0,0,1,1)))
			#check for ends of rows
			elif (adjustedi+1) % ncols == 0:
				newedges.append((nodeid,endrownode,(0,0,1,1)))
			#find the cell to the left if it exists
			if adjustedi > 0 and adjustedi % ncols != 0:
				(nid,nval,ninputq,nfeat) = resnodes[i-1]
				if ninputq == inputq:
					newedges.append((nodeid,nid,(0,1,0,0)))
		formattededges = [(i,source,dest,feat) for i,(source,dest,feat) in enumerate(newedges)]
		resedges += formattededges
	resnodes = tmpnodes
	#these nodes declare a table to be an input or an output respectively
	#they are also a link between input and output tables that is standardized in its position 
	resnodes.append((counter,False,False,(0,0,0,0,0,1)))
	resnodes.append((counter+1,False,False,(0,0,0,0,0,1)))
	edgecounter = len(resedges)
	resedges.append((edgecounter,counter,3,(0,0,0,1)))
	resedges.append((edgecounter+1,counter+1,numnodesininput,(0,0,0,1)))
	resedges.append((edgecounter+2,counter,counter+1,(0,0,0,1)))
	with open(trainfolder+"node.csv",'w') as f:
		f.write("idx,a,b,c,d,e,ff\n")
		for (ix,(idx,_,_,(a,b,c,d,e,ff))) in enumerate(resnodes):
			node_feat_line = ",".join([str(ss) for ss in [ix,a,b,c,d,e,ff]])
			f.write(node_feat_line+"\n")
	with open(trainfolder+"edge.csv",'w') as f:
		f.write("src,dst,a,b,c,d\n")
		for (_,src,dest,(a,b,c,d)) in resedges:
			f.write(str(src)+","+str(dest)+","+sfl(a)+","+sfl(b)+","+sfl(c)+","+sfl(d)+"\n")

def main():
	folders = next(os.walk('.'))[1]
	for folder in folders:
		convert_graphs(folder)