import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\varun\OneDrive\Desktop\SML\Assignment3\data.csv")
mappings ={
    "Income":{"Low":0,"Medium":1,"High":2},
    "Student":{"No":0,"Yes":1},
    "Credit Rating":{"Fair":0,"Excellent":1},
    "Buy Computer":{"No":0,"Yes":1}
}
df.replace(mappings,inplace=True)
X = df.drop(columns=["Buy Computer"]).to_numpy()
y = df["Buy Computer"].to_numpy()
print(X)
def find_cutpoints(X,dimension=[0,1,2,3]):
    cutpoints=[]
    X=np.array(X)
    for i in range(0,X.shape[1]):
        sorted_X=sorted(X[:,i])
        s=set()
        for j in range(1,len(sorted_X)):
            if sorted_X[j - 1]!=sorted_X[j]:
                s.add((sorted_X[j - 1]+sorted_X[j])/2)
        cutpoints.append(sorted(list(s)))
    for i in range(4):
        if(i not in dimension):
            cutpoints[i]=[]
    return cutpoints
def GiniIndex(left_data,right_data):
    nyes=0
    for i in left_data:
        if(i==1):
            nyes+=1
    gleft=2*(nyes/len(left_data))*(1-(nyes)/len(left_data))
    nyes=0
    for i in right_data:
        if(i==1):
            nyes+=1
    gright=2*(nyes/len(right_data))*(1-(nyes)/len(right_data))
    gTotal=((len(left_data))/(len(left_data)+len(right_data)))*gleft
    gTotal+=((len(right_data))/(len(left_data)+len(right_data)))*gright
    return gTotal
class TreeNode:
    def __init__(self,dimension=None,cutpoint=None,left=None,right=None,prediction=None):
        self.dimension=dimension
        self.cutpoint=cutpoint
        self.left=left
        self.right=right
        self.prediction=prediction
def best_cut(X,y,cutpoints,used_cuts):
    dim=None
    cutpoint=None
    gini=float('inf')
    for d in range(X.shape[1]):
        for cut in cutpoints[d]:
            if(d,cut) in used_cuts:
                continue
            else:
                X_left=[]
                y_left=[]
                X_right=[]
                y_right=[]
                for i in range(X.shape[0]):
                    if(X[i][d]<=cut):
                        X_left.append(X[i])
                        y_left.append(y[i])
                    else:
                        X_right.append(X[i])
                        y_right.append(y[i])
                if(len(y_left)==0 or len(y_right)==0):
                    continue
                g=GiniIndex(y_left,y_right)
                if(g<gini):
                    dim=d
                    cutpoint=cut
                    gini=g
    return dim,cutpoint
def build_tree(X,y,cutpoints,max_depth,min_samples,used_cuts=[],depth=0,random_predictors=4):
    X=np.array(X)
    y=np.array(y)
    if(sum(y)==len(y)):
        return TreeNode(prediction=1)
    if(sum(y)==0):
        return TreeNode(prediction=0)
    if depth>max_depth or len(y)<=min_samples:
        nyes=0
        for i in y:
            if(i==1):
                nyes+=1
        if(nyes>len(y)-nyes):
            return TreeNode(prediction=1)
        return TreeNode(prediction=0)
    dimension,cutpoint=best_cut(X,y,cutpoints,used_cuts)
    # print(dimension,cutpoint)

    if dimension is None or cutpoint is None:
        nyes=sum(y)
        if(nyes>len(y)-nyes):
            prediction=1
        else:
            prediction=0
        return TreeNode(prediction=prediction)
    used_cuts.append((dimension,cutpoint))
    X_left=[]
    y_left=[]
    X_right=[]
    y_right=[]
    for i in range(X.shape[0]):
        if(X[i][dimension]<=cutpoint):
            X_left.append(X[i])
            y_left.append(y[i])
        else:
            X_right.append(X[i])
            y_right.append(y[i])
    random_dims=[0,1,2,3]
    if(random_predictors!=4):
        random_dims=np.random.choice(np.arange(4),size=2,replace=False)
    random_dims=list(random_dims)
    left_node=build_tree(X_left,y_left,find_cutpoints(X_left,random_dims),max_depth,min_samples,used_cuts,depth+1)
    right_node=build_tree(X_right,y_right,find_cutpoints(X_right,random_dims),max_depth,min_samples,used_cuts,depth+1)
    return TreeNode(dimension,cutpoint,left_node,right_node)
def test(x_test,tree):
    if(tree.prediction is None):
        if(x_test[tree.dimension]<=tree.cutpoint):
            return test(x_test,tree.left)
        else:
            return test(x_test,tree.right)
    return tree.prediction
def print_tree(tree, indent=""): 
    if tree is None:
        return
    if tree.prediction is not None:
        print(indent + f"Prediction: {tree.prediction}")
    else:
        print(indent + f"Feature {tree.dimension} <= {tree.cutpoint:.2f}")
    if tree.left:
        print(indent + "├── Left:")
        print_tree(tree.left, indent + "│   ")
    
    if tree.right:
        print(indent + "└── Right:")
        print_tree(tree.right, indent + "    ")
cutpoints=find_cutpoints(X)
used_splits=[]
Trees=[]
OOB_size=0
OOB_error = 0
for i in range(10):
    XBag=[]
    YBag=[]
    for j in range(len(X)):
        x = np.random.randint(len(X))
        XBag.append(X[x])
        YBag.append(y[x])
    XBag=np.array(XBag)
    YBag=np.array(YBag)
    cutpoints_bag = find_cutpoints(XBag)
    OOB_X=[]
    OOB_Y=[]
    for k in range(len(X)):
        if not any(np.array_equal(X[k], xb) for xb in XBag):
            OOB_X.append(X[k])
            OOB_Y.append(y[k])
            OOB_size+=1
    OOB_X=np.array(OOB_X)
    bag_tree=build_tree(XBag,YBag,cutpoints_bag,4,2)
    Trees.append(bag_tree)
    for k in range(len(OOB_X)):
        test_answer=test(OOB_X[k], bag_tree)
        OOB_error+=(OOB_Y[k]-test_answer)**2
print("OOB_error for bagging only ",OOB_error/OOB_size)
def test_bagging(X_test,Trees):
    y=[]
    for tree in Trees:
        y.append(test(X_test,tree))
    y=np.array(y)
    print(y)
    return np.bincount(y).argmax()
if(test_bagging([42,0,0,1],Trees)):
    print("Prediction for 42,0,0,1 is Yes")
else:
    print("Prediction for 42,0,0,1 is No")
Trees=[]
OOB_size=0
OOB_error = 0
for i in range(10):
    XBag=[]
    YBag=[]
    for j in range(len(X)):
        x = np.random.randint(len(X))
        XBag.append(X[x])
        YBag.append(y[x])
    XBag=np.array(XBag)
    YBag=np.array(YBag)
    random_dims=np.random.choice(np.arange(4),size=2,replace=False)
    random_dims=list(random_dims)
    cutpoints_bag=find_cutpoints(XBag,random_dims)
    OOB_X=[]
    OOB_Y=[]
    for k in range(len(X)):
        if not any(np.array_equal(X[k], xb) for xb in XBag):
            OOB_X.append(X[k])
            OOB_Y.append(y[k])
            OOB_size+=1
    OOB_X=np.array(OOB_X)
    bag_tree=build_tree(XBag,YBag,cutpoints_bag,4,2,random_predictors=2)
    Trees.append(bag_tree)
    for k in range(len(OOB_X)):
        test_answer=test(OOB_X[k], bag_tree)
        OOB_error+=(OOB_Y[k]-test_answer)**2
print("OOB_error for RF is ",OOB_error/OOB_size)
if(test_bagging([42,0,0,1],Trees)):
    print("Prediction for 42,0,0,1 is Yes")
else:
    print("Prediction for 42,0,0,1 is No")
