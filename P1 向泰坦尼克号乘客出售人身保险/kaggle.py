#P1 向泰坦尼克号乘客出售人身保险
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection,linear_model,metrics

#import data
file=open("D:/工作/Personal/Data Analysis/P1 向泰坦尼克号乘客出售人身保险/data/train.csv")
data=pd.read_csv(file)
#data.head()#默认打印数据的前5行数据；data.tail(),默认打印数据的后5行数据
#选取特征矩阵X和目标因子y
y=data["Survived"]
X=data.drop(["Survived","PassengerId","Name","Ticket","Cabin"],axis=1)
#X.info()
#根据数字特征和分类特征分离特征矩阵X
num_feat=X.select_dtypes("number").columns.values
cat_feat=X.select_dtypes("object").columns.values
X_num=X[num_feat]
X_cat=X[cat_feat]
#数据归一化处理
#数值特征
X_num=(X_num-X_num.mean())/X_num.std()
X_num=X_num.fillna(X_num.mean())
#离散特征 one-hot编码
X_cat=pd.get_dummies(X_cat)
#归一化的数值特征和编码后的离散特征组合
X=pd.concat([X_num,X_cat],axis=1)
#准备训练数据集和测试数据集
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,random_state=0)

#选用模型，训练
model=linear_model.SGDClassifier(loss="log",max_iter=2000,random_state=0)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
metrics.accuracy_score(y_test,y_pred)#0.7937219730941704
###针对当前的数据（训练数据，测试数据）以及特征矩阵和目标因子，用SGD算法，分类的精度是0.79，距离目标还有差距，希望分类精度无限接近1

#验证迭代以上选取的迭代步数能否让模型收敛（取到最优值）
n_iter=np.linspace(1,3000)
scores=np.array([])
for n in n_iter:
    model=linear_model.SGDClassifier(loss="log",max_iter=n,random_state=0)
    model.fit(X_train,y_train)
    scores=np.append(scores,model.score(X_test,y_test))
plt.plot(n_iter,scores)
plt.xlabel("Number of iterations")
plt.ylabel("Accuracy")
plt.show()

accuracy=metrics.accuracy_score(y_test,y_pred)#0.7937219730941704
precision=metrics.precision_score(y_test,y_pred)#0.7261904761904762,for survived
#PR曲线,为避免分类误差影响，选用训练集数据作PR曲线
y_proba_train=model.predict_proba(X_train)[:,1]
#predict_proba返回的是一个 n(X_train行数) 行 k （分类类别）列的数组（668*2），
#第 i 行 第 j 列上的数值是模型预测 第 i 个预测样本为某个标签的概率，并且每一行的概率和为1。
#size=len(X_train)#668
p,r,t=metrics.precision_recall_curve(y_train,y_proba_train)
plt.plot(r,p)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

prt=np.array(list(zip(p,r,t)))
prt_df=pd.DataFrame(data=prt,columns=["Precision","Recall","Threshold"])
tail=prt_df.tail()
#在测试数据上，验证阈值是否合理
y_proba_test = model.predict_proba(X_test)[:, 1]
size=len(y_proba_test)
y_pred = (y_proba_test >= 0.97321).astype(int)
y_pred_count=np.count_nonzero(y_pred)#0

prt_df.tail(50)