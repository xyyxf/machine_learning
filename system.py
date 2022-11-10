import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import csv

import random
import paddle

#警告过滤器控制是否发出警告消息
warnings.filterwarnings('ignore')


plt.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体  
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示问题`

#format字符串格式化方法,train.shape[0]表示train矩阵的行数，shape[1]表示矩阵的列数
train=pd.read_csv('C:/Users/86191/Downloads/train.csv',encoding='utf-8')
print("在训练集中，共有{}条数据，其中每条数据有{}个特征".format(train.shape[0], train.shape[1]))
test  = pd.read_csv('C:/Users/86191/Downloads/evaluation_public.csv', encoding='utf-8')
print("在测试集中，共有{}条数据，其中每条数据有{}个特征".format(test.shape[0], test.shape[1]))

#行拼接
df = pd.concat([train, test])
#打印
df.info()

#统计每类数据对应的平均风险概率
for f in ['user_name', 'department', 'ip_transform', 'device_num_transform', 'browser_version', 'browser', 'os_type', 'os_version', 'ip_type',
    'op_city', 'log_system_transform', 'url']:

    #mean()求平均值，unique()去重
    for v in df[f].unique():
        print(f, v, df[df[f] == v]['is_risk'].mean())
    print('*'*50)

# 查询包含Nan值的行
print(df[df.isnull().T.any()])

print(df.describe())

df.info()

#日期格式转换
df['op_datetime'] = pd.to_datetime(df['op_datetime'])
#获取日期数据中的年月日，星期数等Series.dt()方法
df['hour'] = df['op_datetime'].dt.hour
df['weekday'] = df['op_datetime'].dt.weekday
df['year'] = df['op_datetime'].dt.year
df['month'] = df['op_datetime'].dt.month
df['day'] = df['op_datetime'].dt.day

#去掉不用的特征
#删除列名为‘ ’，inplace=True表示删除原始数据
df.drop(columns = 'op_datetime', inplace=True)
df.drop(columns = 'op_month', inplace=True)

#数据编码
from sklearn.preprocessing import LabelEncoder

for feat in ['user_name', 'department', 'ip_transform', 'device_num_transform', 'browser_version','log_system_transform', 'op_city','browser', 'os_type', 'os_version', 'ip_type',
     'url']:
    #训练LabelEncoder,将df的列名编码为0,1,2...
    lab = LabelEncoder()
    df[feat] = lab.fit_transform(df[feat])

#填充空值，df.colums返回df的所有列标签，df[feat].median()返回中位数
#df.fillna主要用来对缺失值进行填充，可以选择填充具体的数字，或者选择临近填充。
for feat in [item for item in df.columns if item != 'is_risk']:
    df[feat].fillna(df[feat].median(), inplace=True)

#Pandas中的df.corr()函数的作用是返回列与列之间的相关系数。
print(df.corr()['is_risk'])

features = [item for item in df.columns if item != 'is_risk']
#df[ ] : 只取某列的值，是键值， 返回Series类型
#df[[ ]] :取完全的某列，是表格，返回DataFrame类型
#df. :只取某列的值，是键值， 返回Series类型
#df.reset_index( )函数：重置索引直接生成一个新DataFrame或Series,df.isnull()直接查看缺失值情况，是否为缺失值分别以True和False标识
traindata = df[~df['is_risk'].isnull()].reset_index(drop=True)
testdata = df[df['is_risk'].isnull()].reset_index(drop=True)

#reshape改变数组形状，reshape(-1,1)将数组转换为一位数组
data_X = traindata[features].values[:40000]
data_Y = traindata['is_risk'].values[:40000].astype(int).reshape(-1, 1)
data_X_test = traindata[features].values[40000:]
data_Y_test = traindata['is_risk'].values[40000:].astype(int).reshape(-1, 1)
testdata = testdata[features].values

# 归一化
#Min-Max归一化: x = (x - x的最小值) / (x的最大值 - x的最小值)    
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
data_X = mm.fit_transform(data_X)
data_X_test = mm.transform(data_X_test)
testdata = mm.transform(testdata)

print(data_X.shape)
print(data_X_test.shape)
print(testdata.shape)

#模型组网，SubClass组网，针对一些比较复杂的网络结构，在_init_构造函数中进行组网Layer的声明，在forward中使用声明的Layer变量进行前向运行
import random
import paddle
seed = 1234
# 设置随机种子 固定结果
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    paddle.seed(seed)

set_seed(seed)

import paddle
import paddle.nn as nn

# 定义动态图
#super(子类，self).父类的方法(self, 参数)  # super中的self也必须写，但父类方法中不能写self
#在训练时nn.Dropout会以概率p随机的丢弃一些神经元
#Linear其实就是对输入 X 执行了一个线性变化
#sigmoid是激活函数的一种，它会将样本值映射到0到1之间。
#Dropout，简单的说，就是我们在前向传播的时候，让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征。
class Classification(nn.Layer):
    def __init__(self):
        super(Classification, self).__init__()
        self.drop = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(19, 32) #输入的特征数,输出特征数
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
        self.fc4 = nn.Linear(2, 1)
        self.sig = nn.Sigmoid()
    
    # 网络的前向计算函数
    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x) 
        x = self.drop(x)
        x = self.fc4(x)
        pred  = self.sig(x)
        return pred

# 定义绘制训练过程的损失值变化趋势的方法draw_train_process
train_nums = []
train_costs = []
def draw_train_process(iters,train_costs):
    title="training cost"
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("cost", fontsize=14)
    plt.plot(iters, train_costs,color='red',label='training cost') 
    plt.grid()
    plt.show()

import paddle.nn.functional as F
y_preds = []
labels_list = []
BATCH_SIZE =40
train_data = data_X
train_data_y = data_Y
test_data = data_X_test
test_data_y = data_Y_test
def train(model):
    print('start training ... ')
    # 开启模型训练模式
    model.train()
    EPOCH_NUM = 9
    train_num = 0
    #该接口使用 cosine annealing 的策略来动态调整学习率。
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.0025, T_max=int(traindata.shape[0]/BATCH_SIZE*EPOCH_NUM), verbose=False)
    optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
    #optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    for epoch_id in range(EPOCH_NUM):
        # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
        np.random.shuffle(train_data)
        # 将训练数据进行拆分，每个batch包含8条数据
        mini_batches = [np.append(train_data[k: k+BATCH_SIZE], train_data_y[k: k+BATCH_SIZE], axis = 1) for k in range(0, len(train_data), BATCH_SIZE)]
        for batch_id, data in enumerate(mini_batches):
            features_np = np.array(data[:, :19], np.float32)
            labels_np = np.array(data[:, -1:], np.float32)
            #paddle.to_tensor只是将其他数据类型转化为tensor类型，便于构建计算图
            features = paddle.to_tensor(features_np)
            labels = paddle.to_tensor(labels_np)
            #前向计算
            y_pred = model(features)
            #使用二分类损失
            cost = F.binary_cross_entropy(y_pred, labels)
            train_cost = cost.numpy()
            #反向传播
            cost.backward()
            #最小化loss，更新参数
            optimizer.step()
            # 清除梯度
            optimizer.clear_grad()
            if batch_id % 1000 == 0 and epoch_id % 1 == 0:
                print("Pass:%d,Cost:%0.5f"%(epoch_id, train_cost))

            train_num = train_num + BATCH_SIZE
            train_nums.append(train_num)
            train_costs.append(train_cost)
#在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。
def predict(model):
    print('start evaluating ... ')
    model.eval()
    outputs = []
    mini_batches = [np.append(test_data[k: k+BATCH_SIZE], test_data_y[k: k+BATCH_SIZE], axis = 1) for k in range(0, len(test_data), BATCH_SIZE)]
    for data in mini_batches:
        features_np = np.array(data[:, :19], np.float32)
        features = paddle.to_tensor(features_np)
        pred = model(features)
        #out = paddle.argmax(pred, axis=1)
        outputs.extend(pred.numpy())
    return outputs
model = Classification()
train(model)
draw_train_process(train_nums, train_costs)

#模型评估
from sklearn import metrics 
from sklearn.metrics import roc_auc_score,roc_curve, auc
outputs = predict(model)
test_data_y = test_data_y.reshape(-1, )
outputs = np.array(outputs)

print('roc_auc_score', roc_auc_score(test_data_y,outputs))

fpr, tpr, threshold = roc_curve(test_data_y,outputs)   
roc_auc = auc(fpr,tpr)
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

predict_result=[]
for infer_feature in testdata:
    #print(infer_feature.shape)
    infer_feature = paddle.to_tensor(np.array(infer_feature, dtype='float32'))
    result = model(infer_feature)
    # print(result)
    predict_result.append(result.numpy()[0])
    
import os
import pandas as pd

id_list = [item for item in range(0, 25710)]
label_list = []
csv_file = 'submission.csv'
for item in range(len(id_list)):
    label = format(predict_result[item],'.1f')
    label_list.append(label)

data = {'id':id_list, 'is_risk':label_list}
df = pd.DataFrame(data)
df.to_csv(csv_file, index=False, encoding='utf8')
