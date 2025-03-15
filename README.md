GZHU grade 15 Machine learning experiments personal backup
====

This repository is created for personal backup.  
Highly NOT recommand using codes in this repo to submit homework.  

**In all experiments, this requirement must be followed: 允许使用numpy库，需实现详细实验步骤，不允许直接调用scikit-learn中回归、分类等高层API**  


experiment 1  
---
基于California Housing Prices数据集，完成关于房价预测的线性回归模型训练、测试与评估。  
1. 准备数据集并认识数据  
下载[California Housing Prices](https://www.kaggle.com/camnugent/california-housing-prices)数据集  
了解数据集各个维度特征及预测值的含义   
2. 探索数据并预处理数据  
观察数据集各个维度特征及预测值的数值类型与分布  
预处理各维度特征（如将类别型维度ocean_proximity转换为one-hot形式的数值数据），参考：https://blog.csdn.net/SanyHo/article/details/105304292  
3. 求解模型参数  
编程实现线性回归模型的闭合形式参数求解  
编程实现线性回归模型的梯度下降参数优化  
4. 测试和评估模型  
在测试数据集上计算所训练模型的R2指标  

experiment 2
---
基于IRIS鸢尾花数据集，完成样本是否属于维吉尼亚鸢尾(Iris-Virginica)的逻辑回归分类、朴素贝叶斯模型训练、测试与评估。  
1. 准备数据集并认识数据  
下载[IRIS数据集](https://archive.ics.uci.edu/ml/datasets/iris)  
了解数据集各个维度特征及预测值的含义  
2. 探索数据并预处理数据  
观察数据集各个维度特征及预测值的数值类型与分布  
3. 训练模型  
编程实现训练数据集上逻辑回归模型的梯度下降参数求解、朴素贝叶斯参数统计  
4. 测试和评估模型  
在测试数据集上计算所训练模型的准确率、F1-score等指标

experiment 3
---
基于Seeds数据集，完成聚类分析。  
1 准备数据集并认识数据  
下载[Seeds数据集](https://archive.ics.uci.edu/dataset/236/seeds)
2 了解Seeds数据集的构成方式，包括210个样本，前7维是特征，最后1维是标签。对数据集进行整理，使每个样本包括7个特征和1以类别标签。挑选7维特征做聚类研究。  
3 求解聚类中心  
编程实现k-means聚类、层次聚类  
4 测试和评估模型  

experiment 4
---
Breast Cancer Wisconsin (Diagnostic) Data Set 乳腺癌威斯康星州（诊断）数据集, 设计决策树分类、随机森林分类模型训练、测试与评估，预测癌症是良性还是恶性。
1. 准备数据集并认识数据  
[下载数据集](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)，了解数据集各个维度特征及预测值的含义。  
属性信息：  

| 序号 | 项目              | 描述                                                     |
|------|-------------------|----------------------------------------------------------|
| 1    | ID号              | 数据的唯一标识符                                          |
| 2    | 诊断              | M = 恶性，B = 良性                                        |
| 3-32 | 十个实值特征      | 每个细胞核的十个特征，包含以下项目：                     |
| a    | 半径              | 从中心到周边点的距离的平均值                              |
| b    | 纹理              | 灰度值的标准偏差                                          |
| c    | 周边              | 周围的细节（如：轮廓的边缘信息）                          |
| d    | 面积              | 细胞核的面积                                              |
| e    | 平滑度            | 半径长度的局部变化                                        |
| f    | 紧密度            | 周长^2 / 面积 - 1.0                                       |
| g    | 凹度              | 轮廓的凹入部分的严重程度                                  |
| h    | 凹点              | 轮廓的凹入部分的数量                                      |
| i    | 对称性            | 细胞核的对称性                                            |
| j    | 分形维数          | “海岸线近似”-1                                            |

为每个图像计算这些特征的平均值，标准误差和“最差”或最大（三个最大值的平均值），从而得到30个特征。例如，字段3是平均半径，字段13是半径SE，字段23是最差半径。所有功能值都用四个有效数字重新编码。  

2. 探索数据并预处理数据  
观察数据集各个维度特征及预测值的数值类型与分布  
3. 训练模型  
编程实现训练数据集上贪心决策树、随机森林的构建  
4. 测试和评估模型  
在测试数据集上计算所训练模型的准确率、AUC等指标  

