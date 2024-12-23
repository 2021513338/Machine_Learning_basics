from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np
import time
# 初始化SparkSession
spark = SparkSession.builder \
    .appName("Random Forest with PySpark") \
    .getOrCreate()

# 加载数据
X_df = spark.read.csv('hdfs://master:9000/user/root/input/X_mus.csv', header=False, inferSchema=True)
y_df = spark.read.csv('hdfs://master:9000/user/root/input/y_mus.csv', header=False, inferSchema=True)

# 将y_df的列重命名为'label'，以便于后续操作
y_df = y_df.withColumnRenamed(y_df.columns[0], "label")

# 数据预处理
assembler = VectorAssembler(inputCols=X_df.columns, outputCol="features")
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures", min=0, max=1)

# 定义随机森林模型
rf = RandomForestClassifier(labelCol="label", featuresCol="scaledFeatures")

# 定义Pipeline
pipeline = Pipeline(stages=[assembler, scaler, rf])

# 交叉验证参数
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [64, 128, 256]) \
    .addGrid(rf.maxDepth, [4, 8, 16]) \
    .build()

# 定义交叉验证
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(labelCol="label"),
                          numFolds=10)

# 存储每次交叉验证的结果
accuracies = []
aucs = []

start_time = time.time()
# 执行100次10折交叉验证
for i in range(100):
    # 训练模型
    cvModel = crossval.fit(X_df.join(y_df, X_df.columns).withColumnRenamed("label", "label"))

    # 评估模型
    predictions = cvModel.transform(X_df.join(y_df, X_df.columns).withColumnRenamed("label", "label"))
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
    accuracies.append(accuracy)
    aucs.append(auc)
    # 打印进度
    print(f"Finished {i + 1}/{100} times training，accuracy：{accuracy:.4f}，AUC：{auc:.4f}")

end_time = time.time()
# 计算平均ACC、平均AUC、ACC的标准差和AUC的标准差
mean_accuracy = np.mean(accuracies)
mean_auc = np.mean(aucs)
std_accuracy = np.std(accuracies)
std_auc = np.std(aucs)
run_time = end_time - start_time
# 输出结果
print("Result of RF, 100 times 10 fold:")
print("-----------------------------------")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean AUC: {mean_auc:.4f}")
print(f"Std Accuracy: {std_accuracy:.4f}")
print(f"Std AUC: {std_auc:.4f}")
print(f"Total time: {run_time}")
print("-----------------------------------")
# 停止SparkSession
spark.stop()

