from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, round as spark_round
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.appName("RiskScoring").master("local[*]").getOrCreate()

# Read CSV
df = spark.read.csv("transactions.csv", header=True, inferSchema=True)

# Risk UDF (same as before)
def compute_risk(amount, merchant, label):
    if label == 1:
        return min(amount / 5, 100)
    else:
        return min(amount / 50, 30)
risk_udf = udf(compute_risk, DoubleType())

df_risk = df.withColumn("risk_score", spark_round(risk_udf(col("amount"), col("merchant"), col("risk_label")), 2))

# Convert to Pandas and save as CSV (avoids Hadoop on Windows)
pdf = df_risk.toPandas()
pdf.to_csv("risk_results.csv", index=False)
print("✅ Risk scores saved to risk_results.csv")

spark.stop()