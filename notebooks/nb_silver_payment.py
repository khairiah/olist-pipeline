#!/usr/bin/env python
# coding: utf-8

# ## nb_silver_payment
# 
# New notebook

# In[1]:


from pyspark.sql import SparkSession
from pyspark.sql import functions as F 


# In[ ]:


# ============================================
# PART 2: CLEAN PAYMENTS
# ============================================

# Load existing table
payments_df = spark.read.table("br_order_payments")
payments_df.printSchema()
payments_df.show(5)
paymentCount = payments_df.count()
print("Total payments:", paymentCount)


# In[ ]:


# ---- Cleaning ----
payments_df = payments_df.dropDuplicates()
payments_df = payments_df.filter(F.col("payment_value") >= 0)

valid_types = ["credit_card", "boleto", "voucher", "debit_card", "not_defined"]
payments_df = payments_df.withColumn(
    "payment_type",
    F.when(F.col("payment_type").isin(valid_types), F.col("payment_type")).otherwise("other")
)

# capitalize payment_type values and remove underscores
payments_df = payments_df.withColumn(
    "payment_type",
    F.initcap(F.regexp_replace(F.col("payment_type"), "_", " "))
)

# count payments by type
payments_df.groupBy("payment_type").count().orderBy("payment_type").show()

payments_df.printSchema()
payments_df.show(5)
print("✅ Payments cleaned rows:", payments_df.count())
print("Removed ", payments_df.count() - payments_df.count(), " rows with duplicate payment_id")


# In[ ]:


# ---- Save ----
payments_df.write.format("delta").mode("overwrite").saveAsTable("lh_silver_olist.sl_payment")
print("✅ Payments cleaned. Silver rows:", payments_df.count())

# ---- Check & EDA ----
payments_silver = spark.read.table("lh_silver_olist.sl_payment")
payments_silver.printSchema()
print("📊 Payments Silver count:", payments_silver.count())

# Show a sample of cleaned payments
payments_silver.select("payment_type","payment_installments","payment_value").show(5)

# Distribution of payment types
payments_silver.groupBy("payment_type").count().orderBy("count", ascending=False).show()


# In[ ]:


# ============================================
# STOP SPARK
# ============================================
spark.stop()
print("🔚 Spark session stopped.")

