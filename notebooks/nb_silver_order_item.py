#!/usr/bin/env python
# coding: utf-8

# ## nb_silver_order_item
# 
# New notebook

# In[26]:


from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType,
    IntegerType, DoubleType, TimestampType
)
import os
import matplotlib.pyplot as plt
base_path = "abfss://f4041c98-2c4f-44a4-82ec-ab26f8d5d0b3@onelake.dfs.fabric.microsoft.com/d3c25d34-1652-45f9-b4f3-58b4aedb4efa/Files/bronze/"
spark = SparkSession.builder.appName("OrderItemsSilverPipeline").getOrCreate()


# In[27]:


# order_items_schema = StructType([
#     StructField("order_id", StringType(), False),
#     StructField("order_item_id", IntegerType(), False),
#     StructField("product_id", StringType(), False),
#     StructField("seller_id", StringType(), False),
#     StructField("shipping_limit_date", TimestampType(), True),
#     StructField("price", DoubleType(), True),
#     StructField("freight_value", DoubleType(), True),
# ])


# In[28]:


# order_items_path = os.path.join(base_path, "olist_order_items_dataset.csv")

# order_items_df = spark.read.csv(
#     order_items_path,
#     header=True,
#     schema=order_items_schema
# ).repartition(20)

order_items_df = spark.read.table("br_order_items")

print("Order Items schema:")
order_items_df.printSchema()
order_items_df.show(5, truncate=False)


# In[29]:


# added to compare row counts before cleaning
order_items_df.count()


# In[30]:


quarantine_df = order_items_df.filter(
    (F.col("order_id").isNull()) |
    (F.col("order_item_id").isNull()) |
    (F.col("product_id").isNull()) |
    (F.col("seller_id").isNull()) |
    (F.col("price").isNull() | (F.col("price") < 0)) |
    (F.col("freight_value").isNull() | (F.col("freight_value") < 0)) |
    (F.col("shipping_limit_date").isNull())
)

clean_order_items_df = order_items_df.subtract(quarantine_df)

print("Quarantined rows:", quarantine_df.count())
print("Clean rows:", clean_order_items_df.count())


# In[31]:


clean_order_items_df = clean_order_items_df.withColumn(
    "total_item_value",
    F.col("price") + F.col("freight_value")
)


# In[32]:


# Added: Write clean + quarantine to Silver Lakehouse tables
clean_order_items_df.write.format("delta").mode("overwrite").saveAsTable("lh_silver_olist.sl_order_item")
quarantine_df.write.format("delta").mode("overwrite").save("Tables/qr_order_items")

print("✅ Silver tables saved: sl_order_items, sl_order_items_quarantine")


# In[33]:


# silver_path = "file://" + os.path.join(base_path, "silver_olist_order_items")
# quarantine_path = "file://" + os.path.join(base_path, "quarantine_olist_order_items")

# clean_order_items_df.write.mode("overwrite").parquet(silver_path)
# quarantine_df.write.mode("overwrite").parquet(quarantine_path)

# print(f"Silver dataset saved to: {silver_path}")
# print(f"Quarantine dataset saved to: {quarantine_path}")


# In[34]:


top_products = (
    clean_order_items_df.groupBy("product_id")
    .agg(F.sum("total_item_value").alias("total_sales"))
    .orderBy(F.desc("total_sales"))
    .limit(10)
)

print("Top 10 products by sales:")
top_products.show(truncate=False)


# In[35]:


top_products_pd = top_products.toPandas()

plt.figure(figsize=(12,6))
plt.bar(top_products_pd["product_id"], top_products_pd["total_sales"])
plt.title("Top 10 Products by Total Sales Value")
plt.xlabel("Product ID")
plt.ylabel("Total Sales Value")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# In[36]:


sample_pd = clean_order_items_df.select("price", "freight_value").sample(False, 0.01, seed=42).toPandas()

plt.figure(figsize=(8,6))
plt.scatter(sample_pd["price"], sample_pd["freight_value"], alpha=0.5)
plt.title("Price vs Freight Value (Sampled)")
plt.xlabel("Price")
plt.ylabel("Freight Value")
plt.tight_layout()
plt.show()


# In[37]:


items_sold = (
    clean_order_items_df.groupBy("product_id")
    .agg(F.count("order_item_id").alias("total_items_sold"))
    .orderBy(F.desc("total_items_sold"))
    .limit(10)  # top 10 products by quantity
)

print("Top 10 products by total items sold:")
items_sold.show(truncate=False)

# Convert to pandas for plotting
items_sold_pd = items_sold.toPandas()

plt.figure(figsize=(12,6))
plt.bar(items_sold_pd["product_id"], items_sold_pd["total_items_sold"], color='skyblue')
plt.title("Top 10 Products by Total Items Sold")
plt.xlabel("Product ID")
plt.ylabel("Total Items Sold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

total_items_sold = clean_order_items_df.count()
print(f"Total items sold (all products): {total_items_sold}")


# 
