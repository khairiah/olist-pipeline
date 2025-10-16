#!/usr/bin/env python
# coding: utf-8

# ## nb_injest_csv_to_bronze_olist
# 
# New notebook

# In[2]:


from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("IngestCSVtoLakehouseTables").getOrCreate()

# Define base paths
raw_path = "abfss://f4041c98-2c4f-44a4-82ec-ab26f8d5d0b3@onelake.dfs.fabric.microsoft.com/71bee0ad-c68d-43d4-b22d-e8ba18376bb1/Files/raw/"
destination_path = "abfss://f4041c98-2c4f-44a4-82ec-ab26f8d5d0b3@onelake.dfs.fabric.microsoft.com/71bee0ad-c68d-43d4-b22d-e8ba18376bb1/Tables/"

# Define datasets and renamed tables
datasets = {
    "olist_customers_dataset":"customers",
    "olist_geolocation_dataset":"geolocation",
    "olist_order_items_dataset":"order_items",
    "olist_order_payments_dataset":"order_payments",
    "olist_order_reviews_dataset":"order_reviews",
    "olist_orders_dataset": "orders",
    "olist_products_dataset": "products",
    "olist_sellers_dataset": "sellers",
    "product_category_name_translation": "product_category_name_translation"
}

# Loop through each dataset and ingest
for src_name, table_name in datasets.items():
    csv_path = f"{raw_path}{src_name}.csv"
    print(f"Reading {csv_path}")

    # Read CSV with inferred schema
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(csv_path)
    )

    # Show schema (optional)
    df.printSchema()

    # Write to lakehouse table (overwrite mode)
    table_name = "br_" + table_name
    dest_table_path = f"{destination_path}{table_name}"
    print(f"Writing to {dest_table_path}")
    
    (df.write
        .format("delta")
        .mode("overwrite")
        .save(dest_table_path)
    )

print("All tables ingested successfully.")

