#!/usr/bin/env python
# coding: utf-8

# ## nb_silver_order
# 
# New notebook

# In[6]:


# If needed
# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %pip install pyspark
# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %pip install -q findspark


# In[7]:


# Core PySpark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

# Spark bootstrap
import findspark

# Selected functions and types
from pyspark.sql.functions import col, udf, regexp_replace, when, count, datediff
from pyspark.sql.types import StructType, StructField, StringType,TimestampType

from datetime import datetime

# Standard Python libraries
import re
import unicodedata
import os

import matplotlib.pyplot as plt


# # Start a Spark Session

# In[8]:


# locate Spark installation and initialize Spark in Python
findspark.init()

# Import and initiate a SparkSession
spark = SparkSession.builder.appName("BronzeToSilver").getOrCreate()


# # Orders Dataset

# 1. order_id = Generated ID for each order in the dataset (this feature is the primary key for dataset merging)
# 2. customer_id = Customer's ID for each order
# 3. order_status = Order status for the corresponding order
# 4. order_purchase_timestamp = Time when customer places an order
# 5. order_approved_at = Time when order is approved by the seller and customer
# 6. order_delivered_carrier_date = Time when order is picked up by the delivery carrier
# 7. order_delivered_customer_date = Time when order is received by the customer
# 8. order_estimated_delivery_date = Time estimation of order delivery

# In[9]:


# Defining schema
# For production applications, it's best to explicitly define the schema and avoid inference. You don't want to rely on fragile inference rules that may get updated and cause unanticipated changes in your code.

# order_id, customer_id, order_status, order_purchase_timestamp, order_estimated_delivery_date must not be null.
# nullable=true because of CSV ingestion, types will be enforced once saved to Parquet/Delta format

orders_schema = StructType([
    StructField("order_id", StringType(), False),
    StructField("customer_id", StringType(), False), #foreign key link to customers
    StructField("order_status", StringType(), False), #lifecycle state

    StructField("order_purchase_timestamp", TimestampType(), False),
    StructField("order_approved_at", TimestampType(), True),
    StructField("order_delivered_carrier_date", TimestampType(), True),
    StructField("order_delivered_customer_date", TimestampType(), True),
    StructField("order_estimated_delivery_date", TimestampType(), False)
])


# In[10]:


# Partitioning - Spark allows explicit control over partition size and number during data ingestion to help speed up subsequent operations
# Splits DataFrame into 20 partitions across Spark cluster. Since Spark processes data in chunks, having more partitions can improve parallelism, balance workload across cluster and speed up transformations and writes.

# orders_df = spark.read.format("delta").load("Tables/br_orders")
orders_df = spark.read.table("br_orders")

orders_df.printSchema()


# In[11]:


# Check data types
display(orders_df.dtypes)
#display(orders_df.columns)|


# ## Cleaning

# In[12]:


# Select sample rows from date-related fields to inspect actual values
timestamp_cols = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

orders_df.select(*timestamp_cols).show(5, truncate=False)


# In[13]:


# Extract time portion of order_estimated_delivery_date and see if it's always midnight
orders_df.filter(
    (F.hour("order_estimated_delivery_date") != 0) |
    (F.minute("order_estimated_delivery_date") != 0) |
    (F.second("order_estimated_delivery_date") != 0)
).count()


# All order_estimated_delivery_date values have 00:00:00 time component. All other timestamp fields contain actual time information (hours/minutes/seconds). For consistency sake and avoid type errors, this will still be a timestamp field.
# 

# ### Completeness Check

# In[14]:


orders_df_null_check = orders_df.filter(
        F.col("order_id").isNotNull() &
        F.col("customer_id").isNotNull() &
        F.col("order_status").isNotNull() &
        F.col("order_purchase_timestamp").isNotNull()
    )


# In[15]:


# Show how many nulls per columns

missing_summary = orders_df.select([
    (col(c).isNull().cast("int")).alias(c) for c in orders_df.columns
]).groupBy().sum()

missing_summary.show()


# In[16]:


def check_completeness(orders_df):
    """Check completeness for Silver layer quality gates"""

    print("🔍 CHECKING DATA COMPLETENESS...")

    total_records = orders_df.count()
    all_passed = True

    # CRITICAL FIELDS - Must be 100% complete
    critical_fields = ["order_id", "customer_id", "order_status", "order_purchase_timestamp"]

    print("🚨 CRITICAL FIELDS:")
    for field in critical_fields:
        null_count = orders_df.filter(F.col(field).isNull()).count()
        empty_count = orders_df.filter(F.col(field) == "").count()
        missing_total = null_count + empty_count

        completeness = (total_records - missing_total) / total_records * 100

        if completeness < 100.0:
            print(f"❌ {field}: {completeness:.2f}% ({missing_total} missing)")
            all_passed = False
        else:
            print(f"✅ {field}: 100% complete")

    # CONDITIONAL FIELDS - Document NULL patterns
    print("\n📊 CONDITIONAL FIELDS:")
    conditional_fields = ["order_approved_at", "order_delivered_carrier_date",
                          "order_delivered_customer_date"]

    for field in conditional_fields:
        null_count = orders_df.filter(F.col(field).isNull()).count()
        completeness = (total_records - null_count) / total_records * 100
        print(f"📊 {field}: {completeness:.1f}% complete ({null_count} NULL)")

        # Show NULL distribution by status
        null_by_status = orders_df.filter(F.col(field).isNull()).groupBy("order_status").count()
        null_by_status.show()

    return all_passed


check_completeness(orders_df)

if not check_completeness(orders_df):
  raise Exception("Critical completeness check failed")


# ## Check for duplicates

# In[17]:


dup_orders = (
    orders_df.groupBy("order_id")
             .count()
             .filter(F.col("count") > 1)
             .count()
)

print("Duplicate order_id rows:", dup_orders)  # <-- just print the int

# Assert (fail if any duplicates)
if dup_orders > 0:
    raise ValueError(f"Found {dup_orders} duplicate order_id rows")


# ## Membership Constraints for Categorical Data

# In[18]:


# Status for all orders must be among the following:
allowed_status = {
    "approved",
    "canceled",
    "created",
    "delivered",
    "invoiced",
    "processing",
    "shipped",
    "unavailable"
}

invalid_count = (
    orders_df
    .filter(~F.col("order_status").isin(list(allowed_status)))
    .count()
)

print("Invalid order_status rows:", invalid_count)

# Assert: fail if anything outside the allowed set
assert invalid_count == 0, f"Found {invalid_count} rows with invalid order_status"


# ## Outlier Delivery Time

# Observed max delivery time = 210 days (~7 months)
# This is not invalid data, just an extreme late delivery.

# In[19]:


# if you see negatives or > 365 days, those are suspects

delivery_days = orders_df.withColumn(
    "delivery_days",
    datediff(col("order_delivered_customer_date"), col("order_purchase_timestamp"))
)

delivery_days.select("delivery_days").summary("min", "max", "mean").show()


# In[20]:


# Recompute delivery_days
delivery_days = orders_df.withColumn(
    "delivery_days",
    F.datediff(F.col("order_delivered_customer_date"), F.col("order_purchase_timestamp"))
)

# Find the max delivery_days
max_days = delivery_days.agg(F.max("delivery_days")).collect()[0][0]
print("Max delivery_days:", max_days)

# Show the rows with that max value
delivery_days.filter(F.col("delivery_days") == max_days) \
    .select("order_id", "customer_id", "order_status",
            "order_purchase_timestamp", "order_delivered_customer_date",
            "order_estimated_delivery_date", "delivery_days") \
    .show(truncate=False)


# ## Chronology Check 

# Missing-at-random for timestamp related columns. Need to validate timestamp validity against status.
# 
# E.g. If missing where it's expected (cancelled, unavailable) -> expected to have missing value.
# 
# These rules isolate rows that are internally inconsistent.Did not fix data here, we only flag & quarantine it.
# 
# Rows that trigger one or more conditions are sent to QUARANTINE with explicit reason(s). Remaining rows form the SILVER table.
# 
# This keeps the Silver layer clean, consistent, and trustworthy while preserving questionable rows for later investigation.

# In[22]:


# Carrier cannot pick up an item before a customer buys it (purchase date systematically generated)
# Possible explanation - data entry error, or system clock mismatch between seller and platform
carrier_date_before_purchase = (
    F.col("order_delivered_carrier_date") < F.col("order_purchase_timestamp")
)

# Item delivered to customer before carrier pickup
# Possible explanation - wrong timestamp recorded
delivered_date_before_carrier = (
    F.col("order_delivered_customer_date") < F.col("order_delivered_carrier_date")
)

# Status says delivered but no delivery date recorded
# Possible explanation - SLA reporting broken, system failed to log the date
delivered_status_missing_customer_date = (
    (F.col("order_status") == "delivered") & F.col("order_delivered_customer_date").isNull()
)

# Status is not delivered (e.g. cancelled) but has a delivery date
# Possible explanation - system logging error after customer received it.

not_delivered_status_has_customer_date = (
    (F.col("order_status") != "delivered") & F.col("order_delivered_customer_date").isNotNull()
)

# Order approved before purchase 
# Possible explanation - system clock issue or data entry error
approved_before_purchase = (
    F.col("order_approved_at") < F.col("order_purchase_timestamp")
)

# Carrier pickup before order approval (business process violation)
# Possible explanation - approval step bypassed or timestamp recording issue
carrier_before_approval = (
    F.col("order_delivered_carrier_date") < F.col("order_approved_at")
)

# Combine all quarantine conditions
quarantine_conditions = (
    carrier_date_before_purchase |
    delivered_date_before_carrier |
    delivered_status_missing_customer_date |
    not_delivered_status_has_customer_date |
    approved_before_purchase |
    carrier_before_approval
)

# Quarantine set with reasons
quarantine_df = (
    orders_df
    .filter(quarantine_conditions)
    .withColumn(
        "quarantine_reason",
        F.when(carrier_date_before_purchase, "carrier_date_before_purchase")
         .when(delivered_date_before_carrier, "customer_date_before_carrier")
         .when(delivered_status_missing_customer_date, "delivered_status_missing_customer_date")
         .when(not_delivered_status_has_customer_date, "not_delivered_status_has_customer_date")
         .when(approved_before_purchase, "approved_before_purchase")
         .when(carrier_before_approval, "carrier_before_approval")
    )
)

orders_df = orders_df.filter(~quarantine_conditions)

print("Silver rows:", orders_df.count())
print("Quarantine rows:", quarantine_df.count())

# Quarantine reason breakdown
quarantine_df.groupBy("quarantine_reason").count().show()


# In[23]:


# Count rows by quarantine reason
audit_df = (
    quarantine_df
    .groupBy("quarantine_reason")
    .count()
    .withColumnRenamed("count", "row_count")
    .withColumn("run_date", F.lit(datetime.now()))
    .withColumn("table_name", F.lit("orders"))
    .withColumn("total_checked", F.lit(orders_df.count() + quarantine_df.count()))
)

# Save/append to audit table
audit_df.write.mode("append").saveAsTable("dq_audit_orders")


# ## Persist

# In[24]:


orders_df.show(2)


# In[25]:


#  Convert timestamp columns to yyyy-MM-dd format in preparation for Gold  ===
date_cols = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for c in date_cols:
    orders_df = orders_df.withColumn(c, F.to_date(F.col(c)))

# Verify conversion
orders_df.select(date_cols).show(5, truncate=False)


# In[34]:


# capitalize order_status values
orders_df = orders_df.withColumn(
    "order_status",
    F.initcap(F.col("order_status"))
)

orders_df.show(2)


# In[35]:


# orders_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
# .save("Tables/sl_orders")
# quarantine_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
# .save("Tables/qr_orders")

orders_df.write.mode("overwrite").saveAsTable("lh_silver_olist.sl_order")
quarantine_df.write.mode("overwrite").saveAsTable("qr_orders")

