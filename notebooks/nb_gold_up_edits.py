#!/usr/bin/env python
# coding: utf-8

# ## nb_gold_up_edits
# 
# New notebook

# In[1]:


# PySpark imports
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    BooleanType
)
from datetime import timedelta,date
import holidays

from pyspark.sql.functions import row_number, col, coalesce, lit
from pyspark.sql.window import Window
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import broadcast

from pyspark.sql.functions import col

from pyspark.sql.functions import datediff, unix_timestamp, when, col, date_format



# Spark configuration. Tune based on data size and cluster resources. Only affects how Spark parallelizes shuffle operations (e.g. groupby, join, orderBy, row_number(), etc)
spark.conf.set("spark.sql.shuffle.partitions", "200")  # Tune based on data size and cluster resources


# # Load

# In[2]:


sl_order        = spark.read.table("lh_silver_olist.sl_order")
sl_order_item   = spark.read.table("lh_silver_olist.sl_order_item")
sl_customer     = spark.read.table("lh_silver_olist.sl_customer")
sl_seller       = spark.read.table("lh_silver_olist.sl_seller")
sl_geolocation   = spark.read.table("lh_silver_olist.sl_geolocation")
sl_product = spark.read.table("lh_silver_olist.sl_product")
sl_review = spark.read.table("lh_silver_olist.sl_review")
sl_payment = spark.read.table("lh_silver_olist.sl_payment")


# In[3]:


for df_name, df in {
    "orders": sl_order,
    "order_items": sl_order_item,
    "customers": sl_customer,
    "sellers": sl_seller,
    "geo": sl_geolocation,
    "products": sl_product,
    "reviews": sl_review,
    "payments": sl_payment
}.items():
    print(f"{df_name}: {df.count()} rows, {len(df.columns)} cols")


# # Dimension Tables
# 
# Core Principle: Dimensions before facts. 
# 
# Rule from Kimball methodology: Always build dimension tables before fact tables because fact tables depend on dimension surrogate keys.
# 
# Build independent dimensions first (no dependencies). 

# ## dim_date

# In[4]:


# Gold layer: build date dimension for analytics
# Grain: one row per calendar day
# Covers 2 years before and 3 years after the actual data range in fact_orders (to aid forecasting, what-if simulations). 
# If dim_date stops exactly at max(order_delivered_date), Power BI visuals for “next month” break or return blanks.

date_bounds = sl_order.agg(
    F.min("order_purchase_timestamp").alias("min_date"),
    F.max("order_delivered_customer_date").alias("max_date")
).first()

start_date = (date_bounds["min_date"] - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
end_date   = (date_bounds["max_date"] + timedelta(days=365 * 3)).strftime("%Y-%m-%d")

print(f"Building gd_dim_date from {start_date} to {end_date}")

# Generate continuous date sequence
dim_date = (
    spark.createDataFrame([(start_date, end_date)], ["start_date", "end_date"])
         .select(F.explode(F.sequence(F.to_date("start_date"), F.to_date("end_date"))).alias("date"))
)

# Derive date attributes
# In Kimball modelling, every dimension (even Date) must have a surrogate key, instead of relying on natural types like DATE.
dim_date = (
    dim_date
        .withColumn("date_key", F.date_format("date", "yyyyMMdd").cast(IntegerType()))
        .withColumn("year", F.year("date"))
        .withColumn("quarter", F.quarter("date"))
        .withColumn("month_number", F.month("date"))
        .withColumn("month_name", F.date_format("date", "MMMM"))
        .withColumn("week_of_year", F.weekofyear("date"))
        .withColumn("day_of_month", F.dayofmonth("date"))
        .withColumn("day_of_week_number", (((F.dayofweek("date") + 5) % 7) + 1).cast(IntegerType()))  # Monday=1..Sunday=7
        .withColumn("day_of_week_name", F.date_format("date", "EEEE"))
        .withColumn("is_weekend", F.when(F.dayofweek("date").isin(1, 7), True).otherwise(False).cast(BooleanType()))
)

# Add Brazil national holidays
br_holidays = holidays.Brazil(years=range(int(start_date[:4]), int(end_date[:4])))
holiday_list = [str(d) for d in br_holidays.keys()]

dim_date = dim_date.withColumn(
    "is_holiday",
    F.when(F.col("date").cast("string").isin(holiday_list), True).otherwise(False).cast(BooleanType())
)

# Column ordering
dim_date_final = dim_date.select(
    "date_key",
    "date",
    "year",
    "quarter",
    "month_number",
    "month_name",
    "week_of_year",
    "day_of_month",
    "day_of_week_number",
    "day_of_week_name",
    "is_weekend",
    "is_holiday"
)

dim_date_final.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("gd_dim_date")

print(f"✅ gd_dim_date created successfully with {dim_date_final.count()} rows.")


# ### Validation 

# In[5]:


# Sanity check - verifying colum names, date ranges
# check with spark SQL - spark.sql("SELECT * FROM gd_dim_date LIMIT 3").show(truncate=False)
display(dim_date_final.limit(5)) # dataframe UI display


# In[6]:


# Validate uniqueness of primary key (date_key) to ensure every calendar day is unique
dup_check = dim_date_final.groupBy("date_key").count().filter("count > 1").count()
assert dup_check == 0, f"Duplicate date_keys found: {dup_check}"
print("✅ All date_keys are unique.")


# In[7]:


# Row count check - Make sure the number of rows matches the expected day count

# Calculate expected number of days inclusive
expected_days = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1
actual_days = dim_date_final.count()

assert actual_days == expected_days, f"❌ Row count mismatch: expected {expected_days}, got {actual_days}"
print(f"✅ Row count check passed: {actual_days:,} rows (covers full date range).")


# In[8]:


# Completeness of date range - Check that the smallest and largest date match the expected range exactly
minmax = dim_date_final.select(F.min("date"), F.max("date")).first()
assert str(minmax[0]) == start_date, f"❌ Min date mismatch: expected {start_date}, got {minmax[0]}"
assert str(minmax[1]) == end_date, f"❌ Max date mismatch: expected {end_date}, got {minmax[1]}"
print(f"✅ Date range check passed ({start_date} → {end_date}).")


# In[9]:


# No nulls in critical columns - Confirm key columns are fully populated
null_counts = dim_date_final.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c)
    for c in ["date_key", "date", "year", "month_number", "day_of_week_number"]
]).collect()[0].asDict()

null_issues = {k:v for k,v in null_counts.items() if v > 0}
assert len(null_issues) == 0, f"❌ Nulls found in: {null_issues}"
print("✅ Null check passed (no missing key fields).")


# In[10]:


# Logical consistency - Making sure the derived fields align with actual date logic

# Spot-check: is_weekend should match day_of_week_number (Saturday=6, Sunday=7)
bad_weekend = dim_date_final.filter(
    (F.col("is_weekend") == True) &
    (~F.col("day_of_week_number").isin(6, 7))
).count()
assert bad_weekend == 0, f"❌ is_weekend inconsistency found in {bad_weekend} rows"
print("✅ Weekend logic check passed.")


# If you later implement automated data quality monitoring, you can:
# - Log these assertion results to a Silver_Quality_Log table.
# - Use Great Expectations or PyDeequ for declarative rule sets (e.g., "expect_column_values_to_be_unique('date_key')").
# - Add created_at / last_validated_at timestamps for audit.

# ## dim_product

# In[11]:


# sl_product.printSchema()

sl_product.limit(3).show()


# In[12]:


# BEST PRACTICE: Read only required columns for performance (column pruning)
sl_product = sl_product.select(
    "product_id",  # key for joining with facts
    "product_category_english",  # for slicing & filtering in dashboards

    # below could be useful if you want to test whether heavier items have higher late-delivery probability
    # or cluster or classify products (small/medium/large for courier optimization)
    "size_category",
    "weight_category"
)

# Generate surrogate keys using hash
# No shuffle required unlike row_number()
# While monotonically_increasing_id() generates unique IDs without shuffle, it is not deterministic across partitions and returned thousands of duplicate keys
# hash() 32-bit returns duplicates so xxhash64() is used. fast, stable, collision proof

dim_product = (
    sl_product
    .withColumn("product_key", F.abs(F.xxhash64("product_id")))
    .select(
        "product_key",
        "product_id",
        "product_category_english",
        "size_category",
        "weight_category"
    )
)

# BEST PRACTICE: Cache dimension tables if reused multiple times,
# Caching avoids recomputation when dimension is joined to multiple fact tables

dim_product.cache()  # Keep in memory for upcoming fact table joins

dim_product.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("gd_dim_product")

print(f"✅ gd_dim_product created: {dim_product.count()} rows")


# ### Validation

# In[13]:


total_rows = dim_product.count()


# In[14]:


# uniqueness of surrogate key

try:
    unique_keys = dim_product.select("product_key").distinct().count()
    total_rows = dim_product.count()

    assert unique_keys == total_rows, \
        f"Duplicate product_key detected: {total_rows - unique_keys} duplicates."

    print("✅ Surrogate key uniqueness check passed.")

except AssertionError as e:
    print(f"❌ Assertion failed: {e}")

    # === 2️⃣ Debug block: inspect duplicates ===
    dupes = (
        dim_product.groupBy("product_key")
        .agg(F.count("*").alias("cnt"))
        .filter("cnt > 1")
    )

    print(f"Duplicate keys found: {dupes.count()}")
    dupes.show(10, truncate=False)

    # Join back to see which product_ids are affected
    dupe_details = (
        dim_product.join(dupes, on="product_key", how="inner")
        .orderBy("product_key")
    )
    dupe_details.show(20, truncate=False)


# In[15]:


# No missing product_id (business key)
missing_product_id = dim_product.filter("product_id IS NULL").count()
assert missing_product_id == 0, f"Missing product_id in {missing_product_id} rows."


# In[16]:


# No missing category (important for slicing/filtering in Power BI)
missing_category = dim_product.filter("product_category_english IS NULL OR product_category_english = ''").count()
assert missing_category == 0, f"Missing product_category_english in {missing_category} rows."


# In[17]:


# 5️⃣ Uniqueness of natural key
duplicate_product_ids = (
    dim_product.groupBy("product_id")
    .count()
    .filter("count > 1")
    .count()
)
assert duplicate_product_ids == 0, f"Duplicate product_id detected: {duplicate_product_ids} duplicates."


# ## dim_customer
# 
# - Expected null coordinates: ~0.3% of customers (279/99,441)
# - Root cause: 158 customer ZIP prefixes (out of 14,994 unique) do not exist in the geolocation reference table
# - Coverage: 99.72% (exceeds 95% threshold for production BI)
# - Handling: Nulls are ACCEPTABLE for optional dimension attributes. Power BI reports will filter these records in map visualizations. Do NOT impute fake coordinates - preserve data integrity.

# In[18]:


# distinct customer_id - sl_customer.select("customer_id").distinct().count()
sl_customer.show(5)
sl_geolocation.show(5)


# In[19]:


# DATA QUALITY CHECK: ZIP Code Coverage

# Find customer ZIP codes that don't exist in geolocation table
unmatched_zips = sl_customer.select("customer_zip_code_prefix").distinct() \
    .join(
        sl_geolocation.select("geolocation_zip_code_prefix").distinct(),
        sl_customer.customer_zip_code_prefix == sl_geolocation.geolocation_zip_code_prefix,
        how="left_anti"  # Returns ZIPs from customers NOT in geolocation
    )

unmatched_count = unmatched_zips.count()
total_unique_zips = sl_customer.select("customer_zip_code_prefix").distinct().count()
coverage_pct = ((total_unique_zips - unmatched_count) / total_unique_zips) * 100

# Report findings
print(f"\n📊 ZIP Code Coverage Report:")
print(f"   Total unique customer ZIPs: {total_unique_zips:,}")
print(f"   ZIPs missing from geolocation: {unmatched_count:,}")
print(f"   Coverage: {coverage_pct:.2f}%\n")

# Show sample of unmatched ZIPs with customer details
if unmatched_count > 0:
    print(f"⚠️  Sample of unmatched ZIP codes:")
    sl_customer.join(
        unmatched_zips,
        on="customer_zip_code_prefix",
        how="inner"
    ).select(
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state"
    ).distinct().show(20, truncate=False)

# Assertion: Warn if coverage is low
MIN_ZIP_COVERAGE = 90.0
if coverage_pct < MIN_ZIP_COVERAGE:
    print(f"⚠️  WARNING: ZIP coverage {coverage_pct:.2f}% below threshold {MIN_ZIP_COVERAGE}%")
else:
    print(f"✅ ZIP code coverage check passed: {coverage_pct:.2f}%")


# In[20]:


dim_customer = sl_customer.join(
    broadcast(sl_geolocation),
    sl_customer.customer_zip_code_prefix == sl_geolocation.geolocation_zip_code_prefix,
    how="left" # left join allows nulls for missing geolocaiton data
) \
.withColumn("customer_key", F.abs(F.xxhash64("customer_id")))\
.select(
    "customer_key",
    "customer_id",
    "customer_unique_id",
    "customer_city",
    "customer_state",
    "customer_zip_code_prefix",
    col("avg_lat").alias("customer_lat"),    # Rename from silver columns
    col("avg_lng").alias("customer_lng")
)

dim_customer.cache()

dim_customer.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("gd_dim_customer")

print(f"✅ dim_customer created: {dim_customer.count()} rows")


# ### Validation

# In[21]:


dim_customer.show(3)


# In[22]:


# surrogate key uniqueness
# uniqueness of surrogate key

try:
    unique_keys = dim_customer.select("customer_key").distinct().count()
    total_rows = dim_customer.count()

    assert unique_keys == total_rows, \
        f"Duplicate customer_key detected: {total_rows - unique_keys} duplicates."

    print("✅ Surrogate key uniqueness check passed.")

except AssertionError as e:
    print(f"❌ Assertion failed: {e}")

    # === 2️⃣ Debug block: inspect duplicates ===
    dupes = (
        dim_customer.groupBy("customer_key")
        .agg(F.count("*").alias("cnt"))
        .filter("cnt > 1")
    )

    print(f"Duplicate keys found: {dupes.count()}")
    dupes.show(10, truncate=False)

    # Join back to see which product_ids are affected
    dupe_details = (
        dim_customer.join(dupes, on="customer_key", how="inner")
        .orderBy("customer_key")
    )
    dupe_details.show(20, truncate=False)


# In[23]:


# Completeness. No null primary keys: customer_key and customer_id cannot be null.
assert dim_customer.filter(col("customer_key").isNull()).count() == 0, \
    "❌ CRITICAL: customer_key has null values"
    
assert dim_customer.filter(col("customer_id").isNull()).count() == 0, \
    "❌ CRITICAL: customer_id has null values"


# In[24]:


# referential integrity. all customers have records, ensures no customers were lost during the join
source_count = sl_customer.count()
target_count = dim_customer.count()
assert source_count == target_count, \
    f"❌ CRITICAL: Lost {source_count - target_count} customers during transformation"


# In[25]:


# coordinate validaty - removed - already added in silver
"""invalid_coords = dim_customer.filter(
    (col("customer_lat").isNotNull() & 
     ((col("customer_lat") < -34) | (col("customer_lat") > 6))) |
    (col("customer_lng").isNotNull() & 
     ((col("customer_lng") < -74) | (col("customer_lng") > -34)))
).count()
assert invalid_coords == 0, \
    f"❌ WARNING: {invalid_coords} coordinates outside Brazil"
"""


# ## dim_seller

# In[26]:


# check seller_id is distinct 
sl_seller.select("seller_id").distinct().count()


# In[27]:


# number of rows in silver sellers
print(sl_seller.count())


# In[28]:


sl_seller.show(2)


# In[29]:


dim_seller = sl_seller.join(
    broadcast(sl_geolocation),
    sl_seller.seller_zip_code_prefix == sl_geolocation.geolocation_zip_code_prefix,
    how="left" # left join allows nulls for missing geolocaiton data
) \
.withColumn("seller_key", F.abs(F.xxhash64("seller_id")))\
.select(
    "seller_key",
    "seller_id",
    "seller_city",
    "seller_state",
    "seller_zip_code_prefix",
    col("avg_lat").alias("seller_lat"),    # Rename from silver columns
    col("avg_lng").alias("seller_lng")
)

dim_seller.cache()

dim_seller.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("gd_dim_seller")

print(f"✅ gd_dim_seller created: {dim_seller.count()} rows")


# ### Validation 
# - same as in dim_customer. Future enhancement: Use PyTest fixtures. 

# In[ ]:


# surrogate key uniqueness
# uniqueness of surrogate key

try:
    unique_keys = dim_seller.select("seller_key").distinct().count()
    total_rows = dim_seller.count()

    assert unique_keys == total_rows, \
        f"Duplicate seller_key detected: {total_rows - unique_keys} duplicates."

    print("✅ Surrogate key uniqueness check passed.")

except AssertionError as e:
    print(f"❌ Assertion failed: {e}")

    # === 2️⃣ Debug block: inspect duplicates ===
    dupes = (
        dim_seller.groupBy("seller_key")
        .agg(F.count("*").alias("cnt"))
        .filter("cnt > 1")
    )

    print(f"Duplicate keys found: {dupes.count()}")
    dupes.show(10, truncate=False)

    # Join back to see which product_ids are affected
    dupe_details = (
        dim_seller.join(dupes, on="seller_key", how="inner")
        .orderBy("seller_key")
    )
    dupe_details.show(20, truncate=False)


# In[31]:


# Completeness. No null primary keys: seller_key and seller_id cannot be null.
assert dim_seller.filter(col("seller_key").isNull()).count() == 0, \
    "❌ CRITICAL: seller_key has null values"
    
assert dim_seller.filter(col("seller_id").isNull()).count() == 0, \
    "❌ CRITICAL: seller_id has null values"


# In[32]:


# referential integrity. all sellers have records, ensures no sellers were lost during the join
source_count = sl_seller.count()
target_count = dim_seller.count()
assert source_count == target_count, \
    f"❌ CRITICAL: Lost {source_count - target_count} sellers during transformation"


# # Fact Tables

# ## fact_order_fulfillment (Primary Fact)
# 
# This is an accumulating snapshot fact table, which tracks orders through multiple predictable stages from start to finish. The telltale signs are:
# - Multiple date foreign keys for each milestone (purchase, approved, carrier pickup, delivered, estimated)
# - Lag calculations between stages (processing time, dispatch time, transit time)
# - One row per order that represents the complete lifecycle
# 
# Accumulating snapshots are specifically designed to measure velocity and time spent at various stages in a business process.
# 
# Grain: One row per order. 
# 
# Primary Key: order_key (INT, surrogate key)
# 
# Natural Keys:order_id (STRING, for traceability)
# 
# - order_status
# 
# Foreign Keys
# - datekey_purchase (FK to Dim_Date)
# - datekey_approved (FK to Dim_Date)
# - datekey_carrier (FK to Dim_Date)
# - datekey_delivered  (FK to Dim_Date)
# - datekey_estimated (FK to Dim_Date)
# - customer_key (FK to Dim_Customer)
# 
# 
# Measures (Pre-calculated)
# Natural Keys:
# order_id (STRING, for traceability)
# 
# delivery_lead_time_days (DECIMAL: delivered_customer_date − delivered_carrier_date)
# 
# order_cycle_time_hours (DECIMAL: delivered_customer_date − purchase_timestamp, in hours)
# 
# processing_time_days (DECIMAL: approved_date − purchase_date)
# 
# dispatch_time_days (DECIMAL: delivered_carrier_date − approved_date)
# 
# transit_time_days (DECIMAL: delivered_customer_date − delivered_carrier_date)
# 
# delivery_variance_days (DECIMAL: delivered_customer_date − estimated_delivery_date)
# 
# is_on_time (BOOLEAN: 1 if delivered ≤ estimated, else 0)
# 
# is_late (BOOLEAN: 1 if delivered > estimated, else 0)
# 
# freight_value (DECIMAL, shipping cost)
# 
# order_total_amount (DECIMAL, sum of payments)
# 

# In[ ]:


# Kimball recommends role-based date keys (e.g. order_date_key, ship_date_key, etc.)
# This ensures all keys can join cleanly to Dim_Date.date_key

fact_order_fulfillment = (
    sl_order
    .withColumn("purchase_date_key",  F.date_format("order_purchase_timestamp", "yyyyMMdd").cast("int"))
    .withColumn("approved_date_key",  F.date_format("order_approved_at", "yyyyMMdd").cast("int"))
    .withColumn("carrier_date_key",   F.date_format("order_delivered_carrier_date", "yyyyMMdd").cast("int"))
    .withColumn("delivered_date_key", F.date_format("order_delivered_customer_date", "yyyyMMdd").cast("int"))
    .withColumn("estimated_date_key", F.date_format("order_estimated_delivery_date", "yyyyMMdd").cast("int"))
)

# Join with Dim_Customer to get customer_key
gd_dim_customer = spark.read.table("lh_gold_olist.gd_dim_customer")

fact_order_fulfillment = (
    fact_order_fulfillment
    .join(
        gd_dim_customer.select("customer_key", "customer_id"),
        on="customer_id",
        how="inner"
    )
)

# ----------------------------------------------------
# Derived metrics (KPI pre-computations)
# ----------------------------------------------------


# processing time (order placed -> approved)
# dispatch time (approved -> carrier pickup)
# transit time (carrier pickup -> customer delivery)
# delivery lead time (from total duration from when a customer places an order to when they receive the product)
# delivery variance (actual vs promised)
# on-time delivery flag (1 = on-time, 0 = late)
# late delivery flag (1 = late, 0 = on-time/early)


fact_order_fulfillment = (
    fact_order_fulfillment

    # ==== Stage Duration Metrics (only compute when both timestamps exist) ===

    # Processing time: purchase → approved
    .withColumn(
        "processing_time_days",
        F.when(
            F.col("order_purchase_timestamp").isNotNull() & F.col("order_approved_at").isNotNull(),
            F.datediff("order_approved_at", "order_purchase_timestamp")
        )
    )
    # Dispatch time: approved → carrier pickup
    .withColumn(
        "dispatch_time_days",
        F.when(
            F.col("order_approved_at").isNotNull() & F.col("order_delivered_carrier_date").isNotNull(),
            F.datediff("order_delivered_carrier_date", "order_approved_at")
        )
    )
    # Transit time: carrier pickup → delivery
    .withColumn(
        "transit_time_days",
        F.when(
            F.col("order_delivered_carrier_date").isNotNull() & F.col("order_delivered_customer_date").isNotNull(),
            F.datediff("order_delivered_customer_date", "order_delivered_carrier_date")
        )
    )
    # Lead time: purchase → delivery
    .withColumn(
        "lead_time_days",
        F.when(
            F.col("order_purchase_timestamp").isNotNull() & F.col("order_delivered_customer_date").isNotNull(),
            F.datediff("order_delivered_customer_date", "order_purchase_timestamp")
        )
    )
    # Delivery variance: actual vs promised
    .withColumn(
        "delivery_variance_days",
        F.when(
            F.col("order_estimated_delivery_date").isNotNull() & F.col("order_delivered_customer_date").isNotNull(),
            F.datediff("order_delivered_customer_date", "order_estimated_delivery_date")
        )
    )
    # === SLA Flags (only meaningful when delivered) === 

    .withColumn(
        "is_on_time",
        F.when(
            F.col("order_delivered_customer_date").isNotNull() &
            (F.col("order_delivered_customer_date") <= F.col("order_estimated_delivery_date")),
            F.lit(1)
        ).when(F.col("order_delivered_customer_date").isNotNull(), F.lit(0))
    )
    .withColumn(
        "is_late",
        F.when(
            F.col("order_delivered_customer_date").isNotNull() &
            (F.col("order_delivered_customer_date") > F.col("order_estimated_delivery_date")),
            F.lit(1)
        ).when(F.col("order_delivered_customer_date").isNotNull(), F.lit(0))
    )
)

# === Reorder columns following Kimball-style convention ===
# (Business key → foreign keys → facts → indicators)

fact_order_fulfillment = fact_order_fulfillment.select(
    "order_id",                # Degenerate PK (business key)
    "customer_key",            # FK → Dim_Customer
    "purchase_date_key",       # FK → Dim_Date
    "approved_date_key",       # FK → Dim_Date
    "carrier_date_key",        # FK → Dim_Date
    "delivered_date_key",      # FK → Dim_Date
    "estimated_date_key",      # FK → Dim_Date
    "order_status",

    # Derived metrics (in computation order)
    "processing_time_days",    # 1. purchase → approved
    "dispatch_time_days",      # 2. approved → carrier pickup
    "transit_time_days",       # 3. carrier pickup → delivery
    "lead_time_days",          # 4. purchase → delivery
    "delivery_variance_days",  # 5. actual vs promised

    # SLA Flags
    "is_on_time",              # 6. 1 if delivered ≤ estimated
    "is_late"                  # 7. 1 if delivered > estimated
)

# ----------------------------------------------------
# Persist to Gold layer
# ----------------------------------------------------
fact_order_fulfillment.cache() # BEST PRACTICE: Persist fact table—will be reused for other fact builds

fact_order_fulfillment.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save("Tables/gd_fact_order_fulfillment")

print(f"✅ fact_order_fulfillment created: {fact_order_fulfillment.count()} rows")


fact_order_fulfillment.printSchema()


# | Category                 | Behavior                                            | Benefit                                |
# | ------------------------ | --------------------------------------------------- | -------------------------------------- |
# | **Duration metrics**     | Only calculated when both timestamps exist          | Avoids null cascades, negative days    |
# | **SLA flags**            | Only assigned when delivery date exists             | Prevents false “late” flags            |
# | **Non-delivered orders** | Keep their rows, but all derived columns are `NULL` | Table remains an accumulating snapshot |
# | **Delivered orders**     | Have valid values in all metrics                    | Fully compatible with Power BI visuals |
# 
# 
# In Power BI:
# 
# Use filters like WHERE order_status = 'delivered' when computing KPI averages.
# 
# Or define DAX measures that ignore nulls automatically:
# 
# Avg Transit Time (days) = AVERAGE('fact_order_fulfillment'[transit_time_days])

# Visual Example
# 
# | order_id | order_status | order_purchase_timestamp | order_delivered_customer_date | is_on_time | lead_time_days |
# | -------- | ------------ | ------------------------ | ----------------------------- | ---------- | -------------- |
# | 1001     | delivered    | 2021-05-01               | 2021-05-05                    | 1          | 4              |
# | 1002     | shipped      | 2021-05-02               | *(null)*                      | *(null)*   | *(null)*       |
# | 1003     | canceled     | 2021-05-03               | *(null)*                      | *(null)*   | *(null)*       |
# 

# ### Vlaidation

# In[ ]:


gd_fact_order_fulfillment = spark.read.table("lh_gold_olist.gd_fact_order_fulfillment")


# In[ ]:


print("=== Validation: fact_order_fulfillment Derived Metrics ===")

# 1️⃣ Row Count Check
before_rows = sl_order.count()
print(f"Rows in Silver: {before_rows}")
total_rows = gd_fact_order_fulfillment.count()
print(f"Total rows in Gold: {total_rows:,}")

# 2️⃣ Negative Duration Assertions
invalid_durations = (
    gd_fact_order_fulfillment
    .filter(
        (F.col("processing_time_days") < 0) |
        (F.col("dispatch_time_days") < 0) |
        (F.col("transit_time_days") < 0) |
        (F.col("lead_time_days") < 0)
    )
    .count()
)
assert invalid_durations == 0, f"❌ Found {invalid_durations} negative duration(s)!"

# 3️⃣ SLA Flag Consistency (is_on_time vs is_late)
inconsistent_flags = (
    gd_fact_order_fulfillment
    .filter((F.col("is_on_time") == 1) & (F.col("is_late") == 1))
    .count()
)
assert inconsistent_flags == 0, f"❌ Found {inconsistent_flags} inconsistent flags (both 1)!"

# 4️⃣ Missing KPIs for Delivered Orders
delivered_orders = gd_fact_order_fulfillment.filter(F.col("order_status") == "delivered")
missing_kpis = (
    delivered_orders
    .filter(
        F.col("lead_time_days").isNull() |
        F.col("is_on_time").isNull() |
        F.col("is_late").isNull()
    )
    .count()
)
assert missing_kpis == 0, f"❌ Found {missing_kpis} delivered orders missing KPI fields!"

# 5️⃣ Statistical Range Sanity (informational)
stats = gd_fact_order_fulfillment.select(
    F.mean("processing_time_days").alias("avg_processing"),
    F.mean("dispatch_time_days").alias("avg_dispatch"),
    F.mean("transit_time_days").alias("avg_transit"),
    F.mean("lead_time_days").alias("avg_lead")
).collect()[0]

print(f"""
✅ Average Processing Time (days): {stats['avg_processing']:.2f}
✅ Average Dispatch Time (days):   {stats['avg_dispatch']:.2f}
✅ Average Transit Time (days):    {stats['avg_transit']:.2f}
✅ Average Lead Time (days):       {stats['avg_lead']:.2f}
""")

print("✅ All assertions passed — metrics validated successfully.")


# In[317]:


gold_duplicates = (
    gd_fact_order_fulfillment.groupBy("order_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)
assert gold_duplicates == 0, f"❌ Found {gold_duplicates} duplicate order_id(s) in Gold!"
print("✅ Each order_id appears exactly once in Gold.")


# In[318]:


# referential integrity check per date key

gd_dim_date = spark.read.format("delta").load("Tables/gd_dim_date")

# List of FK columns to check
date_keys = [c for c in gd_fact_order_fulfillment.columns if c.endswith("_date_key")]


# Loop through each date key and check referential integrity
for key in date_keys:
    f = gd_fact_order_fulfillment.alias("f")
    d = gd_dim_date.alias("d")

    missing = (
        f.select(F.col(key).alias("fk_date_key"))
        .distinct()
        .join(d, F.col("fk_date_key") == F.col("d.date_key"), "left_anti")
        .count()
    )

    if missing > 0:
        print(f"❌ {missing:,} {key} values not found in gd_dim_date!")
    else:
        print(f"✅ {key} fully matches with gd_dim_date.")



# In[319]:


# Reusable helper function - Checks referential integrit
# Verifies all foreign keys in the fact table exist in the corresponding dimension.
# Do all the foreign keys (FKs) in the fact table exist in the primary key (PK) column of the dimension table?
# If any foreign keys don’t match a valid dimension record, those are orphaned rows — a serious data integrity issue.

def fk_check(fact_df, dim_df, fact_fk, dim_pk, dim_name):
    missing = (
        fact_df
        .select(F.col(fact_fk).alias("fk")) # extract just the foreing key column
        .distinct()
        .join(dim_df.select(F.col(dim_pk).alias("pk")), F.col("fk") == F.col("pk"), "left_anti")
        .count()
    )

    if missing > 0: # show count
        print(f"❌ {missing:,} {fact_fk} values not found in {dim_name}.")
    else: 
        print(f"✅ {fact_fk} fully matches {dim_name}.")


# In[320]:


# customer key integrity
fk_check(gd_fact_order_fulfillment, gd_dim_customer, "customer_key", "customer_key", "dim_customer")


# ## fact_reviews
# 
# Purpose: Examine possible link between delivery performance to customer satisfaction
# 
# | Column Name                | Type      | Role / Description                       |
# | -------------------------- | --------- | ---------------------------------------- |
# | `review_key`               | INT       | Primary Key (surrogate from `review_id`) |
# | `order_id`                 | STRING    | FK → `fact_order_fulfillment`            |
# | `customer_key`             | INT       | FK → `dim_customer`                      |
# | `review_creation_date_key` | INT       | FK → `dim_date` (format `yyyyMMdd`)      |
# | `review_score`             | INT       | Customer’s rating (1–5)                  |
# | `is_low_rating`            | INT (0/1) | Derived flag: 1 if `review_score` ≤ 2    |
# 
# 

# In[ ]:


# Join reviews with fulfillment data for KPI analysis ---
fact_order_fulfillment_lookup = spark.table("gd_fact_order_fulfillment").select(
    "order_id",
    "delivery_variance_days",    # For KPI analysis only (won't store)
    "is_late"                    # For KPI analysis only (won't store)
)

fact_review = sl_review.join(
    fact_order_fulfillment_lookup,
    on="order_id",
    how="inner"
)


# In[ ]:


# Keep only reviews tied to valid fulfilled orders
fact_review = (
    sl_review.join(
        fact_order_fulfillment.select(
            "order_id",
            "delivery_variance_days", 
            "is_late"
        ),
        on="order_id",
        how="inner"
    )
)

print(f"✅ Joined dataset ready for KPI analysis: {fact_review.count():,} rows")


# In[ ]:


# Check join loss
src_count = sl_review.count()
joined_count = fact_review.count()
loss = src_count - joined_count
print(f"🧩 Silver table for reviews: {src_count:,}")
print(f"🔍 Lost during join: {loss:,} ({(loss/src_count*100):.2f}%)")

# NOTE: These lost reviews are ORPHANED - their order_ids don't exist in sl_order
# This is a source data quality issue (4,190 reviews reference non-existent orders).
# Since we can't analyze reviews without order context, excluding them is correct.
# The 4.26% loss is acceptable.

assert joined_count > 0, "❌ No records after join"
assert (loss / src_count) < 0.05, f"⚠️ More than 5% lost ({loss/src_count*100:.2f}%)"


# In[ ]:


# DIAGNOSTIC: Why are reviews being lost during join?

print("\n=== JOIN LOSS DIAGNOSTIC ===")

# Count reviews
src_count = sl_review.count()
print(f"📊 Total reviews in sl_review: {src_count:,}")

# Count orders in fulfillment
fulfillment_orders = spark.table("gd_fact_order_fulfillment").select("order_id").distinct().count()
print(f"📦 Unique orders in fact_order_fulfillment: {fulfillment_orders:,}")

# Do the join
fact_order_fulfillment_lookup = spark.table("gd_fact_order_fulfillment").select(
    "order_id",
    "delivery_variance_days",
    "is_late"
)

fact_review = sl_review.join(
    fact_order_fulfillment_lookup,
    on="order_id",
    how="inner"
)


joined_count = fact_review.count()
loss = src_count - joined_count

print(f"✅ Reviews after join: {joined_count:,}")
print(f"❌ Reviews lost: {loss:,} ({(loss/src_count*100):.2f}%)")

# Find which reviews were lost
lost_reviews = sl_review.join(
    fact_order_fulfillment_lookup.select("order_id"),
    on="order_id",
    how="left_anti"  # Keep rows from sl_review that DON'T match fulfillment
)

print(f"\n🔍 Analyzing {loss:,} lost reviews...")

# Check order_id patterns in lost reviews
lost_order_ids = lost_reviews.select("order_id").distinct()
print(f"   - Lost reviews belong to {lost_order_ids.count():,} unique orders")

# Sample some lost order_ids
print("\n📋 Sample order_ids that caused review loss:")
lost_order_ids.show(10, truncate=False)

# Check if these orders exist in sl_order
sl_order = spark.read.table("lh_silver_olist.sl_order")
orders_in_source = lost_reviews.join(
    sl_order.select("order_id"),
    on="order_id",
    how="inner"
).count()

print(f"\n🔎 Of the {loss:,} lost reviews:")
print(f"   - {orders_in_source:,} have matching orders in sl_order")
print(f"   - {loss - orders_in_source:,} don't even exist in sl_order (orphaned!)")

# Check order status of lost reviews
if orders_in_source > 0:
    lost_with_status = lost_reviews.join(
        sl_order.select("order_id", "order_status"),
        on="order_id",
        how="inner"
    )
    
    print("\n📊 Order status breakdown of lost reviews:")
    lost_with_status.groupBy("order_status").count().orderBy(F.desc("count")).show()

print("\n=== END DIAGNOSTIC ===\n")


# In[325]:


# Ensure low-rating flag exists (if not already in fact_reviews)
if "is_low_rating" not in fact_review.columns:
    fact_review = fact_review.withColumn(
        "is_low_rating",
        F.when(F.col("review_score") <= 2, 1).otherwise(0).cast("int")
    )


print(f"✅ fact_review rows: {fact_review.count():,} rows")


# In[326]:


# quick sanity checks

fact_review.select("review_score", "delivery_variance_days", "is_late").summary().show()

missing_delay = fact_review.filter(F.col("delivery_variance_days").isNull()).count()
missing_score = fact_review.filter(F.col("review_score").isNull()).count()

print(f"⚙️ Missing delay values: {missing_delay:,}")
print(f"⚙️ Missing review scores: {missing_score:,}")


# In[327]:


# KPI calculation

# --- Step 3: KPI Calculations (using temp columns) ---

print("\n=== KPI Analysis ===")

# (a) Average Review Score by Delivery Status
avg_score = (
    fact_review
    .groupBy("is_late")
    .agg(
        F.avg("review_score").alias("avg_review_score"),
        F.count("*").alias("num_reviews")
    )
    .orderBy("is_late")
)
print("\n📊 Average Review Score by Delivery Status:")
avg_score.show()

# (b) % of Low Ratings Linked to Late Deliveries
late_low = fact_review.filter((F.col("is_late") == 1) & (F.col("is_low_rating") == 1)).count()
late_total = fact_review.filter(F.col("is_late") == 1).count()
low_total = fact_review.filter(F.col("is_low_rating") == 1).count()
pct_low_due_to_late = (late_low / low_total * 100) if low_total > 0 else 0
print(f"\n📉 {pct_low_due_to_late:.2f}% of low ratings (1–2 stars) are linked to late deliveries.")

# (c) Correlation Between Delay and Review Score
fact_review_clean = fact_review \
    .withColumn("delivery_variance_days", F.col("delivery_variance_days").cast("double")) \
    .withColumn("review_score", F.col("review_score").cast("double")) \
    .filter(F.col("delivery_variance_days").isNotNull() & F.col("review_score").isNotNull())

corr_value = fact_review_clean.stat.corr("delivery_variance_days", "review_score")
print(f"\n📈 Correlation: {corr_value:.4f}")


# Expect a negative correlation (e.g., −0.3 → longer delays → lower ratings)


# IMPORTANT: This correlation tells you that delivery delays negatively impact customer satisfaction, but delays aren't the only factor influencing review scores. Other variables likely matter too—product quality, packaging, customer service, price expectations, or even factors unrelated to the order itself.
# For your Power BI dashboard and analytics project, this validates that on-time delivery is important for customer satisfaction, but you should also investigate other drivers of low ratings beyond just late deliveries. The fact that it's not a stronger correlation (-0.5 or below) suggests there are additional opportunities to improve review scores even when deliveries are on time.
# Remember: correlation doesn't prove causation, though in this logistics context, the causal relationship (late delivery → frustrated customer → lower rating) is fairly intuitive.


# In[328]:


# Check uniqueness first of review_id. otherwise create surrogate key
total_reviews = fact_review.count()
unique_review_ids = fact_review.select("review_id").distinct().count()
duplicates = total_reviews - unique_review_ids

print(f"📊 Total reviews: {total_reviews:,}")
print(f"🔑 Unique review_ids: {unique_review_ids:,}")
print(f"⚠️  Duplicates: {duplicates:,}")

# Generate surrogate key
fact_review = fact_review.withColumn(
    "review_key", 
    F.monotonically_increasing_id()
)
print("✅ Surrogate key created")


# In[329]:


# Generate surrogate key
fact_review = fact_review.withColumn(
    "review_key", 
    F.monotonically_increasing_id()
)

print("✅ Surrogate key 'review_key' created")

# Select final columns for Gold layer
fact_review_final = fact_review.select(
    "review_key",                    # PK (surrogate)
    "review_id",                     # degenerate dimension for traceability
    "order_id",                      # FK → fact_order_fulfillment (use this to join for delivery metrics!)
    "review_score",                  # Measure (1-5)
    "is_low_rating"                  # Flag (1 if score ≤ 2)
)



# In[330]:


# Write to Gold layer (overwrite mode for reproducibility)
fact_review_final.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("gd_fact_review")

print(f"\n✅ gd_fact_review created: {fact_review_final.count():,} rows")


# ### Validation

# In[331]:


# 1. Primary Key
dup_check = fact_review_final.groupBy("review_key").count().filter("count > 1").count()
assert dup_check == 0, f"❌ Duplicate review_key: {dup_check}"
assert fact_review_final.filter("review_key IS NULL").count() == 0, "❌ Null review_key"
print("✅ Primary key valid")

# 2. Foreign Keys
assert fact_review_final.filter("order_id IS NULL").count() == 0, "❌ Missing order_id"
print("✅ Foreign key valid")


# 3. Value Validity
assert fact_review_final.filter("review_score < 1 OR review_score > 5").count() == 0, \
    "❌ Invalid review_score"

bad_flag = fact_review_final.filter(
    "(review_score <= 2 AND is_low_rating != 1) OR (review_score > 2 AND is_low_rating != 0)"
).count()
assert bad_flag == 0, f"❌ {bad_flag} mismatched flags"
print("✅ Values valid")

# 4. Summary
total = fact_review_final.count()
low = fact_review_final.filter("is_low_rating = 1").count()

print(f"\n📊 Total reviews: {total:,}")
print(f"📉 Low ratings: {low:,} ({low/total*100:.2f}%)")
print("\n✅ All validations passed!")


# In[332]:


display(fact_review_final)


# ## fact_order_items 

# Purpose: Transaction-level fact table linking orders, products, sellers and dates. Track seller dispatch performance and product-level metrics. 
# 
# Grain: One row per item per order (order_id, order_item_id). Each row represents a single product sold within a particular order. 
# 
# Questions that can be answered: 
# - Which sellers consistently take the longest to dispatch?
# - Which product categories have the longest delivery times?
# - Are heavier orders likely to face delyas?
# 
# 
# | Column                    | Type    | Role                            | Description                                                        |
# | ------------------------- | ------- | ------------------------------- | ------------------------------------------------------------------ |
# | `order_item_id`           | PK      | Primary Key                     | Unique identifier for each row in `order_items` (check uniqueness) |
# | `order_id`                | FK      | Link to `fact_order_fulfillment` | Allows joining with order-level delivery metrics                   |
# | `shipping_limit_date_key` | FK      | Link to `dim_date`              | Date by which the seller should ship the item                      |
# | `seller_key`              | FK      | Link to `dim_seller`            | Surrogate key for seller                                           |
# | `product_key`             | FK      | Link to `dim_product`           | Surrogate key for product                                          |
# | `price`                   | DECIMAL | Measure                         | Selling price of the item                                          |
# | `freight_value`           | DECIMAL | Measure                         | Freight (delivery charge) for the item                             |
# 

# In[ ]:


sl_product.printSchema()
sl_order_item.printSchema()


# Quality issues - Referential integrity audit between silver and gold layer through quality_check_sl_order_items at lakehouse saved queries.
# 
# There are two issues to handle before promoting to Gold:
# - 4,069 orphaned order_items → missing parent order.
# - 1,636 orphaned product_ids → missing from product dimension.
# 
# | Check                                    | Purpose               | Result      | Interpretation                        |
# | ---------------------------------------- | --------------------- | ----------- | ------------------------------------- |
# | Total records                            | `COUNT(*)`            | **112,650** | Matches Olist dataset’s known count ✅ |
# | `order_id` null check                    | Verify mandatory key  | **0**       | Good – every item belongs to an order |
# | Missing `order_id` in `sl_orders`        | Referential integrity | **4,069**   | ❌ Orphaned items (orders not found)   |
# | `product_id` null check                  | Verify mandatory key  | **0**       | Good                                  |
# | Missing `product_id` in `gd_dim_product` | Referential integrity | **1,636**   | ❌ Orphaned items (products missing)   |
# | `seller_id` null check                   | Verify mandatory key  | **0**       | Good                                  |
# | Missing `seller_id` in `gd_dim_seller`   | Referential integrity | **0**       | All sellers found ✅                   |
# 
# Root cause for 1) Missing parent orders (4069)
# - Known defect in Olist dataset. Some `order_items` refer to `order_id`s that don't appear in the raw `orders` dataset. 
# - Resolution: Drop orphans before joining to Facts OR if want to preserve transparency/full traceability, we keep them in a quarantine table (chosen). 
# 
# Root cause for 2) Missing products (1636).
# 
# Solution options:
# 
# | Option                                                             | Description                                                                                 | Consequences / Pros & Cons                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
# | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
# | **A. Drop the rows (inner‐join, exclude missing products)** (chosen)       | Only keep `order_item` records whose `product_id` matches a record in the product dimension | **Pros**: <br> • Referential integrity in Gold — every fact row has valid foreign keys <br> • Easier to build clean joins and aggregations without worrying about null dimension keys <br> • You avoid misleading metrics referencing “phantom” products <br> **Cons**: <br> • You lose ~1,636 rows (in your numbers) of sales/data. That loss may slightly undercount revenue, sales volume, etc. <br> • If missing products are systematically nontrivial (e.g. new or niche items), you introduce bias (i.e. dropped segments) <br> • Auditors may ask “why were these records dropped?” and expect a reconciliation with Bronze counts |
# | **B. Use a surrogate / “Unknown Product” dimension / placeholder** | Retain those rows, but assign them a special product_key (e.g. 0 or “unknown”)              | **Pros**: <br> • You preserve *all* order_item data, so aggregates (e.g. total sales, item counts) remain full <br> • You can still flag and trace “unknown product” cases in reports <br> **Cons**: <br> • Some metrics by product will have an “Unknown” bucket which may dilute clarity <br> • Joins that expect “real” products may need extra logic to exclude or handle the unknown key <br> • Analytics like “top products by revenue” may be off if you don’t filter out the unknown bucket                                                                                                                                        |
# 
# 
# Which is “better”?
# 
# If your goal is a clean, well‐enforced star schema where every fact row has valid dimension links, then approach A is safer.
# 
# If your priority is full completeness of data — you want to preserve every sale record even if the dimension is missing — then approach B is more tolerant.
# 
# In practice, many production pipelines combine both:
# 
# Use Option B (surrogate/unknown) so you don’t “lose” data.
# 
# Also log and monitor how many “unknown product” rows exist, and inspect them (maybe supplement your product dimension later).
# 
# And for certain analyses, analysts can choose to exclude the “unknown product” bucket if they want “pure” product-based metrics.
# 

# In[ ]:


gd_dim_product = spark.table("gd_dim_product").select("product_id", "product_key")
gd_dim_seller = spark.table("gd_dim_seller").select("seller_id", "seller_key")

# --- Referential Integrity Checks ---
missing_orders = sl_order_item.join(sl_order, "order_id", "left_anti").count()
missing_products = sl_order_item.join(gd_dim_product, "product_id", "left_anti").count()
missing_sellers = sl_order_item.join(gd_dim_seller, "seller_id", "left_anti").count()

print("🔎 Referential Integrity Summary:")
print(f" - Missing orders: {missing_orders}")
print(f" - Missing products: {missing_products}")
print(f" - Missing sellers: {missing_sellers}")

# quarantine invalid records
qr_missing_orders = sl_order_item.join(sl_order, "order_id", "left_anti")
qr_missing_products = sl_order_item.join(gd_dim_product, "product_id", "left_anti")


# In[ ]:


# clean, referentially valid dataset - inner joins only for valid relationship 

# Bring in surrogate keys from dimensions
# In gold, every fact table should use surrogate keys from the corresponding dimension tables
# Why? To ensure referential integrity (every seller, product points to a valid dim entry)
# To support slowly changing dimensions (SCD) or conformed dimensions in the future
# To make joins faster and cleaner in Power BI or SQL because you'll always join by integer surrogate keys instead of long string IDs

fact_order_items = (
    sl_order_item
    .join(sl_order, "order_id", "inner")
    .join(gd_dim_product.select("product_id", "product_key"), "product_id", "inner")
    .join(gd_dim_seller.select("seller_id", "seller_key"), "seller_id", "inner")
)


# In[336]:


# no derived metrics. shipping limit date surrogate key (yyyyMMdd -> ints)
fact_order_items = (
    fact_order_items
    # Date surrogate key (yyyyMMdd → int)
    .withColumn("shipping_limit_date_key",
        F.date_format("shipping_limit_date", "yyyyMMdd").cast("int")
    )
    # derived measures
)


# In[337]:


# --- Select Gold columns 
fact_order_items = fact_order_items.select(
    "order_id",                 # PK part 1
    "order_item_id",            # PK part 2
    "product_key",              # FK → Dim_Product
    "seller_key",               # FK → Dim_Seller
    "shipping_limit_date_key",  # FK → Dim_Date
    "price",                    # item's product price (actual sale price)
    "freight_value",            # delivery charge (paid by customer) 
)


# In[338]:


assert fact_order_items.count() > 0, "❌ No data left after cleaning"

assert fact_order_items.filter(F.col("shipping_limit_date_key").isNull()).count() == 0, \
    "❌ Null shipping_limit_date_key detected"

# FK checks
fact = spark.table("gd_fact_order_items")
dim_date = spark.table("gd_dim_date")
dim_product = spark.table("gd_dim_product")
dim_seller = spark.table("gd_dim_seller")

fk_check(fact, dim_date, "shipping_limit_date_key", "date_key", "gd_dim_date")
fk_check(fact, dim_product, "product_key", "product_key", "gd_dim_product")
fk_check(fact, dim_seller, "seller_key", "seller_key", "gd_dim_seller")

# Uniqueness check for composite PK
dup_check = (
    fact_order_items
    .groupBy("order_id", "order_item_id")
    .count()
    .filter("count > 1")
    .count()
)
assert dup_check == 0, f"❌ Duplicate composite key found: {dup_check}"

# Ensure date key integrity
null_dates = fact_order_items.filter(F.col("shipping_limit_date_key").isNull()).count()
assert null_dates == 0, f"❌ Null shipping_limit_date_key found in {null_dates} rows"

print("✅ All integrity checks passed. Gold fact table ready for write.")


# In[339]:


print("✅ Gold fact table ready for write")

fact_order_items.cache()


fact_order_items.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("Tables/gd_fact_order_items")
qr_missing_orders.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("Tables/qr_fact_order_items_missing_orders")
qr_missing_products.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("Tables/qr_fact_order_items_missing_products")

# summary report
clean_count = fact_order_items.count()
total_quarantine = qr_missing_orders.count() + qr_missing_products.count()
print(f"Summary → Clean: {clean_count:,} | Quarantined: {total_quarantine:,}")

print("📊 Gold Fact Build Summary")
print(f" - Clean records written : {clean_count:,}")
print(f" - Quarantined (orders)  : {qr_missing_orders.count():,}")
print(f" - Quarantined (products): {qr_missing_products.count():,}")
print(f" - Total quarantined     : {total_quarantine:,}")
print("💾 Saved → gd_fact_order_items, qr_fact_order_items_missing_orders, qr_fact_order_items_missing_products")


# ## Misc

# In[340]:


# List of Gold fact and dimension tables
gold_tables = [
    "gd_dim_date",
    "gd_dim_product",
    "gd_dim_customer",
    "gd_dim_seller",
    "gd_fact_order_fulfillment",
    "gd_fact_order_items",
    "gd_fact_review"
]

for table in gold_tables:
    print(f"\n🧩 Schema for {table}")
    df = spark.read.format("delta").load(f"Tables/{table}")
    df.printSchema()

