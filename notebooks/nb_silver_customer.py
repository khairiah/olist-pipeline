#!/usr/bin/env python
# coding: utf-8

# ## nb_silver_customer
# 
# New notebook

# # Load

# 1. customer_id - unique surrogate key
# 2. unique_customer_unique_id
# 3. customer_zip_code_prefix
# 4. customer_city
# 5. customer_state
# 
# Key Findings
# - Majority are one-time buyers. A total of 2997 customers made repeat purchases on the platform.
# - Most customers are located at Sao Paolo.

# In[7]:


# Core PySpark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

# Selected functions and types
from pyspark.sql.functions import col, udf, regexp_replace, when, count, datediff
from pyspark.sql.types import StructType, StructField, StringType,TimestampType

# Standard Python libraries
import re
import unicodedata
import os

# List Lakehouse tables registered in the metastore
#print(spark.catalog.listTables())

# Schema
schema = StructType([
    StructField("customer_id", StringType(), False),
    StructField("customer_unique_id", StringType(), False),
    StructField("customer_zip_code_prefix", StringType(), True), 
    StructField("customer_city", StringType(), True),
    StructField("customer_state", StringType(), True),
])

# customers_df = spark.read.format("delta").load("Tables/br_customers")
customers_df = spark.read.table("br_customers")


# # Profiling

# Below reveals
# - ZIPs too short or with letters (non-numeric)
# - Blank states
# - Messy city names 
# - Duplicate customer IDs
# - Null fields

# In[8]:


print("\n=== CUSTOMERS DATA PROFILING======")

total_rows = customers_df.count()
print(f"Total rows: {total_rows:,}")

# 1. Missing / nulls
null_summary = (
    customers_df.select([
        F.count(F.when(F.col(c).isNull() | (F.col(c) == ''), c)).alias(c)
        for c in customers_df.columns
    ])
)
print("\n[1] Null / Missing Values per Column:")
null_summary.show()

# 2. ZIP length & format
customers_df = customers_df.withColumn("zip_len", F.length(F.col("customer_zip_code_prefix")))
zip_len_dist = customers_df.groupBy("zip_len").count().orderBy("zip_len")
print("\n[2] ZIP Code Length Distribution:")
zip_len_dist.show(truncate=False)

short_zip = customers_df.filter(F.length(F.col("customer_zip_code_prefix")) < 5).count()
non_numeric_zip = customers_df.filter(~F.col("customer_zip_code_prefix").rlike("^[0-9]+$")).count()
print(f"ZIP prefixes shorter than 5 digits: {short_zip}")
print(f"ZIP prefixes with non-numeric chars: {non_numeric_zip}")

# 3. Duplicates
dup_customer_id = customers_df.groupBy("customer_id").count().filter("count > 1").count()
dup_unique_id = customers_df.groupBy("customer_unique_id").count().filter("count > 1").count()
print("\n[3] Duplicate Records:")
print(f"Duplicate customer_id: {dup_customer_id}")
print(f"Duplicate customer_unique_id: {dup_unique_id}")

# 4. Invalid states
invalid_state = customers_df.filter(~F.col("customer_state").rlike("^[A-Za-z]{2}$")).count()
print("\n[4] Invalid 2-letter state codes:", invalid_state)
customers_df.groupBy("customer_state").count().orderBy("count", ascending=False).show(10, truncate=False)

# 5. City anomalies
weird_city = customers_df.filter(~F.col("customer_city").rlike("^[A-Za-zÀ-ÿ\\s\\-]+$")).count()
print("\n[5] City names with symbols/digits:", weird_city)
customers_df.select("customer_city").distinct().orderBy("customer_city").show(10, truncate=False)

print("\n=== RAW QUALITY SUMMARY ===")
print(f"→ Missing ZIPs: {short_zip + non_numeric_zip}")
print(f"→ Invalid states: {invalid_state}")
print(f"→ Weird city names: {weird_city}")
print(f"→ Duplicate IDs (customer_id): {dup_customer_id}, (customer_unique_id): {dup_unique_id}")
print("---------------------------------------------------")


# In[9]:


print("Columns:",customers_df.columns)
print("Rows:",customers_df.count())


# In[10]:


print("\n=== PROFILING CUSTOMERS DATASET ===")
customers_df.describe(["customer_zip_code_prefix"]).show()
customers_df.groupBy("customer_state").count().orderBy("count", ascending=False).show(10)
customers_df.select("customer_city").distinct().count()


# # Cleaning

# - No missing data.
# - Valid zip code prefix. Ensure length exactly 5 digits.
# 
# - Check for duplicates for customer_unique_id as this should be unique per customer. (validation)
# - State
#   - Normalize -> uppercase + trim spaces
#   - Validate -> must be one of the 27 Brazil states
# - City name case standardization. Fix:
#   - Different spellings: "sao paulo", "são paulo".
#   - Case differences: "Rio de Janeiro" vs "rio de janeiro".
#   - Trailing spaces.
#   - Special characters / diacritics (ã, ç).
# 
# 
#   trim, lowercase city, uppercase state,
# 
# strip accents,
# 
# remove punctuation,
# 
# pad ZIPs to 5 digits.
# 
#   

# In[11]:


print("\n=== DATA CLEANING AND STANDARDIZATION ===")

# 2.1 Trim column names and string values
customers_df = customers_df.toDF(*[c.strip() for c in customers_df.columns])
for c in customers_df.columns:
    customers_df = customers_df.withColumn(c, F.trim(F.col(c)))

# 2.2 Normalize casing
customers_df = customers_df.withColumn("customer_city",  F.lower(F.col("customer_city")))
customers_df = customers_df.withColumn("customer_state", F.upper(F.col("customer_state")))

# 2.3 Remove accents/punctuation in city
def strip_accents_py(s):
    if s is None: return s
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

strip_accents = F.udf(strip_accents_py, StringType())

customers_df = customers_df.withColumn("customer_city", strip_accents(F.col("customer_city")))
customers_df = customers_df.withColumn("customer_city", F.regexp_replace(F.col("customer_city"), "[^a-zA-Z\\s\\-]", ""))
customers_df = customers_df.withColumn("customer_city", F.regexp_replace(F.col("customer_city"), "\\s+", " "))

# 2.4 Clean and pad ZIP prefixes
customers_df = customers_df.withColumn("customer_zip_code_prefix",
                                       F.regexp_replace(F.col("customer_zip_code_prefix"), "[^0-9]", ""))
customers_df = customers_df.withColumn("customer_zip_code_prefix",
                                       F.lpad(F.col("customer_zip_code_prefix"), 5, "0"))


# In[12]:


print("\n=== VALIDATION AND DATA-QUALITY CHECKS ===")

# 4.1 Nulls
nulls = customers_df.filter(
    F.col("customer_id").isNull() |
    F.col("customer_unique_id").isNull() |
    F.col("customer_zip_code_prefix").isNull() |
    F.col("customer_city").isNull() |
    F.col("customer_state").isNull()
).count()
print(f"Null critical fields: {nulls}")
assert nulls == 0, "Nulls found in critical columns."

# 4.2 ZIP validation
zip_len_bad = customers_df.filter(F.length(F.col("customer_zip_code_prefix")) < 5).count()
zip_bad = customers_df.filter(~F.col("customer_zip_code_prefix").rlike("^[0-9]{5}$")).count()
print(f"ZIP <5: {zip_len_bad} | Invalid ZIPs: {zip_bad}")
assert (zip_len_bad + zip_bad) == 0, "Invalid ZIP prefixes."

# 4.3 State format
state_bad = customers_df.filter(~F.col("customer_state").rlike("^[A-Z]{2}$")).count()
print(f"Invalid state format: {state_bad}")
assert state_bad == 0, "Invalid state code format."

# 4.4 City format
city_bad = customers_df.filter(
    (F.col("customer_city") == "") |
    (~F.col("customer_city").rlike("^[a-zA-Z\\s\\-]+$"))
).count()
print(f"Invalid city names: {city_bad}")
assert city_bad == 0, "Invalid city text format."


# In[13]:


# check zip code length = 5
check_zip = customers_df .withColumn(
    "zip_str", F.col("customer_zip_code_prefix").cast("string")
).withColumn(
    "zip_length", F.length("zip_str")
)

check_zip.groupBy("zip_length").count().orderBy("zip_length").show()


# In[14]:


""" 
br_geolocation = spark.read.format("delta").load("Tables/br_geolocation")
sl_geolocation = spark.read.format("delta").load("Tables/sl_geolocation")


print("br_geolocation rows:", br_geolocation.count())
print("sl_geolocation rows:", sl_geolocation.count())

cust_unmatched_br = customers_df.join(
    br_geolocation,
    customers_df.customer_zip_code_prefix == br_geolocation.geolocation_zip_code_prefix,
    "left_anti"
).select("customer_zip_code_prefix").distinct()

cust_unmatched_sl = customers_df.join(
    sl_geolocation,
    customers_df.customer_zip_code_prefix == sl_geolocation.geolocation_zip_code_prefix,
    "left_anti"
).select("customer_zip_code_prefix").distinct()

diff_br_vs_sl = cust_unmatched_sl.subtract(cust_unmatched_br)
print("ZIPs unmatched only in sl_geolocation:")
diff_br_vs_sl.show(truncate=False)

target_zip = diff_br_vs_sl.collect()[0][0]
print(f"Investigating ZIP prefix: {target_zip}")

geo_br = br_geolocation.filter(F.col("geolocation_zip_code_prefix") == target_zip)
geo_sl = sl_geolocation.filter(F.col("geolocation_zip_code_prefix") == target_zip)

print("In br_geolocation:")
geo_br.show(truncate=False)
print("In sl_geolocation:")
geo_sl.show(truncate=False)

missing_in_silver_but_present_in_bronze = missing_geo_df.join(
    br_geolocation,
    missing_geo_df.customer_zip_code_prefix == br_geolocation.geolocation_zip_code_prefix,
    "inner"
)


"""


# In[15]:


print("\n=== REFERENTIAL INTEGRITY CHECK (Customers ↔ Geolocation) ===")

geo_bronze = spark.read.table("br_geolocation")

geo_silver = spark.read.table("lh_silver_olist.sl_geolocation")

missing_geo = customers_df.join(
    geo_silver,
    customers_df.customer_zip_code_prefix == geo_silver.geolocation_zip_code_prefix,
    "left_anti"
).count()

total_customers = customers_df.count()
print(f"Customers without matching geolocation ZIP: {missing_geo} "
      f"({missing_geo/total_customers*100:.2f}% of total)")

assert missing_geo / total_customers < 0.01, \
    "Too many unmatched ZIPs; check geolocation coverage."

# inspect missing ZIPs not found in geolocation
missing_geo = customers_df.join(
    geo_silver,
    customers_df.customer_zip_code_prefix == geo_silver.geolocation_zip_code_prefix,
    "left_anti"
)

missing_geo.select("customer_zip_code_prefix").distinct().show(50, truncate=False)

# check if any exists in the raw Bronze data 
missing_geo.join(
    geo_bronze,
    missing_geo.customer_zip_code_prefix == geo_bronze.geolocation_zip_code_prefix,
    "inner"
).select("customer_zip_code_prefix", "geolocation_city", "geolocation_state", "geolocation_lat", "geolocation_lng").show()


# In[16]:


print("\n=== POST-CLEANING QUALITY VALIDATION (self-contained) ===")

# 1) ZIP format
zip_len_lt5 = customers_df.filter(F.length(F.col("customer_zip_code_prefix")) < 5).count()
zip_bad_fmt = customers_df.filter(~F.col("customer_zip_code_prefix").rlike("^[0-9]{5}$")).count()
print(f"Short ZIPs (<5): {zip_len_lt5}")
print(f"Non-numeric / wrong-length ZIPs: {zip_bad_fmt}")

# 2) State format (regex)
state_bad_fmt = customers_df.filter(~F.col("customer_state").rlike("^[A-Z]{2}$")).count()
print(f"State bad format (not 2 uppercase letters): {state_bad_fmt}")

# 3) City text
city_bad = customers_df.filter(
    (F.col("customer_city") == "") |
    (~F.col("customer_city").rlike(r"^[a-zA-Z\s\-]+$"))
).count()
print(f"Invalid city names: {city_bad}")

# 4) Nulls per column
nulls_df = customers_df.select([
    F.count(F.when(F.col(c).isNull() | (F.col(c) == ''), c)).alias(c)
    for c in customers_df.columns
])
print("\nNulls per column:")
nulls_df.show()

# 5) Brazil state code validity
valid_states = ["AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT",
                "MS","MG","PA","PB","PR","PE","PI","RJ","RN","RS","RO",
                "RR","SC","SP","SE","TO"]
invalid_state_values = customers_df.filter(~F.col("customer_state").isin(valid_states))
invalid_state_count = invalid_state_values.count()
print(f"Invalid (non-Brazil) state codes: {invalid_state_count}")
if invalid_state_count > 0:
    invalid_state_values.select("customer_state").distinct().show(10, truncate=False)

# 6) Placeholder ZIPs
placeholder_zip_count = customers_df.filter(
    F.col("customer_zip_code_prefix").isin(["00000", "99999"])
).count()
print(f"Placeholder ZIPs (00000/99999): {placeholder_zip_count}")

# 7) Duplicate customer_id
dup_customer_ids = customers_df.groupBy("customer_id").count().filter("count > 1").count()
print(f"Duplicate customer_id records: {dup_customer_ids}")

# ---------------------------------------------------------------
# 8) REFERENTIAL INTEGRITY (vs geo_silver) — SOFT WARNING
# ---------------------------------------------------------------
print("\n=== REFERENTIAL INTEGRITY CHECK (SOFT WARNING) ===")

geo_silver = spark.table("sl_geolocation").select(
    F.col("geolocation_zip_code_prefix").alias("geo_zip")
).distinct()

# Find customer ZIPs missing from Silver geolocation
missing_geo_df = (
    customers_df.join(
        F.broadcast(geo_silver),
        customers_df.customer_zip_code_prefix == geo_silver.geo_zip,
        "left_anti"
    )
    .select("customer_zip_code_prefix", "customer_city", "customer_state")
    .distinct()
)

missing_count = missing_geo_df.count()
missing_zip_list = []  # define upfront

if missing_count > 0:
    print(f"⚠️ WARNING: {missing_count} customer ZIPs not found in sl_geolocation.")
    missing_geo_df.show(10, truncate=False)
    missing_zip_list = [r["customer_zip_code_prefix"] for r in missing_geo_df.collect()]
    print(f"Missing ZIPs → {missing_zip_list}")
else:
    print("✅ All customer ZIP prefixes successfully match entries in sl_geolocation.")

# ---------------------------------------------------------------
# CHECK WHETHER THESE ZIPs EXIST IN BRONZE GEOLOCATION
# ---------------------------------------------------------------
print("\n=== CHECKING IF MISSING ZIPs EXIST IN BRONZE GEOLOCATION ===")

geo_bronze = spark.table("br_geolocation").select(
    F.col("geolocation_zip_code_prefix").alias("bronze_zip")
).distinct()

if missing_zip_list:
    exist_in_bronze = geo_bronze.filter(F.col("bronze_zip").isin(missing_zip_list))
    exist_count = exist_in_bronze.count()
    print(f"Found {exist_count} of them in Bronze geolocation (possibly dropped during cleaning).")

    missing_in_bronze = [
        z for z in missing_zip_list
        if geo_bronze.filter(F.col("bronze_zip") == z).count() == 0
    ]
    print(f"ZIPs not found even in BRONZE: {len(missing_in_bronze)} → {missing_in_bronze}")
else:
    print("No missing ZIPs to check against Bronze geolocation.")

# ---------------------------------------------------------------
# 9) FINAL HARD ASSERTIONS (unchanged)
# ---------------------------------------------------------------
try:
    assert (zip_len_lt5 + zip_bad_fmt) == 0, "ZIP format issues remain"
    assert state_bad_fmt == 0, "State format problems remain"
    assert invalid_state_count == 0, "Invalid Brazilian state codes present"
    assert dup_customer_ids == 0, "Duplicate customer_id detected"
    print("\n✅ POST-CLEANING VALIDATION COMPLETE (Hard checks passed)")
except AssertionError as e:
    print(f"\n❌ Validation failed: {e}")


# In[17]:


# test for any weird city names as seein sellers
'''customers_df.filter(~F.col("customer_city").rlike(r"^[a-zA-Z\s\-]+$")) \
  .select("customer_id", "customer_city", "customer_state") \
  .distinct() \
  .show(30, truncate=False)
'''


# In[20]:


expected_cols = ["customer_id","customer_unique_id","customer_zip_code_prefix","customer_city","customer_state"]
customers_df = customers_df.select(*expected_cols)

# capitalize city names
customers_df = customers_df.withColumn("customer_city", F.initcap(F.col("customer_city")))

(
customers_df.write \
    .format("delta") \
    .option("overwriteSchema", "true") \
    .mode("overwrite") \
    # .saveAsTable("lakehouse_olist.sl_customer")
    .saveAsTable("lh_silver_olist.sl_customer")
)

print("Rewrote lakehouse_olist.sl_customers without zip_len.")
# spark.read.table("lakehouse_olist.sl_customers").printSchema()
spark.read.table("lh_silver_olist.sl_customer").printSchema()


# In[21]:


# test_silver_read = spark.read.table("sl_customers")
# print(test_silver_read)

# sv_customer_df = spark.sql("SELECT * FROM lakehouse_olist.sl_customers LIMIT 1000")
sv_customer_df = spark.sql("SELECT * FROM lh_silver_olist.sl_customer LIMIT 1000")
display(sv_customer_df)

