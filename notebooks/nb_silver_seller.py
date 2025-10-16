#!/usr/bin/env python
# coding: utf-8

# ## nb_silver_seller
# 
# New notebook

# ### Data cleaning seller_data

# In[1]:


from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window as W
from unidecode import unidecode


# ### Reading the seller_data.csv file

# In[2]:


# df = spark.read.format('csv').options(header='True',inferSchema = 'True').load(
# 'abfss://f4041c98-2c4f-44a4-82ec-ab26f8d5d0b3@onelake.dfs.fabric.microsoft.com/d3c25d34-1652-45f9-b4f3-58b4aedb4efa/Files/bronze/olist_sellers_dataset.csv')
df = spark.read.table("br_sellers")


# In[3]:


# Cache the original columns for comparison
before_df = df.select(
    "seller_id",
    F.col("seller_city").alias("city_before"),
    F.col("seller_state").alias("state_before")
)


# # Profling

# In[4]:


# Count total rows
row_count = df.count()
print(f'Total_rows: {row_count}')


# In[5]:


# Count any NULL values in each column
null_counts = df.select([sum(col(c).isNull().cast('int')).alias(c) for c in df.columns])
null_counts.show()


# In[6]:


duplicates = (
    df.groupBy(df.columns).count().filter(col('count')>1)
)
duplicates.show()


# In[7]:


# ----------------------------------------------------------------
print("\n=== BASIC PROFILING ===")

profiling_df = (
    df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(f"{c}_nulls") for c in df.columns] +
        [F.countDistinct(c).alias(f"{c}_unique") for c in df.columns]
    )
)
profiling_df.show(truncate=False)

# ZIP code diagnostics
zip_len_issue = df.filter(F.length(F.col("seller_zip_code_prefix").cast("string")) != 5).count()
zip_non_numeric = df.filter(~F.col("seller_zip_code_prefix").cast("string").rlike("^[0-9]+$")).count()
city_with_digits = df.filter(F.col("seller_city").rlike(r"\d")).count()
bad_state_fmt = df.filter(~F.col("seller_state").rlike("^[A-Z]{2}$")).count()

print("\n=== RAW DATA QUALITY FLAGS ===")
print(f"ZIPs not 5 digits: {zip_len_issue}")
print(f"ZIPs non-numeric: {zip_non_numeric}")
print(f"City names with digits: {city_with_digits}")
print(f"States not 2-letter uppercase: {bad_state_fmt}")


# In[8]:


# seller_city contains slashes, commas, parantheses, email address-like text and composite names like "novo hamburgo, rio grade do sul, brasil"

df.filter(~F.col("seller_city").rlike(r"^[a-zA-Z\s\-]+$")) \
  .select("seller_id", "seller_city", "seller_state") \
  .distinct() \
  .show(30, truncate=False)


# In[9]:


df.printSchema()


# # Cleaning

# In[10]:


# Trim white spaces & normalize case
df = (
    df
    .withColumn('seller_city', lower(trim(col('seller_city'))))
    .withColumn('seller_state', upper(trim(col('seller_state'))))
)


# In[11]:


# Check specific known issue (ZIP 22790 should map to 'rio de janeiro')
df_filtered = df.filter(col('seller_zip_code_prefix') == 22790)
df_filtered.show(truncate=False)


# In[12]:


# Fix manually identified error
df = df.withColumn(
    'seller_city',
    when(col('seller_city') == '04482255', 'rio de janeiro').otherwise(col('seller_city'))
)


# In[13]:


# Display same zip code of 22790. seller_city for 22790 should be rio de janeiro
df_filtered = df.filter(col('seller_zip_code_prefix') == 22790)
df_filtered.show()


# ### Brazilian ZIP Codes (CEP)
# 
# Format: NNNNN-NNN (5 digits + hyphen + 3 digits)
# 
# Example: 01001-000 → São Paulo
# 
# Numeric part:
# 
# The first five digits are the main code for the city/region.
# 
# The last three digits indicate a more specific area or street.
# 
# Length:
# 
# Strictly numeric, 5 digits minimum (before the hyphen).
# 
# Officially, the full code has 8 digits, usually written as NNNNN-NNN.
# 
# From the data set, the prefix for should be 5 digits. example sau paulo 4195 is missing a prefix "0" 

# ### Count missing prefix 'zero'

# In[14]:


count_lt6 = df.filter(length(col('seller_zip_code_prefix').cast('string'))>5).count()
count_lt5 = df.filter(length(col('seller_zip_code_prefix').cast('string'))<5).count()
count_lt4 = df.filter(length(col('seller_zip_code_prefix').cast('string'))<4).count()
count_lt3 = df.filter(length(col('seller_zip_code_prefix').cast('string'))<3).count()
print(f'Rows with > 5 digits : {count_lt6}')
print(f'Rows with < 5 digits : {count_lt5}')
print(f'Rows with < 4 digits : {count_lt4}')
print(f'Rows with < 3 digits : {count_lt3}')


# 

# ### Total 1027 rows with missing prefix '0'

# In[15]:


# Show sample of missing prefix '0'
Missing_zero = df.filter(length(col('seller_zip_code_prefix').cast('string'))<5)
display(Missing_zero.limit(5))


# ### Left pad to add prefix zero so all prefix have 5 digits

# In[16]:


df = df.withColumn(
    'seller_zip_code_prefix',lpad(col('seller_zip_code_prefix').cast('string'),5, '0')

)
display(df.limit(5))


# In[17]:


# Verify all prefix are 5 digits
missing_zero = df.filter(length(col('seller_zip_code_prefix'))< 5) 
count_missing = missing_zero.count()
print (F'Count of prefix less than 5 digits: {count_missing}')


# ### City name cleaning

# In[18]:


print("\n=== ADDITIONAL CLEANING & NORMALIZATION ===")

# ---------------------------------------------------------------
# City normalization (unidecode + deep cleaning)
# ---------------------------------------------------------------
# 1️⃣ Normalize accents and lowercase
udf_unaccent = F.udf(lambda s: unidecode(s) if s else None)
df = df.withColumn("seller_city", udf_unaccent("seller_city"))
df = df.withColumn("seller_city", F.lower(F.trim(F.col("seller_city"))))

# 2️⃣ Remove digits and text inside parentheses
df = df.withColumn("seller_city", F.regexp_replace("seller_city", r"\d", ""))
df = df.withColumn("seller_city", F.regexp_replace("seller_city", r"\(.*?\)", ""))

# 3️⃣ Keep only the first token before ',' or '/' or '\'
df = df.withColumn("seller_city", F.regexp_replace("seller_city", r"[,/\\].*", ""))

# 4️⃣ Remove email-like entries (replace with 'unknown')
df = df.withColumn(
    "seller_city",
    F.when(F.col("seller_city").rlike("@"), F.lit("unknown"))
     .otherwise(F.col("seller_city"))
)

# 5️⃣ Remove non-letter characters except spaces/hyphens
df = df.withColumn("seller_city", F.regexp_replace("seller_city", r"[^a-z\s\-]", ""))

# 6️⃣ Collapse multiple spaces and trim
df = df.withColumn("seller_city", F.regexp_replace("seller_city", r"\s+", " "))
df = df.withColumn("seller_city", F.trim(F.col("seller_city")))

# ---------------------------------------------------------------
# State normalization
# ---------------------------------------------------------------
df = df.withColumn("seller_state", F.upper(F.trim(F.col("seller_state"))))

print("✅ Seller city/state cleaning & normalization complete.")


# In[19]:


# ================================================================
# CITY & STATE CLEANING VERIFICATION (before vs after)
# ================================================================

print("\n=== CLEANING CHANGE TRACKER: City & State ===")

# Join before vs after
compare_df = (
    df.alias("after")
      .join(before_df.alias("before"), "seller_id")
      .select(
          "seller_id",
          F.col("before.city_before"),
          F.initcap(F.col("after.seller_city").alias("city_after")),
          F.col("before.state_before"),
          F.col("after.seller_state").alias("state_after")
      )
)

# Rows where either city or state changed
diff_df = compare_df.filter(
    (F.col("city_before") != F.col("city_after")) |
    (F.col("state_before") != F.col("state_after"))
)

total_changed = diff_df.count()
print(f"Total rows changed (city or state): {total_changed}")

# --- Pattern-based subsets -------------------------------------
print("\n--- Examples: had slashes ---")
diff_df.filter(F.col("city_before").rlike("[/\\\\]")).show(10, truncate=False)

print("\n--- Examples: had commas ---")
diff_df.filter(F.col("city_before").contains(",")).show(10, truncate=False)

print("\n--- Examples: had parentheses ---")
diff_df.filter(F.col("city_before").rlike("\\(")).show(10, truncate=False)

print("\n--- Examples: contained digits ---")
diff_df.filter(F.col("city_before").rlike("\\d")).show(10, truncate=False)

print("\n--- Examples: looked like email ---")
diff_df.filter(F.col("city_before").rlike("@")).show(10, truncate=False)


# # Validation

# In[ ]:


# POST-CLEANING VALIDATION & ASSERTIONS

print("\n=== POST-CLEANING VALIDATION & ASSERTIONS ===")

# 1) Data quality assertions
invalid_zip = df.filter(~F.col("seller_zip_code_prefix").rlike("^[0-9]{5}$")).count()
city_with_digits = df.filter(F.col("seller_city").rlike(r"\d")).count()
invalid_states = df.filter(~F.col("seller_state").rlike("^[A-Z]{2}$")).count()
dupes = df.groupBy("seller_id").count().filter("count > 1").count()

print(f"ZIPs invalid format (<5 or non-numeric): {invalid_zip}")
print(f"City names still containing digits: {city_with_digits}")
print(f"Invalid state codes (not 2-letter uppercase): {invalid_states}")
print(f"Duplicate seller_id rows: {dupes}")

# ASSERTIONS
assert invalid_zip == 0, f"❌ Found {invalid_zip} invalid ZIP prefixes"
assert city_with_digits == 0, f"❌ Found {city_with_digits} cities still containing digits"
assert invalid_states == 0, f"❌ Found {invalid_states} invalid state codes"
assert dupes == 0, f"❌ Found {dupes} duplicate seller_id rows"

print("✅ Basic format & duplication checks passed.")


# 

# In[ ]:


# 2) Referential integrity check (Seller ZIP vs Geolocation)
# ---------------------------------------------------------------
print("\n=== REFERENTIAL INTEGRITY CHECK ===")

geo = spark.table("sl_geolocation").select("geolocation_zip_code_prefix").distinct()

# Left anti join to find sellers whose ZIPs are missing in geolocation
missing_zip = df.join(
    geo,
    df["seller_zip_code_prefix"] == geo["geolocation_zip_code_prefix"],
    "left_anti"
)

count_missing = missing_zip.count()
if count_missing > 0:
    print(f"⚠️ Sellers with unmatched ZIP prefixes: {count_missing}")
    missing_zip.select("seller_zip_code_prefix", "seller_city", "seller_state").show(10, truncate=False)
else:
    print("✅ All seller ZIP prefixes successfully match entries in sl_geolocation")


# In[ ]:


# ================================================================
# POST-CLEANING VALIDATION (Hardcore Assertions Only)
# ================================================================

print("\n=== SELLERS: HARD VALIDATION ASSERTIONS ===")

# ---------------------------------------------------------------
# 1) ZIP FORMAT CHECKS
# ---------------------------------------------------------------
zip_len_lt5 = df.filter(F.length(F.col("seller_zip_code_prefix")) < 5).count()
zip_bad_fmt  = df.filter(~F.col("seller_zip_code_prefix").rlike("^[0-9]{5}$")).count()
assert (zip_len_lt5 + zip_bad_fmt) == 0, f"❌ ZIP format issues: short={zip_len_lt5}, bad_fmt={zip_bad_fmt}"

# ---------------------------------------------------------------
# 2) STATE CODE VALIDITY
# ---------------------------------------------------------------
valid_states = ["AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS","MG",
                "PA","PB","PR","PE","PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO"]

state_bad_fmt = df.filter(~F.col("seller_state").rlike("^[A-Z]{2}$")).count()
state_invalid = df.filter(~F.col("seller_state").isin(valid_states)).count()
assert state_bad_fmt == 0, f"❌ Found {state_bad_fmt} malformed state codes"
assert state_invalid == 0, f"❌ Found {state_invalid} non-Brazilian state codes"

# ---------------------------------------------------------------
# 3) CITY NAME SANITY
# ---------------------------------------------------------------
city_invalid = df.filter(~F.col("seller_city").rlike(r"^[a-zA-Z\s\-]+$")).count()
assert city_invalid == 0, f"❌ Found {city_invalid} invalid city names (non-alphabetic)"

# ---------------------------------------------------------------
# 4) DUPLICATES
# ---------------------------------------------------------------
dupes = df.groupBy("seller_id").count().filter("count > 1").count()
assert dupes == 0, f"❌ Found {dupes} duplicate seller_id records"

# ---------------------------------------------------------------
# 5) REFERENTIAL INTEGRITY (vs sl_geolocation)
# ---------------------------------------------------------------
#geo = spark.table("sl_geolocation").select("geolocation_zip_code_prefix").distinct()

#missing_geo = df.join(
   # F.broadcast(geo),
  #  df.seller_zip_code_prefix == geo.geolocation_zip_code_prefix,
  #  "left_anti"
#).count()
#assert missing_geo == 0, f"❌ Found {missing_geo} seller ZIPs not in sl_geolocation"

print("\n=== REFERENTIAL INTEGRITY CHECK (SOFT WARNING) ===")

geo_silver = spark.table("sl_geolocation").select(
    F.col("geolocation_zip_code_prefix").alias("geo_zip")
).distinct()

# Find seller ZIPs missing from Silver geolocation
missing_geo_df = (
    df.join(
        F.broadcast(geo_silver),
        df.seller_zip_code_prefix == geo_silver.geo_zip,
        "left_anti"
    )
    .select("seller_zip_code_prefix", "seller_city", "seller_state")
    .distinct()
)

missing_count = missing_geo_df.count()
missing_zip_list = []  # always define the variable

if missing_count > 0:
    print(f"⚠️ WARNING: {missing_count} seller ZIPs not found in sl_geolocation.")
    missing_geo_df.show(10, truncate=False)

    # Build list of missing ZIPs
    missing_zip_list = [r["seller_zip_code_prefix"] for r in missing_geo_df.collect()]
    print(f"Missing ZIPs → {missing_zip_list}")
else:
    print("✅ All seller ZIP prefixes successfully match entries in sl_geolocation.")

# ---------------------------------------------------------------
# 6) CHECK WHETHER THESE ZIPs EXIST IN BRONZE GEOLOCATION
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

print("\n✅ Referential integrity check complete (warnings logged, pipeline not stopped).")

# ---------------------------------------------------------------
# 6) FINAL SUCCESS MESSAGE
# ---------------------------------------------------------------
print("✅ ALL SELLER VALIDATION ASSERTIONS PASSED — TABLE IS SILVER-READY.")


# In[ ]:


# should be 0 if the 27 problematic rows (as seen in profiling) are fixed 
df.filter(~F.col("seller_city").rlike(r"^[a-zA-Z\s\-]+$")) \
  .select("seller_id", "seller_city", "seller_state") \
  .distinct() \
  .show(30, truncate=False)


# # Writing Data Frame to Table

# In[ ]:


# df.write.format('delta').mode('overwrite').saveAsTable('lakehouse_olist.sl_seller')
df.write.format('delta').mode('overwrite').saveAsTable('lh_silver_olist.sl_seller')


# In[ ]:


df.printSchema()

