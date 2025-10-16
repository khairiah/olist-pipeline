#!/usr/bin/env python
# coding: utf-8

# ## nb_silver_geolocation
# 
# New notebook

# ## Data Cleaning
# 

# In[1]:


from pyspark.sql.functions import *
from pyspark.sql.types import *

from pyspark.sql import types as T
from pyspark.sql import functions as F

from pyspark.sql.window import Window    # <-- defines Window for row_number / rank, etc.
import unicodedata


# ### Reading the geolocation data 

# In[2]:


# geolocation_df = spark.read.format('csv').options(header='True',inferSchema = 'False').load(
# 'abfss://f4041c98-2c4f-44a4-82ec-ab26f8d5d0b3@onelake.dfs.fabric.microsoft.com/d3c25d34-1652-45f9-b4f3-58b4aedb4efa/Files/bronze/olist_geolocation_dataset.csv')
geolocation_df = spark.read.table("br_geolocation")


# In[3]:


# Count total rows
print(F'Total Rows = {geolocation_df.count()}')


# In[4]:


# Display data frame
display(geolocation_df.limit(4))


# In[5]:


# Trim white spaces and title case city names, upper case state names
geolocation_df = geolocation_df.withColumn('geolocation_city', initcap(trim(col('geolocation_city')))) \
    .withColumn('geolocation_state', upper(trim(col('geolocation_state'))))

display(geolocation_df.limit(4))


# In[6]:


# Count any NULL values in each column
null_counts = geolocation_df.select([sum(col(c).isNull().cast('int')).alias(c) for c in geolocation_df.columns])
null_counts.show()


# In[7]:


geolocation_df.printSchema()


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

# 

# ### Count geolocation_zip_code_prefix that does not meet 5 digits format

# In[8]:


count_lt6 = geolocation_df.filter(length(col('geolocation_zip_code_prefix').cast('string'))>5).count()
count_lt5 = geolocation_df.filter(length(col('geolocation_zip_code_prefix').cast('string'))<5).count()
count_lt4 = geolocation_df.filter(length(col('geolocation_zip_code_prefix').cast('string'))<4).count()
count_lt3 = geolocation_df.filter(length(col('geolocation_zip_code_prefix').cast('string'))<3).count()
print(f'Rows with > 5 digits : {count_lt6}')
print(f'Rows with < 5 digits : {count_lt5}')
print(f'Rows with < 4 digits : {count_lt4}')
print(f'Rows with < 3 digits : {count_lt3}')


# ## Sample of Geolocation zip with 4 digits

# In[9]:


missing_zero = geolocation_df.filter(col('geolocation_zip_code_prefix') < 10000) 
display(missing_zero.limit(10))


# 

# ### Cleaning
# - Add a leading zero to all zip prefix that are < 10000 in geolocation_zip_code_prefix

# In[10]:


print("\n=== DATA CLEANING AND STANDARDIZATION ===")

# 2.1 Trim all strings to remove leading/trailing spaces. For removing invisible whitespace that can cause false duplicates.
geolocation_df = geolocation_df.toDF(*[c.strip() for c in geolocation_df.columns])
for c in geolocation_df.columns:
    geolocation_df = geolocation_df.withColumn(c, F.trim(F.col(c)))

# 2.2 Normalize case (city = lowercase, state = uppercase). Consistent casing prevents mismatched joins.
geolocation_df = geolocation_df.withColumn("geolocation_city",  F.lower(F.col("geolocation_city")))
geolocation_df = geolocation_df.withColumn("geolocation_state", F.upper(F.col("geolocation_state")))

# 2.3 Remove accents, punctuation, and excess spaces from city names
def strip_accents_py(s):
    if s is None: return s
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

strip_accents = F.udf(strip_accents_py, StringType())

# “São Paulo” → “sao paulo”, “rio-de-janeiro” → “rio de janeiro” for standardization.
geolocation_df = geolocation_df.withColumn("geolocation_city", strip_accents(F.col("geolocation_city")))
geolocation_df = geolocation_df.withColumn("geolocation_city", F.regexp_replace(F.col("geolocation_city"), "[^a-zA-Z\\s\\-]", ""))
geolocation_df = geolocation_df.withColumn("geolocation_city", F.regexp_replace(F.col("geolocation_city"), "\\s+", " "))

# Ensure ZIP prefixes are 5-digit numeric strings. For consistency for referential joins later.
geolocation_df = geolocation_df.withColumn(
    "geolocation_zip_code_prefix", F.lpad(
        F.regexp_replace(F.col("geolocation_zip_code_prefix").cast("string"), "[^0-9]", ""),
        5,
        "0"
    )
)

# Cast latitude/longitude to numeric and flag invalids
geolocation_df = geolocation_df.withColumn("lat_d", F.col("geolocation_lat").cast(DoubleType()))
geolocation_df = geolocation_df.withColumn("lng_d", F.col("geolocation_lng").cast(DoubleType()))

# Brazil valid latitude range (-35.0 to +6.0), longitude range (-75.0 to -30.0). 
# Chosen bounds below gives a safe buffer to capture all valid Brazilian coordinates while excluding outliers (E.g. 0, 999, etc)
invalid_coords = geolocation_df.filter(
    F.col("lat_d").isNull() | F.col("lng_d").isNull() |
    (~F.col("lat_d").between(-35.0, 6.0)) |
    (~F.col("lng_d").between(-75.0, -30.0))
)

invalid_count = invalid_coords.count()
print(f"Invalid or out-of-range coordinates identified: {invalid_count}")
invalid_coords.select("geolocation_zip_code_prefix", "geolocation_city", "geolocation_state", "lat_d", "lng_d").show(20, truncate=False)

# Quarantine invalid coordinates for auditing to maintain traceability. Invalid coordinates (e.g., 0, 9999, null) are quarantined.
if invalid_count > 0:
    invalid_coords.write.mode("overwrite").saveAsTable("qr_geolocation") # when building pipeline, change to append
    print(f"Quarantined {invalid_count} invalid rows to qr_geolocation")

# Keep only valid coordinates for Silver
geolocation_valid = geolocation_df.subtract(invalid_coords)
print(f"Remaining valid rows: {geolocation_valid.count()}")


# In[11]:


print("\n=== CANONICALIZATION AND AGGREGATION ===")

# 3.1 Determine canonical (city, state) for each ZIP prefix by frequency ("mode")
zip_city_state_counts = (
    geolocation_valid
    .groupBy("geolocation_zip_code_prefix", "geolocation_city", "geolocation_state")
    .count()
)

w = Window.partitionBy("geolocation_zip_code_prefix").orderBy(F.desc("count"), F.asc("geolocation_city"))
canon = (
    zip_city_state_counts
    .withColumn("rn", F.row_number().over(w))
    .filter(F.col("rn") == 1)
    .select(
        F.col("geolocation_zip_code_prefix").alias("zip"),
        F.col("geolocation_city").alias("canonical_city"),
        F.col("geolocation_state").alias("canonical_state")
    )
)
# Justification: Picks the most frequently occurring city/state combination for each ZIP prefix.

# Aggregate coordinates (average per ZIP prefix)
avg_coords = (
    geolocation_valid
    .groupBy("geolocation_zip_code_prefix")
    .agg(
        F.avg("lat_d").alias("avg_lat"),
        F.avg("lng_d").alias("avg_lng")
    )
)
# Averaging yields a representative “centroid” coordinate per postal prefix.
# Join canonical city/state + averaged coordinates
geo_silver = (
    avg_coords.join(canon, avg_coords.geolocation_zip_code_prefix == canon.zip, "left")
    .select(
        F.col("geolocation_zip_code_prefix"),
        F.col("canonical_city").alias("geolocation_city"),
        F.col("canonical_state").alias("geolocation_state"),
        "avg_lat", "avg_lng"
    )
)

print(f"Rows in final Silver dataset (unique ZIP prefixes): {geo_silver.count()}")
geo_silver.show(10, truncate=False)


# ## Validation and data quality checks

# In[12]:


print("\n=== VALIDATION AND DATA-QUALITY CHECKS ===")

# 1. Nulls in critical columns
nulls = geo_silver.filter(
    F.col("geolocation_zip_code_prefix").isNull() |
    F.col("geolocation_city").isNull() |
    F.col("geolocation_state").isNull() |
    F.col("avg_lat").isNull() |
    F.col("avg_lng").isNull()
).count()
print(f"Null critical fields: {nulls}")
assert nulls == 0, "Nulls found in critical columns."

# 2. ZIP format and length (5) validation
zip_len_bad = geo_silver.filter(F.length(F.col("geolocation_zip_code_prefix")) < 5).count() # validate completeness of ZIP prefix 
print(f"ZIP prefixes shorter than 5 digits: {zip_len_bad}")
zip_bad = geo_silver.filter(~F.col("geolocation_zip_code_prefix").rlike("^[0-9]{5}$")).count() # ensure format correctness - exactly 6 digits
print(f"Invalid ZIP format count (non-numeric or wrong length): {zip_bad}")
assert (zip_len_bad + zip_bad) == 0, "Invalid ZIP prefix length or format detected." # combined assertion. guarantees all ZIPs are standardized and usable for joins

# 3. State format
state_bad = geo_silver.filter(~F.col("geolocation_state").rlike("^[A-Z]{2}$")).count()
print(f"Invalid state format count: {state_bad}")
assert state_bad == 0, "Invalid state code format."

# 4. City text
city_bad = geo_silver.filter(
    (F.col("geolocation_city") == "") | 
    (~F.col("geolocation_city").rlike("^[a-zA-Z\\s\\-]+$"))
).count()
print(f"Invalid city names count: {city_bad}")
assert city_bad == 0, "Invalid city text format."

# 5. Coordinate ranges (Brazil)
coord_bad = geo_silver.filter(~(F.col("avg_lat").between(-35.0, 6.0) & F.col("avg_lng").between(-75.0, -30.0))).count()
print(f"Out-of-range coordinates: {coord_bad}")
assert coord_bad == 0, "Coordinates out of Brazil range."

# 6. Uniqueness of ZIP prefix
dup_zip = geo_silver.groupBy("geolocation_zip_code_prefix").count().filter("count > 1").count()
print(f"Duplicate ZIP prefixes: {dup_zip}")
assert dup_zip == 0, "ZIP prefixes not unique."

print("All data-quality checks passed successfully.")


# ## Enrichment - Practical Use for Geolocation co-ordinates
# 
# Standard GPS datasets usually store coordinates with 5–6 decimal places.
# 
# Example: -23.550520, -46.633308 → São Paulo center (6 decimal places).
# 
# 14 decimal places is overkill — it’s far beyond any GPS accuracy and just adds unnecessary precision.
# 
# Recommended for Brazil / mapping purposes: 5–6 decimal places for meters-level accuracy.
# 
# Added: For subsequent Power BI mapping visual: 4-6 decimal places (~10cm - 1m accuracy), ignores trailing zeros as Power BI treats values numerically. It does not care whether -23.56 shows as -23.560000. It only cares that the numeric value is accurate enough.

# In[17]:


geo_silver = (
    avg_coords.join(canon, avg_coords.geolocation_zip_code_prefix == canon.zip, "left")
    .select(
        F.col("geolocation_zip_code_prefix"),
        F.initcap(F.col("canonical_city")).alias("geolocation_city"),
        F.col("canonical_state").alias("geolocation_state"),
        F.round(F.col("avg_lat"), 6).alias("avg_lat"),
        F.round(F.col("avg_lng"), 6).alias("avg_lng")
    )
)

print(f"Rows in final Silver dataset (unique ZIP prefixes): {geo_silver.count()}")
geo_silver.show(10, truncate=False)


# In[18]:


geo_silver.printSchema()


# ### Write geolocation data frame to table

# In[19]:


(
    geo_silver.write \
    .format("delta") \
    .option("overwriteSchema", "true") \
    .mode("overwrite") \
    # .saveAsTable("lakehouse_olist.sl_geolocation") \
    .saveAsTable("lh_silver_olist.sl_geolocation")
)

# geo_silver.write.format('delta').mode('overwrite').saveAsTable('lakehouse_olist.sl_geolocation')
# quarantine table written above


# In[ ]:


# for testing only
print("✅ sl_geolocation written successfully.")
#spark.sql("SHOW TABLES IN lakehouse_olist").show()
spark.sql("SELECT COUNT(*) FROM lh_silver_olist.sl_geolocation").show()

