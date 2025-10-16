#!/usr/bin/env python
# coding: utf-8

# ## nb_silver_review
# 
# New notebook

# In[88]:


# ============================================
# Noelle's Notebook: Clean + Check
# br_reviews (Reviews) & br_payments (Payments)
# ============================================

# Install translation library (only runs once)

# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %pip install textblob
# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %pip install deep-translator


from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StructType, StructField, StringType
import pandas as pd
from textblob import TextBlob
from deep_translator import GoogleTranslator


# In[89]:


# Start Spark session
spark = SparkSession.builder.appName("NoelleCleaningEDA").getOrCreate()


# In[90]:


# ============================================
# PART 1: CLEAN REVIEWS
# ============================================

# Load existing table
reviews_df = spark.read.table("br_reviews")
reviews_df.printSchema()
reviews_df.show(5)
reviewCount = reviews_df.count()
print("Total reviews:", reviewCount)


# In[91]:


# remove duplicates
reviews_df = reviews_df.dropDuplicates(["review_id"])
# filter out invalid review scores
reviews_df = reviews_df.filter(F.col("review_score").between(1,5))
# add new column for yes or no review comment
reviews_df = reviews_df.withColumn("review_comment_present", F.when(F.col("review_comment_message").isNull(), "No").otherwise("Yes"))
reviews_df = reviews_df.withColumn("review_title_present", F.when(F.col("review_comment_title").isNull(), "No").otherwise("Yes"))
# check if review comment and title are both yes or no, count the rest as partial
reviews_df = reviews_df.withColumn("review_complete", 
                                   F.when((F.col("review_comment_present") == "Yes") & (F.col("review_title_present") == "Yes"), "Yes")
                                   .when((F.col("review_comment_present") == "No") & (F.col("review_title_present") == "No"), "No")
                                   .otherwise("Partial"))
# 
print("✅ Cleaned reviews rows:", reviews_df.count())
print("Removed ", reviewCount - reviews_df.count(), " rows with duplicate review_id")
reviews_df.printSchema()
reviews_df.show(5)


# In[ ]:


# Define schema for combined output
import time


schema = StructType([
    StructField("review_comment_message_en", StringType(), True),
    StructField("review_sentiment", StringType(), True),
    StructField("translation_status", StringType(), True)
])

@pandas_udf(schema)
def translate_and_sentiment(text_series: pd.Series) -> pd.DataFrame:
    text_series = text_series.fillna("")
    
    translated_list = []
    sentiment_list = []
    status_list = []

    for text in text_series:
        if text.strip() == "":
            translated_list.append("")
            sentiment_list.append("Neutral")
            status_list.append("empty")
        else:
            # Translation
            try:
                translated_text = GoogleTranslator(source='pt', target='en').translate(text)
                status_list.append("success")
                time.sleep(0.2)  # ~5 requests/sec max
            except Exception as e:
                translated_text = text  # fallback to original
                status_list.append(f"failed: {str(e)}")

            translated_list.append(translated_text)

            # Sentiment analysis
            try:
                polarity = TextBlob(translated_text).sentiment.polarity
                if polarity > 0:
                    sentiment_list.append("Positive")
                elif polarity < 0:
                    sentiment_list.append("Negative")
                else:
                    sentiment_list.append("Neutral")
            except Exception:
                sentiment_list.append("Neutral")

    return pd.DataFrame({
        "review_comment_message_en": translated_list,
        "review_sentiment": sentiment_list,
        "translation_status": status_list
    })

# Apply UDF
reviews_df = reviews_df.withColumn(
    "translation_and_sentiment",
    translate_and_sentiment(col("review_comment_message"))
)

# Split struct into separate columns
reviews_df = reviews_df.withColumn("review_comment_message_en", col("translation_and_sentiment.review_comment_message_en")) \
                       .withColumn("review_sentiment", col("translation_and_sentiment.review_sentiment")) \
                       .withColumn("translation_status", col("translation_and_sentiment.translation_status")) \
                       .drop("translation_and_sentiment")

# Show sample
reviews_df.select("review_comment_message", "review_comment_message_en", "review_sentiment", "translation_status").show(5, truncate=False)

reviews_df = reviews_df.cache()  # caches the computed column in memory
reviews_df.count()  # triggers computation


# In[ ]:


# print distribution of scores, with review completeness
reviews_df.groupBy("review_score", "review_complete").count().orderBy("review_score").show()

# print distribution of sentiments
reviews_df = reviews_df.repartition("review_sentiment")
reviews_df.groupBy("review_sentiment").count().show()


# In[97]:


# ---- Save ----
reviews_df.write.mode("overwrite").format("delta").saveAsTable("lh_silver_olist.sl_review")

# ---- Check & EDA ----
reviews_silver = spark.read.table("lh_silver_olist.sl_review")
reviews_silver.printSchema()
reviews_silver.show(5, truncate=False)
print("✅ Silver reviews rows:", reviews_silver.count())

