## Overview

This project implements a multi-stage ETL pipeline using a Lakehouse Medallion architecture (Bronze, Silver, Gold). Raw data is incrementally ingested, transformed using PySpark, and stored as progressively cleaner and more analytics-ready datasets.

The pipeline demonstrates how large-scale data can be structured, validated, and curated using lakehouses while maintaining clear separation of concerns across data layers.

## Dataset

This project uses the public Olist Brazilian E-commerce Dataset retrieved using the Kaggle API, which contains:
- Orders, customers, sellers, products, payments, and reviews
- Over 100k orders across multiple years

## Architecture

Bronze → Silver → Gold

Bronze: Raw, immutable ingested data

Silver: Cleaned, validated, and standardized data

Gold: Business-ready, aggregated datasets for analytics and reporting

All layers reside in the same lakehouse, but each represents a logical stage in the ETL process.

## Pipeline Flow

**Ingestion (Bronze Layer)**
- Raw data is loaded into the Bronze layer with minimal processing.
- Data is stored in its original structure to preserve lineage and enable reprocessing.

**Transformation to Silver:** 
PySpark is used to:
- Clean and normalize data
- Handle missing or invalid values
- Apply basic business rules

**Transformation to Gold:**
Additional PySpark transformations are applied on Silver data to:
- Aggregate metrics 
- Denormalize tables where needed
- Prepare analytics- and reporting-ready datasets

The Gold Layer enables analysis of key operational questions pertaining to delivery fulfillment, including but not limited to:
- On-time vs late delivery rates by seller and product category
- Average delivery time compared to estimated delivery dates
- Impact of late deliveries on customer review scores
- Order volume and fulfillment trends over time

## Data Quality & Testing
PyTest is used to validate Silver and Gold layer transformations, including:
- Schema validation (expected columns and data types)
- Null and duplicate checks on key identifiers
- Business rule validation (e.g. delivery date ≥ order date)
- Row count checks to detect data loss during transformations

## Technologies Used
- Microsoft Fabric (Data Pipeline for automating data ingestion)
- Apache Spark (PySpark) & SQL for data transformations & querying
- PyTest (for unit testing)
- Lakehouse / Delta-style storage (for loading of data)

## Outcome
The final Gold datasets are structured for:
BI dashboard reporting (specifically focused on delivery fulfillment metrics)

## Key Lessons Learnt
- Deeper understanding of features available on Microsoft Fabric for data engineers/data analysts/data scientists
- Design of galaxy schema for dimensional modelling 
- Pipeline performance optimization techniques (e.g. partitioning, caching, broadcast joins, query tuning)
- Advanced SQL querying: Used joins, subqueries, CTES to answer business questions
- Writing PySpark code for big data transformations

## Future Enhancements
- Introduce Slowly Changing Dimensions (SCD Type 2)
- Implement automated tests using Great Expectations
