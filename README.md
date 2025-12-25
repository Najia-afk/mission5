# Project: Olist E-commerce Customer Segmentation
## Customer Experience Dashboard & Targeted Marketing Strategy

[![Docker](https://img.shields.io/badge/Docker-24.0+-blue.svg)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.12+-yellow.svg)](https://www.python.org/)

###  Project Context
Olist is a Brazilian e-commerce platform. This project involves analyzing their extensive database to develop customer segmentation and key performance indicators (KPIs) to improve marketing efficiency and customer experience.

###  Business & Technical Objectives
- **Customer Profiling**: Develop a robust segmentation using RFM (Recency, Frequency, Monetary) metrics.
- **Clustering Analysis**: Apply unsupervised learning (K-Means, DBSCAN) to identify hidden customer patterns.
- **Business Intelligence**: Create KPIs for a Customer Experience Dashboard to monitor delivery performance and satisfaction.

###  Technical Architecture
1. **Database Integration**: Direct connection to SQLite database for efficient querying.
2. **Feature Engineering**: Transformation of raw transaction data into behavioral features.
3. **Clustering Pipeline**: Comparison of multiple clustering algorithms with PCA for dimensionality reduction.
4. **Visualization**: Interactive maps and charts using Plotly and Seaborn.

---

###  Quick Start (Docker)

#### 1. Prerequisites
- Docker Desktop
- Docker Compose V2

#### 2. Launch the System
```bash
docker-compose up --build
```

#### 3. Access the Services
- **Jupyter Notebook**: [http://localhost:8885](http://localhost:8885) (Open mission5.ipynb)

---

###  Project Structure
```text
 mission5.ipynb       # Main analysis and segmentation notebook
 src/
    classes/         # Feature engineering, PCA, and clustering classes
    scripts/         # SQL connectors and visualization scripts
 dataset/             # Olist SQLite database (olist.db)
 docker-compose.yml   # Container orchestration
 Dockerfile           # Python environment
```

###  Key Insights
- **Geographic Concentration**: A significant portion of orders and high-value customers are concentrated in the Southeast region (São Paulo, Rio de Janeiro).
- **Delivery Impact**: Shipping time and cost are the primary drivers of customer satisfaction scores.
- **Segment Diversity**: Clustering reveals distinct groups, such as "Loyal High-Spenders" vs. "One-Time Bargain Hunters," requiring tailored marketing strategies.

---
*This project demonstrates the ability to transform raw e-commerce data into actionable business insights and advanced customer segments.*
