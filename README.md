# Intelligent E-Commerce Recommender System
An intelligent product recommendation system built for e-commerce use cases, implementing multiple recommendation techniques through a user-friendly interactive web interface.

# Overview
This project delivers a fully functional recommender system that suggests products to users based on their preferences and past interactions. It was built as part of the AIE425 — Intelligent Recommender Systems course and covers both Collaborative Filtering and Content-Based Filtering paradigms, with several algorithms implemented across both categories.
Recommendation Methods
Collaborative Filtering

User-Based Collaborative Filtering — measures user similarity using Pearson Correlation (fully manual implementation)
Item-Based Collaborative Filtering — identifies similar items based on user rating patterns
Matrix Factorization (SVD) — decomposes the user-item interaction matrix to uncover latent factors

Content-Based Filtering

TF-IDF with Cosine Similarity — recommends items based on textual product features
Additional content-based method — further expands coverage using item attribute analysis

# Technologies Used

Python
Streamlit (interactive web UI)
NumPy, Pandas
Scikit-learn (for selected utility functions)

# Interface
The application is built with Streamlit, providing an intuitive dashboard where users can select a recommendation method, choose a user or product, and instantly view personalized recommendations.

# Purpose
This project was developed as the final project for the AIE425 Intelligent Recommender Systems course. It demonstrates a practical understanding of core recommendation algorithms and the ability to deploy them in an interactive application.
