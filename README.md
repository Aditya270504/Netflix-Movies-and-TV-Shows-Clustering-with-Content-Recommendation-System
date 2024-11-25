# Netflix Movies and TV Shows Clustering with Content Recommendation System

This project explores the Netflix Movies and TV Shows dataset, performs data cleaning, visualization, and content clustering using multiple machine learning algorithms. It involves preprocessing textual data, feature extraction, dimensionality reduction, and clustering techniques such as K-Means and Agglomerative Clustering. The project also includes content-based recommendation based on cosine similarity.

---

## 1. Libraries and Modules Import

The essential libraries for data analysis, visualization, and machine learning are imported:
- **`pandas`, `numpy`**: Data manipulation and numerical computation.
- **`seaborn`, `matplotlib`**: Data visualization for plotting graphs and charts.
- **`nltk`**: Natural language processing for stopword removal, lemmatization, and tokenization.
- **`scikit-learn`**: For machine learning tasks like clustering, vectorization, and cosine similarity calculation.
- **`google.colab`**: For mounting Google Drive and accessing the dataset.

---

## 2. Data Loading and Overview

- **Loading Data**: The dataset (`netflix_titles.csv`) is loaded from Google Drive.
- **Initial Exploration**: The data is previewed using `head()`, `tail()`, `info()`, and `nunique()` to understand its structure and detect any missing or duplicated values.

---

## 3. Data Cleaning and Preprocessing

- **Handling Missing Values**: Missing values in `director`, `cast`, and `country` columns are filled with the label 'Unknown'. Rows with remaining missing values are dropped.
- **Feature Transformation**: The `country` column is simplified by keeping only the first country listed.
- **Outlier Detection**: Duplicated rows are identified and removed, and a count of unique values in each column is displayed.

---

## 4. Exploratory Data Analysis (EDA)

- **Rating Distribution**: Visualized the distribution of different movie ratings using a count plot and mapped ratings to broader categories (e.g., 'Adults', 'Kids').
- **Type Distribution**: A pie chart shows the distribution of movies and TV shows.
- **Top Directors**: The top 10 directors by number of movies/TV shows are visualized using a horizontal bar chart.
- **Top Genres**: The top 10 genres are visualized using a donut chart.
- **Top Countries**: The number of movies/TV shows in the top 10 countries is plotted by type.

---

## 5. Feature Engineering for Content Clustering

- **Text Feature Creation**: A new feature `content_clustering_attribute` is created by combining `listed_in` and `description`.
- **Text Cleaning**: Non-ASCII characters are removed, stopwords are filtered, punctuation is stripped, and verbs are lemmatized to clean the textual data.
- **Tokenization**: Tokenized the cleaned text using the `TweetTokenizer` from `nltk`.

---

## 6. Feature Extraction and Dimensionality Reduction

- **TF-IDF Vectorization**: The cleaned text data is transformed into numerical features using `TfidfVectorizer`.
- **PCA (Principal Component Analysis)**: Applied PCA to reduce the dimensionality of the data and keep 4000 components for further analysis.

---

## 7. Clustering

- **K-Means Clustering**: 
  - Performed the elbow method and silhouette analysis to find the optimal number of clusters for K-Means clustering.
  - K-Means was applied to the PCA-transformed data, and the clusters were added to the dataframe.
- **Agglomerative Clustering**: 
  - A hierarchical clustering approach was applied, and the dendrogram was visualized to choose an appropriate number of clusters.
  - The clusters were added to the dataframe and visualized.

---

## 8. Content-Based Recommendation System

- **Cosine Similarity**: 
  - Built a content-based recommendation system based on the `content_clustering_attribute`.
  - Used `CountVectorizer` to convert text data into a matrix, followed by calculating the cosine similarity between items.
- **Recommendation Function**: 
  - A function `recommend_10()` is created to recommend the top 10 similar content based on the cosine similarity for a given show.
