# AICTE EDUNET TREE CLASSIFICATION


This repository contains the "Tree Intelligence Assistant," a comprehensive tool for tree species recommendation and identification. The project integrates a K-Nearest Neighbors model for location-based recommendations and a Convolutional Neural Network (CNN) for image-based species identification, all packaged into an interactive Streamlit web application.

## Features

The Tree Intelligence Assistant offers three main functionalities:

1.  **🌲 Recommend Trees by Location:** Based on user-provided latitude, longitude, tree diameter, and native status, this feature recommends the top 5 most suitable tree species for that specific area using a k-NN model.
2.  **📍 Find Locations for a Tree:** For a selected tree species, the application displays the top 10 most common cities and states where it is found, based on the project's dataset.
3.  **📷 Identify Tree from Image:** Users can upload an image of a tree, and the application's CNN model will predict its species. It provides the top prediction with a confidence score and also lists the top 3 most likely species. Following identification, it shows the common locations for the predicted species.

## Dataset

The project utilizes two primary data sources:

*   **Image Dataset:** Located in the `Tree_Species_Dataset/` directory, this dataset contains 1,454 images across 30 different tree species native to India. The species include Amla, Banyan, Coconut, Mango, Neem, and more. Each species has its own sub-directory.
*   **Tabular Dataset (`tree_data.pkl`):** A pickled Pandas DataFrame that contains information about individual trees, including their common name, latitude, longitude, diameter, native status, city, and state. This dataset is used to train the location-based recommendation model.

## Models

Two machine learning models power the application:

1.  **K-Nearest Neighbors (k-NN) Recommender (`nn_model.joblib`):** A k-NN model trained on the tabular tree data. It finds trees with similar geographical and physical characteristics to recommend species suitable for a given location. Data is preprocessed using a standard scaler (`scaler.joblib`).
2.  **Convolutional Neural Network (CNN) Classifier (`basic_cnn_tree_species.h5`):** A custom CNN trained on the `Tree_Species_Dataset`. It is designed to classify a tree's species from an image. The model architecture consists of three convolutional layers followed by max-pooling, and a dense head with dropout for regularization.

The development and training process for the CNN is detailed in the `tree_CNN.ipynb` notebook.

## File Structure

```
.
├── Tree_Species_Dataset/       # Directory with 30 sub-folders of tree images
├── basic_cnn_tree_species.h5   # Trained CNN model for image classification
├── nn_model.joblib             # Trained k-NN model for recommendation
├── scaler.joblib               # Scaler for the k-NN model
├── streamlit_integrated.py     # The main Streamlit application script
├── tree_CNN.ipynb              # Jupyter Notebook for CNN model development
└── tree_data.pkl               # Tabular data for the recommendation system
```

## How to Run the Application

To run the Tree Intelligence Assistant locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/watcher141/aicte_edunet_tree_classification-final-submission.git
    cd aicte_edunet_tree_classification-final-submission
    ```

2.  **Install the required Python libraries:**
    ```bash
    pip install streamlit pandas numpy scikit-learn tensorflow Pillow
    ```

3.  **Ensure all model and data files** (`basic_cnn_tree_species.h5`, `nn_model.joblib`, `scaler.joblib`, `tree_data.pkl`) are in the root directory alongside `streamlit_integrated.py`.

4.  **Run the Streamlit app:**
    ```bash
    streamlit run streamlit_integrated.py
    ```

The application will open in your default web browser. You can then use the sidebar to navigate between the different modes.
