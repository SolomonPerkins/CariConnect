import pandas as pd
from src.model_training import (
    train_tfidf, 
    train_scaler, 
    train_svd, 
    train_birch, 
    save_model
)

def main():
    try:
        # Load the dataset
        df = pd.read_csv('data/isbndb-caribbean-books.csv', encoding= 'latin1')

        # Fill NaN values
        df['title'] = df['title'].fillna('')
        df['subjects'] = df['subjects'].fillna('')
        df['synopsis'] = df['synopsis'].fillna('')

        # Merge text fields
        df['merged'] = df['title'] + ' ' + df['subjects'] + ' ' + df['synopsis']

        # Train TF-IDF
        print("Training TF-IDF...")
        tfidf, tfidf_matrix = train_tfidf(df, 'merged')
        save_model(tfidf, 'models/tfidf_vectorizer.pkl')

        # Train MaxAbsScaler
        print("Training MaxAbsScaler...")
        scaler, scaled_matrix = train_scaler(tfidf_matrix)
        save_model(scaler, 'models/maxabs_scaler.pkl')

        # Train TruncatedSVD
        print("Training TruncatedSVD...")
        svd, reduced_matrix = train_svd(scaled_matrix)
        save_model(svd, 'models/svd_model.pkl')

        # Train BIRCH
        print("Training BIRCH...")
        birch = train_birch(reduced_matrix)
        save_model(birch, 'models/birch_model.pkl')

        # Save reduced data
        reduced_df = pd.DataFrame(reduced_matrix, index=df.index)
        reduced_df.to_pickle('data/processed/reduced_data.pkl')

        print("All models trained and saved successfully!")

    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()
