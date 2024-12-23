import streamlit as st
import torch
import pandas as pd
from model import WideAndDeepModel 
import pickle
from model import get_user_recommendations
from dash import html, dcc


# Fix file paths
MODEL_PATH = "music_recommender.pth"
DATA_PATH = "song_dataset.csv"
ENCODERS_PATH = "label_encoders.pkl"
@st.cache_resource
def load_model_and_data():
    try:
        song_df = pd.read_csv(DATA_PATH)
        
        with open(ENCODERS_PATH, 'rb') as f:
            le_dict = pickle.load(f)
            
        song_df['user_idx'] = le_dict[0].transform(song_df['user'])
        song_df['song_idx'] = le_dict[1].transform(song_df['song'])
        song_df['artist_idx'] = le_dict[2].transform(song_df['artist_name'])
        song_df['release_idx'] = le_dict[3].transform(song_df['release'])
        song_df['year_idx'] = le_dict[4].transform(song_df['year'].astype(str))
        
        model = WideAndDeepModel(
            num_users=len(le_dict[0].classes_),
            num_songs=len(le_dict[1].classes_),
            num_artists=len(le_dict[2].classes_),
            num_releases=len(le_dict[3].classes_),
            num_years=len(le_dict[4].classes_)
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        
        return model, song_df, le_dict
        
    except Exception as e:
        st.error(f"error: {str(e)}")
        return None, None, None

def get_user_recommendations(model, user_id, song_df, listened_songs, top_k=5):
    model.eval()
    with torch.no_grad():
        # Get unlistened songs
        available_songs = song_df[~song_df['song'].isin(listened_songs)]
        if available_songs.empty:
            return [], []
            
        # Get unique song information
        unique_songs = available_songs[['song_idx', 'song', 'artist_idx', 'release_idx', 'year_idx']].drop_duplicates('song_idx')
        unique_songs = unique_songs.sort_values('song_idx').reset_index(drop=True)
        num_songs = len(unique_songs)
        
        # Prepare features
        user_ids = torch.full((num_songs,), user_id, dtype=torch.long)
        song_ids = torch.tensor(unique_songs['song_idx'].values, dtype=torch.long)
        artist_ids = torch.tensor(unique_songs['artist_idx'].values, dtype=torch.long)
        release_ids = torch.tensor(unique_songs['release_idx'].values, dtype=torch.long)
        year_ids = torch.tensor(unique_songs['year_idx'].values, dtype=torch.long)
        
        # Batch processing
        batch_size = 1024
        all_scores = []
        
        for i in range(0, num_songs, batch_size):
            batch_end = min(i + batch_size, num_songs)
            batch_slice = slice(i, batch_end)
            
            # Wide features
            wide_features = torch.stack([
                user_ids[batch_slice].float(),
                song_ids[batch_slice].float(),
                artist_ids[batch_slice].float(),
                release_ids[batch_slice].float(),
                year_ids[batch_slice].float()
            ], dim=1)
            
            play_count = torch.zeros(batch_end - i)
            
            # Get predictions
            scores = model(
                wide_features,
                user_ids[batch_slice],
                song_ids[batch_slice],
                artist_ids[batch_slice],
                release_ids[batch_slice],
                year_ids[batch_slice],
                play_count
            )
            all_scores.append(scores)
        
        # Combine all batch scores
        scores = torch.cat(all_scores, dim=0).squeeze()
        
        # Get top-k recommendations
        top_k_scores, top_k_indices = torch.topk(scores, k=min(top_k, len(scores)))
        
        # Get original song IDs and convert indices to list
        indices = top_k_indices.cpu().tolist()
        recommended_songs = unique_songs.iloc[indices]['song'].values.tolist()
        scores_list = top_k_scores.cpu().tolist()
        
        return recommended_songs, scores_list
def main():
    st.title("🎵 Music Recommendation System")
    
    #       
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("recommendation based on song")
        # load data and model
        model, song_df, le_dict = load_model_and_data()
        
        if model is None or song_df is None or le_dict is None:
            st.error("system initialization failed. Please ensure all required files exist.")
            st.stop()
        
        # create song selector
        unique_songs = song_df[['title', 'artist_name']].drop_duplicates()
        song_options = [f"{row['title']} - {row['artist_name']}" 
                       for _, row in unique_songs.iterrows()]
        
        selected_songs = st.multiselect(
            "select the songs you have listened to:",
            options=song_options,
            max_selections=10
        )
        
        if st.button("get recommendation based on song"):
            if selected_songs:
                with st.spinner("generating recommendation..."):
                    # get song_ids of selected songs
                    listened_songs = []
                    for song in selected_songs:
                        title = song.split(" - ")[0]
                        artist = song.split(" - ")[1]
                        song_id = song_df[(song_df['title'] == title) & 
                                        (song_df['artist_name'] == artist)]['song'].iloc[0]
                        listened_songs.append(song_id)
                    
                                # use model to generate recommendation
                    recommended_songs, scores = get_user_recommendations(
                        model, 
                        user_id=0,  # new user default ID
                        song_df=song_df,
                        listened_songs=listened_songs,
                        top_k=5  # recommend 5 songs
                    )
                    
                    st.success("recommendation generated successfully!")
                    
                    # display recommendation results
                    st.subheader("recommendation songs:")
                    for song_id, score in zip(recommended_songs, scores):
                        song_info = song_df[song_df['song'] == song_id].iloc[0]
                        with st.expander(f"🎵 {song_info['title']} - {song_info['artist_name']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"artist: {song_info['artist_name']}")
                                st.write(f"album: {song_info['release']}")
                            with col2:
                                st.write(f"year: {song_info['year']}")
                                st.write(f"match score: {score:.2f}")
            else:
                st.warning("please select at least one song!")
    
    with col2:
        st.subheader("recommendation based on user")
        # get all unique users
        unique_users = song_df['user'].unique()
        
        # create user selector
        selected_user = st.selectbox(
            " select a user:",
            options=unique_users,
            format_func=lambda x: f"user {x[:8]}..."  # only display the first 8 characters of user ID
        )
        
        if st.button("get recommendation based on user"):
            if selected_user:
                with st.spinner("generating recommendation..."):
                    # get songs listened by the user
                    user_songs = song_df[song_df['user'] == selected_user]['song'].unique()
                    
                    # use model to generate recommendation
                    recommended_songs, scores = get_user_recommendations(
                        model,
                        user_id=0,  # here we can use the actual user ID mapping
                        song_df=song_df,
                        listened_songs=user_songs.tolist(),
                        top_k=5
                    )
                    
                    st.success("recommendation generated successfully!")
                    
                    st.write(f"the user has listened to {len(user_songs)} songs")
                    

                    st.subheader("recommendation songs:")
                    for song_id, score in zip(recommended_songs, scores):
                        song_info = song_df[song_df['song'] == song_id].iloc[0]
                        with st.expander(f"🎵 {song_info['title']} - {song_info['artist_name']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"artist: {song_info['artist_name']}")
                                st.write(f"album: {song_info['release']}")
                            with col2:
                                st.write(f"year: {song_info['year']}")
                                st.write(f"match score: {score:.2f}")
            else:
                st.warning("please select a user!")

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>music recommendation system | based on Wide & Deep model</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()