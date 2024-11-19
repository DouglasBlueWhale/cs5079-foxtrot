import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class WideAndDeepModel(nn.Module):
    def __init__(self, num_users, num_songs, num_artists, num_releases, num_years,
                 embedding_dim=64, hidden_layers=[256, 128, 64]):
        super().__init__()
        
        self.wide = nn.Linear(5, 1)
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.song_embedding = nn.Embedding(num_songs, embedding_dim)
        self.artist_embedding = nn.Embedding(num_artists, embedding_dim)
        self.release_embedding = nn.Embedding(num_releases, embedding_dim)
        self.year_embedding = nn.Embedding(num_years, embedding_dim)
    
        deep_input_dim = embedding_dim * 5 + 1  
        self.deep_layers = nn.ModuleList()
        prev_dim = deep_input_dim
        
        for hidden_dim in hidden_layers:
            self.deep_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.deep_layers.append(nn.ReLU())
            self.deep_layers.append(nn.BatchNorm1d(hidden_dim))
            self.deep_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        self.final = nn.Linear(hidden_layers[-1] + 1, 1)
        
    def forward(self, wide_features, user_ids, song_ids, artist_ids, 
                release_ids, year_ids, play_count):
        wide_out = self.wide(wide_features.float())
        
        user_emb = self.user_embedding(user_ids)
        song_emb = self.song_embedding(song_ids)
        artist_emb = self.artist_embedding(artist_ids)
        release_emb = self.release_embedding(release_ids)
        year_emb = self.year_embedding(year_ids)
        
        deep_input = torch.cat([
            user_emb, song_emb, artist_emb, release_emb, year_emb,
            play_count.unsqueeze(1)
        ], dim=1)
        
        deep_out = deep_input
        for layer in self.deep_layers:
            deep_out = layer(deep_out)
        
        final_input = torch.cat([deep_out, wide_out], dim=1)
        output = self.final(final_input)
        return output

class MusicDataset(Dataset):
    def __init__(self, df, is_training=True):
        self.is_training = is_training
        
        self.user_ids = torch.tensor(df['user_idx'].values, dtype=torch.long)
        self.song_ids = torch.tensor(df['song_idx'].values, dtype=torch.long)
        self.artist_ids = torch.tensor(df['artist_idx'].values, dtype=torch.long)
        self.release_ids = torch.tensor(df['release_idx'].values, dtype=torch.long)
        self.year_ids = torch.tensor(df['year_idx'].values, dtype=torch.long)
        
        self.wide_features = torch.tensor(
            df[['user_idx', 'song_idx', 'artist_idx', 'release_idx', 'year_idx']].values,
            dtype=torch.float
        )
        
        self.play_count = torch.tensor(df['play_count_log'].values, dtype=torch.float)
        
        if is_training:
            self.labels = torch.tensor(df['play_count_log'].values, dtype=torch.float)
            
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        if self.is_training:
            return {
                'wide_features': self.wide_features[idx],
                'user_ids': self.user_ids[idx],
                'song_ids': self.song_ids[idx],
                'artist_ids': self.artist_ids[idx],
                'release_ids': self.release_ids[idx],
                'year_ids': self.year_ids[idx],
                'play_count': self.play_count[idx],
                'labels': self.labels[idx]
            }
        else:
            return {
                'wide_features': self.wide_features[idx],
                'user_ids': self.user_ids[idx],
                'song_ids': self.song_ids[idx],
                'artist_ids': self.artist_ids[idx],
                'release_ids': self.release_ids[idx],
                'year_ids': self.year_ids[idx],
                'play_count': self.play_count[idx]
            }

def preprocess_data(song_df):
    le_user = LabelEncoder()
    le_song = LabelEncoder()
    le_artist = LabelEncoder()
    le_release = LabelEncoder()
    le_year = LabelEncoder()
    
    song_df['user_idx'] = le_user.fit_transform(song_df['user'])
    song_df['song_idx'] = le_song.fit_transform(song_df['song'])
    song_df['artist_idx'] = le_artist.fit_transform(song_df['artist_name'])
    song_df['release_idx'] = le_release.fit_transform(song_df['release'])
    song_df['year_idx'] = le_year.fit_transform(song_df['year'])
    
    song_df['play_count_log'] = np.log1p(song_df['play_count'])  # log1p = log(1+x)
    
    return song_df, le_user, le_song, le_artist, le_release, le_year

def train_model(model, train_loader, val_loader, optimizer, 
                patience=5, epochs=100, min_delta=1e-4):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            outputs = model(
                batch['wide_features'].to(device),
                batch['user_ids'].to(device),
                batch['song_ids'].to(device),
                batch['artist_ids'].to(device),
                batch['release_ids'].to(device),
                batch['year_ids'].to(device),
                batch['play_count'].to(device)
            )
            
            loss = F.mse_loss(outputs.squeeze(), batch['labels'].to(device))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    batch['wide_features'].to(device),
                    batch['user_ids'].to(device),
                    batch['song_ids'].to(device),
                    batch['artist_ids'].to(device),
                    batch['release_ids'].to(device),
                    batch['year_ids'].to(device),
                    batch['play_count'].to(device)
                )
                loss = F.mse_loss(outputs.squeeze(), batch['labels'].to(device))
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            model.load_state_dict(best_model_state)
            break
    
    return model


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
        user_ids = torch.full((num_songs,), user_id, dtype=torch.long, device=device)
        song_ids = torch.tensor(unique_songs['song_idx'].values, dtype=torch.long, device=device)
        artist_ids = torch.tensor(unique_songs['artist_idx'].values, dtype=torch.long, device=device)
        release_ids = torch.tensor(unique_songs['release_idx'].values, dtype=torch.long, device=device)
        year_ids = torch.tensor(unique_songs['year_idx'].values, dtype=torch.long, device=device)
        
        # Batch processing
        batch_size = 1024
        all_scores = []
        
        for i in range(0, num_songs, batch_size):
            batch_end = min(i + batch_size, num_songs)
            batch_slice = slice(i, batch_end)
            
            # Wide features
            wide_features = torch.cat([
                user_ids[batch_slice].float().unsqueeze(1),
                song_ids[batch_slice].float().unsqueeze(1),
                artist_ids[batch_slice].float().unsqueeze(1),
                release_ids[batch_slice].float().unsqueeze(1),
                year_ids[batch_slice].float().unsqueeze(1)
            ], dim=1)
            
            play_count = torch.zeros(batch_end - i, device=device)
            
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
        
        # Get original song IDs
        recommended_songs = unique_songs.iloc[top_k_indices.cpu().tolist()]['song'].values.tolist()
        scores_list = top_k_scores.cpu().tolist()
        
        return recommended_songs, scores_list