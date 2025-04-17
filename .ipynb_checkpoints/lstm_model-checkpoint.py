import tensorflow as tf
tf.config.optimizer.set_jit(True)
from tensorflow.keras import mixed_precision  # NEW (TF 2.17)
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Conv3D, TimeDistributed, Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
import pandas as pd
from math import sin, cos, radians
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error





def overlay_hevc_tiles(frame, video_width, video_height, N_x, N_y):

    tile_width = video_width // N_x
    tile_height = video_height // N_y

    # Draw horizontal lines
    for i in range(1, N_y):
        y = i * tile_height
        cv2.line(frame, (0, y), (video_width, y), (255, 0, 0), 2)  # Blue grid lines

    # Draw vertical lines
    for j in range(1, N_x):
        x = j * tile_width
        cv2.line(frame, (x, 0), (x, video_height), (255, 0, 0), 2)  # Blue grid lines

    # Add tile indices
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(N_y):
        for j in range(N_x):
            x = j * tile_width + 10
            y = i * tile_height + 30
            cv2.putText(frame, f"Tile {i},{j}", (x, y), font, 0.5, (0, 0, 255), 1)  # Red text

    return frame


def calculate_fov_trig(head_pitch, head_yaw, eye_direction, video_width, video_height, fov_width=90, fov_height=90):
    # Convert head_pitch and head_yaw from degrees to radians
    pitch_rad = radians(head_pitch)
    yaw_rad = radians(head_yaw)
    
    radius = 1.0

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * cos(yaw_rad) * sin(pitch_rad)  # x-axis component
    y = radius * sin(yaw_rad) * sin(pitch_rad)  # y-axis component
    z = radius * cos(pitch_rad)                # z-axis component)               

    # Adjust for gaze direction by adding eye_direction (normalized vector)
    eye_x, eye_y, eye_z = eye_direction
    gaze_x = x + eye_x
    gaze_y = y + eye_y
    gaze_z = z + eye_z

    # Normalize the gaze vector to prevent scaling issues
    # norm = np.sqrt(gaze_x**2 + gaze_y**2 + gaze_z**2)
    # gaze_x /= norm
    # gaze_y /= norm
    # gaze_z /= norm

    # Map normalized Cartesian coordinates to 2D video plane
    center_x = int(video_width / 2 + gaze_x * video_width / 2)
    center_y = int(video_height / 2 - gaze_y * video_height / 2) 

    # Map FOV size in degrees to pixels
    fov_width_px = int((fov_width / 360) * video_width)
    fov_height_px = int((fov_height / 180) * video_height)

    # Define FOV rectangle 
    top_left = (max(0, center_x - fov_width_px // 2), max(0, center_y - fov_height_px // 2))
    bottom_right = (min(video_width, center_x + fov_width_px // 2), min(video_height, center_y + fov_height_px // 2))

    return top_left, bottom_right


def process_video_fov(video_path, frame_number, predictions_df,  actual_df, video_width, video_height,N_x, N_y):
    

    pred_row = predictions_df[predictions_df['frame'] == frame_number]
    if pred_row.empty:
        raise ValueError(f"No predictions found for frame {frame_number}.")
    
    head_pitch_pred = pred_row['predicted_head_pitch'].values[0]
    head_yaw_pred = pred_row['predicted_head_yaw'].values[0]
    eye_direction_pred = (
        pred_row['predicted_eye_x_t'].values[0],
        pred_row['predicted_eye_y_t'].values[0],
        pred_row['predicted_eye_z_t'].values[0]
    )

    # Get actual data for the specified frame
    actual_row = actual_df[actual_df['frame'] == frame_number]
    if actual_row.empty:
        raise ValueError(f"No actual data found for frame {frame_number}.")
    
    head_pitch_actual = actual_row['actual_head_pitch'].values[0]
    head_yaw_actual = actual_row['actual_head_yaw'].values[0]
    eye_direction_actual = (
        actual_row['actual_eye_x_t'].values[0],
        actual_row['actual_eye_y_t'].values[0],
        actual_row['actual_eye_z_t'].values[0]
    )

    # Calculate the optimal FOV for predicted and actual data
    fov_coordinates_pred = calculate_fov_trig(
        head_pitch_pred, head_yaw_pred, eye_direction_pred, video_width, video_height
    )
    fov_coordinates_actual = calculate_fov_trig(
        head_pitch_actual, head_yaw_actual, eye_direction_actual, video_width, video_height
    )

    # Overlay FOVs on the frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Frame {frame_number} could not be read from video.")
    cap.release()


    #draw HEVC tiles
    frame_with_tiles = overlay_hevc_tiles(frame, video_width, video_height, N_x, N_y)

    top_left_pred, bottom_right_pred = fov_coordinates_pred
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left_pred, bottom_right_pred, (0, 255, 0), -1)  # Green for predicted

    top_left_actual, bottom_right_actual = fov_coordinates_actual
    cv2.rectangle(overlay, top_left_actual, bottom_right_actual, (0, 0, 255), -1)  # Red for actual
    
    alpha = 0.35
    frame_with_fov = cv2.addWeighted(overlay, alpha, frame, 1-alpha,0)

    # Add frame number to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame_with_fov, 
        f"Frame: {frame_number} data: 10 users", 
        (10, 30), 
        font, 
        1, 
        (0, 255, 0), 
        2
    )

    return fov_coordinates_pred, fov_coordinates_actual, frame_with_fov



def plot_tile_overlap(df_results, save_path="tile_overlap_plot.png"):

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=df_results["Frame"], y=df_results["Tile Overlap Ratio (%)"], marker="o", color="b", label="Tile Overlap (%)")
    
    plt.xlabel("Frame Number")
    plt.ylabel("Tile Overlap Ratio (%)")
    plt.title("Tile Overlap Ratio Over Frames")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(save_path)
    plt.show()

def plot_tile_overlap_distribution(df_results, save_path="tile_overlap_histogram.png"):

    plt.figure(figsize=(8, 5))
    sns.histplot(df_results["Tile Overlap Ratio (%)"], bins=10, kde=True, color="green")
    
    plt.xlabel("Tile Overlap Ratio (%)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Tile Overlap Ratios")
    plt.grid(True)
    
    # Save the histogram
    plt.savefig(save_path)
    plt.show()

def calculate_tile_overlap(fov_pred, fov_actual, video_width, video_height, N_x, N_y):

    tile_width = video_width // N_x
    tile_height = video_height // N_y

    def get_tiles_in_fov(fov_coords):
        """Returns a set of tile indices covered by the given FOV coordinates."""
        (x1, y1), (x2, y2) = fov_coords
        tile_x1, tile_y1 = x1 // tile_width, y1 // tile_height
        tile_x2, tile_y2 = x2 // tile_width, y2 // tile_height
        
        covered_tiles = set()
        for i in range(tile_y1, min(tile_y2 + 1, N_y)):
            for j in range(tile_x1, min(tile_x2 + 1, N_x)):
                covered_tiles.add((i, j))  # Store as (row, column)
        return covered_tiles

    # Get tile sets for predicted and actual FOV
    tiles_pred = get_tiles_in_fov(fov_pred)
    tiles_actual = get_tiles_in_fov(fov_actual)

    # Calculate tile overlap ratio
    intersection = tiles_pred & tiles_actual
    union = tiles_pred | tiles_actual

    overlap_ratio = len(intersection) / len(union) if len(union) > 0 else 0
    return overlap_ratio * 100  

def process_video_tile_overlap(video_path, frame_range, predictions_df, actual_df, video_width, video_height, N_x, N_y):
 
    results = []

    for frame_number in frame_range:
        try:
           
            fov_pred, fov_actual, _ = process_video_fov(
                video_path, frame_number, predictions_df, actual_df, video_width, video_height, N_x, N_y
            )

            # Calculate tile overlap ratio
            tile_overlap_ratio = calculate_tile_overlap(fov_pred, fov_actual, video_width, video_height, N_x, N_y)
            
            # Store the result
            results.append({"Frame": frame_number, "Tile Overlap Ratio (%)": tile_overlap_ratio})

        except ValueError as e:
            print(f"Skipping frame {frame_number}: {e}")


    df_results = pd.DataFrame(results)

    # Compute average overlap ratio
    avg_overlap_ratio = df_results["Tile Overlap Ratio (%)"].mean()
    
    return df_results, avg_overlap_ratio






policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)



# HEVC tile configuration
N_x = 6  
N_y = 5  


#for avg tile ratio
frame_range = list(range(3000,4000))


# Define users and videos
users_list = [15, 30, 45]
#users_list = [15]



videos = ['/Users/adi/Documents/University/YEAR 4/FYP thesis/videos/Surfing.mp4', '/Users/adi/Documents/University/YEAR 4/FYP thesis/videos/Alien.mp4', '/Users/adi/Documents/University/YEAR 4/FYP thesis/videos/OculusQuest.mp4']
#videos = ['/Users/adi/Documents/University/YEAR 4/FYP thesis/videos/Surfing.mp4']

video_ids = [13, 1, 8]
#video_ids = [13]

#
# Create directory for saving results
output_dir = '/Users/adi/Documents/University/YEAR 4/FYP thesis/fyp_code/LSTM_model_metrics'
os.makedirs(output_dir, exist_ok=True)

# Function to get video dimensions
def get_video_dimensions(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height



def load_filtered_data(base_dir, video_id, num_users):
    all_data = []
    video_dir = os.path.join(base_dir, f"video_{video_id}")
    
    for user_id in range(1, num_users + 1):
        file_path = os.path.join(video_dir, f"user{user_id}.csv")
        
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            df['video_ID'] = video_id
            df['user_ID'] = user_id
            
            df['player_time'] = df['player_time'].str.strip()
            try:
                df['player_time'] = pd.to_timedelta(df['player_time'])
                df['Elapsed_Time'] = (df['player_time'] - df['player_time'].iloc[0]).dt.total_seconds()
            except ValueError as e:
                print(f"Error parsing time in file {file_path}: {e}")
                continue
            
            all_data.append(df)
    
    consolidated_data =  pd.concat(all_data, ignore_index=True) 
    return consolidated_data


# Initialize results list
metrics_results = []
for video, video_id in zip(videos, video_ids):
    video_width, video_height = get_video_dimensions(video)
    
    for num_users in users_list:
        print(f"Running model for {num_users} users on {video}...")
        df_filtered_data = load_filtered_data('/Users/adi/Documents/University/YEAR 4/FYP thesis/Panonut360/dataset', video_id, num_users) 
        

        features = ['head_pitch', 'head_yaw', 'head_roll', 'eye_x_t', 'eye_y_t', 'eye_z_t']


        df = df_filtered_data[:]
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])

        df['frame'] = df_filtered_data['frame']

        grouped_by_user = df.groupby('user_ID')

        # Separate users into training and testing (80% train, 20% test)
        user_ids = df['user_ID'].unique()
        np.random.shuffle(user_ids)
        train_users, test_users = train_test_split(user_ids, test_size=0.2, random_state=50)

        # Create separate datasets for training and testing
        train_data = df[df['user_ID'].isin(train_users)]
        test_data = df[df['user_ID'].isin(test_users)]

        sequence_length = 6  
        n_frames_ahead = 12 

        def create_sequences_with_multiple_targets(df, features, frame_column, sequence_length, n_frames_ahead):
            X, y, frames = [], [], []
            for i in range(len(df) - sequence_length - n_frames_ahead + 1):
                X.append(df[features].iloc[i:i + sequence_length].values)
                # Flatten the next n_frames_ahead targets
                y.append(df[features].iloc[i + sequence_length:i + sequence_length + n_frames_ahead].values.flatten())
                frames.append(df[frame_column].iloc[i + sequence_length:i + sequence_length + n_frames_ahead].values)
            return np.array(X), np.array(y), np.array(frames)

    
        grouped_train = train_data.groupby('user_ID')

        X_train, y_train, frame_train = [], [], []
        for user_id, user_data in grouped_train:
            X_user, y_user, frames_user = create_sequences_with_multiple_targets(
                user_data, features, 'frame', sequence_length, n_frames_ahead
            )
            X_train.append(X_user)
            y_train.append(y_user)
            frame_train.append(frames_user)

        # Group test_data by user_ID
        grouped_test = test_data.groupby('user_ID')

        X_test, y_test, frame_test = [], [], []
        for user_id, user_data in grouped_test:
            X_user, y_user, frames_user = create_sequences_with_multiple_targets(
                user_data, features, 'frame', sequence_length, n_frames_ahead
            )
            X_test.append(X_user)
            y_test.append(y_user)
            frame_test.append(frames_user)

        # Combine data for all users
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        frame_train = np.concatenate(frame_train, axis=0)

        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        frame_test = np.concatenate(frame_test, axis=0)

        X_train = X_train.reshape((X_train.shape[0], sequence_length, 1, len(features), 1))
        X_test = X_test.reshape((X_test.shape[0], sequence_length, 1, len(features), 1))


        # Build the ConvLSTM model
        model = Sequential([
            ConvLSTM2D(filters=64, kernel_size=(1, 3), padding="same", return_sequences=True, 
                    input_shape=(sequence_length, 1, len(features), 1)),
            Dropout(0.5),
            ConvLSTM2D(filters=32, kernel_size=(1, 3), padding="same", return_sequences=False),
            Dropout(0.5),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(n_frames_ahead * len(features))  # Output predicts n_frames_ahead
        ])

        # Compile the model
        model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse')

        # Define checkpoint directory
        checkpoint_dir = "./checkpoints"
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt.weights.h5")

        # Create checkpoint object to save model & optimizer state
        checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model)

        # Restore latest checkpoint (if available)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"Restoring from checkpoint: {latest_checkpoint}")
            checkpoint.restore(latest_checkpoint)
            print("Checkpoint restored, resuming training.")

        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True, 
            save_freq='epoch', 
            verbose=1
        )

        # Train the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

        history = model.fit(
            X_train, y_train, epochs=20, batch_size=1, validation_split=0.1,
            callbacks=[early_stopping, reduce_lr,checkpoint_callback]
        )

        # Evaluate the model
        loss = model.evaluate(X_test, y_test)
        print("Test Loss:", loss)

        # Predict on the test set
        predictions = model.predict(X_test)

        
        predictions = predictions.reshape(-1, n_frames_ahead, len(features))
        y_test_reshaped = y_test.reshape(-1, n_frames_ahead, len(features))

        predictions = scaler.inverse_transform(predictions.reshape(-1, len(features)))
        y_test_original = scaler.inverse_transform(y_test_reshaped.reshape(-1, len(features)))

        preds = pd.DataFrame({
            'frame' : frame_test.flatten(),
            'predicted_head_pitch': predictions[:, 0],
            'predicted_head_yaw': predictions[:, 1],
            'predicted_head_roll': predictions[:, 2],
            'predicted_eye_x_t': predictions[:, 3],
            'predicted_eye_y_t': predictions[:, 4],
            'predicted_eye_z_t': predictions[:, 5],
        })
        sorted_df = preds.sort_values(by=['frame'], ascending=True)

        orig = pd.DataFrame({
            'frame': frame_test.flatten(),
            'actual_head_pitch': y_test_original[:,0],
            'actual_head_yaw': y_test_original[:,1],
            'actual_head_roll': y_test_original[:,2],
            'actual_eye_x_t': y_test_original[:,3],
            'actual_eye_y_t': y_test_original[:,4],
            'actual_eye_z_t': y_test_original[:,5]# Use the frame numbers from the test set
        })
        sorted_df2 = orig.sort_values(by=['frame'], ascending=True)



        predictions_df = sorted_df
        actual_df = sorted_df2


        df_overlap_results, avg_tile_overlap = process_video_tile_overlap( video, frame_range, predictions_df, actual_df, video_width, video_height, N_x, N_y)


        
        
        mse = mean_squared_error(y_test_original, predictions)
        r2 = r2_score(y_test_original,predictions)
        fit_accuracy = r2 * 100  # Placeholder calculation
        tile_overlap_ratio = avg_tile_overlap
        
        # Store metrics
        metrics_results.append([r2, fit_accuracy, mse, tile_overlap_ratio, num_users])
        
        # Save overlap ratio dataset
        overlap_filename = os.path.join(output_dir, f"overlapratio_{num_users}_{video_id}_unlabelled_5x6.xlsx")
        df_overlap_results.to_excel(overlap_filename, index=False)
        
        print(f"Completed {num_users} users for {video}.")

# Save all metrics to a final Excel file
metrics_df = pd.DataFrame(metrics_results, columns=["R2_Score", "Fit_Accuracy", "MSE", "Tile_Overlap_Ratio", "Num_Users"])
metrics_df.to_excel(os.path.join(output_dir, "metrics_final_unlabelled_5x6.xlsx"), index=False)

print("All model iterations completed.")
