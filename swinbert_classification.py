import os
import logging
from moviepy.video.io.VideoFileClip import VideoFileClip
import pandas as pd
try:
    from moviepy.video.io.VideoFileClip import VideoFileClip

    # Check if subclip is accessible
    if hasattr(VideoFileClip, 'subclip'):
        print("subclip is successfully imported and accessible.")
    else:
        print("subclip is not accessible in VideoFileClip.")

except ModuleNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
import cv2
import csv
import json
import random
import multiprocessing
import torch
import av
from datasets import load_dataset, load_from_disk
import numpy as np
import evaluate
from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from transformers import AutoProcessor
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import torch
import datetime
import torchvision
from av import error as av_error
import evaluate
from transformers import TrainingArguments, Trainer
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from transformers import AutoTokenizer, VisionEncoderDecoderModel
from torchvision.transforms import Compose, Lambda, Normalize, Resize
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, VisionEncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    VisionEncoderDecoderModel,
    AutoImageProcessor,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    Seq2SeqTrainingArguments,
    TrainingArguments,
)
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms import Compose, Lambda, Normalize, Resize
from torch.nn.utils.rnn import pad_sequence
import torchvision
import os
import torch
from transformers import AutoImageProcessor
from transformers import VisionEncoderDecoderModel, AutoTokenizer
from PIL import Image  # Add this import
import os
import torch
from transformers import (
    VisionEncoderDecoderModel, 
    AutoTokenizer, 
    AutoImageProcessor, 
    VisionEncoderDecoderConfig
)
from PIL import Image
from torchvision import transforms
import cv2
from datasets import load_dataset
import os
import torch
from transformers import (
    VisionEncoderDecoderModel, 
    AutoTokenizer, 
    AutoImageProcessor, 
    VisionEncoderDecoderConfig
)
from PIL import Image
from torchvision import transforms
import cv2
from datasets import load_dataset
from transformers import PreTrainedModel
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import Compose, Normalize, Resize, Lambda
from torchvision.io import read_video
from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoImageProcessor,
    PreTrainedModel,
    TrainerCallback
)
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder


# input_dir = "/app/input"
# output_dir = "/app/output"
# video_folder = '/app/input/extracted_gray_videos'
# output_folder = '/app/input/outputliftfolder'
# os.makedirs(output_folder, exist_ok=True)

# # Setup logging
# log_file = "/app/input/check.log"  # Specify your log file path
# logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.info("Script started.")


# # Path to the CSV file
# csv_file = "/app/input/attributes.csv"

# # Check input and output directories and CSV file
# logging.info(f"Input directory exists: {os.path.exists(input_dir)}")
# logging.info(f"Output directory exists: {os.path.exists(output_dir)}")
# logging.info(f"CSV file exists: {os.path.exists(csv_file)}")

# # Helper function to convert time to seconds
# def convert_to_seconds(time_str):
#     try:
#         parts = time_str.split(':')
#         if len(parts) == 1:  # Format: seconds.milliseconds
#             return float(parts[0])
#         elif len(parts) == 2:  # Format: seconds:milliseconds
#             seconds, milliseconds = parts
#             return float(seconds) + float(milliseconds) / 100
#         elif len(parts) == 3:  # Format: minutes:seconds:milliseconds
#             minutes, seconds, milliseconds = parts
#             return int(minutes) * 60 + float(seconds) + float(milliseconds) / 100
#         else:
#             raise ValueError("Invalid time format")
#     except ValueError as e:
#         logging.error(f"Error converting time {time_str}: {e}")
#         return None

# # Read the CSV file
# if not os.path.exists(csv_file):
#     logging.error("CSV file not found. Exiting script.")
# else:
#     df = pd.read_csv(csv_file)

#     # Iterate over each row in the CSV
#     for index, row in df.iterrows():
#         video_source = row['video source']
        
#         # Convert time start and time end to seconds, adding an extra second
#         start_time = convert_to_seconds(str(row['time start']))
#         end_time = convert_to_seconds(str(row['time end']))

#         if start_time is None or end_time is None:
#             logging.warning(f"Skipping video {video_source} due to invalid time format.")
#             continue

#         # Adjust times to include an extra second before and after
#         start_time = max(0, start_time - 1)  # Ensure start_time is not negative

#         stroke_number = row['stroke number']
#         feedback = row['feedback']

#         # Construct the full path to the video file
#         video_path = os.path.join(video_folder, video_source)

#         # Construct the output filename
#         output_filename = f"{os.path.splitext(video_source)[0]}_lift_{stroke_number}.mp4"
#         output_path = os.path.join(output_folder, output_filename)

#         # Check if the output file already exists
#         if os.path.exists(output_path):
#             logging.info(f"Clipped video {output_filename} already exists, skipping.")
#             continue

#         # Check if the video file exists
#         if not os.path.exists(video_path):
#             logging.warning(f"Video file {video_source} not found, skipping.")
#             continue

#         try:
#             with VideoFileClip(video_path) as video:
#                 # Get the duration of the video
#                 video_duration = video.duration
                
#                 # Check if start_time and end_time are within the video duration
#                 if start_time < 0 or end_time > video_duration:
#                     logging.warning(f"Start time {start_time} or end time {end_time} is out of bounds for video {video_source}. Skipping.")
#                     continue
                
#                 # Clip the video based on start and end times
#                 clipped = video.subclip(start_time, end_time)
                
#                 # Write the clipped video to the output file
#                 clipped.write_videofile(output_path, codec='libx264')
#                 logging.info(f"Clipped video {output_filename} created successfully.")

#         except Exception as e:
#             logging.error(f"Error processing video {video_source}: {e}")


# def resize_videos(input_folder, output_folder, target_size):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     for filename in os.listdir(input_folder):
#         if filename.endswith(".mp4"): 
#             input_path = os.path.join(input_folder, filename)
#             output_path = os.path.join(output_folder, filename)
            
#             # Check if the resized video already exists
#             if os.path.exists(output_path):
#                 print(f"Resized video already exists: {output_path}, skipping.")
#                 continue
            
#             # Open the video file
#             cap = cv2.VideoCapture(input_path)
#             if not cap.isOpened():
#                 print(f"Error opening video file: {input_path}")
#                 continue
            
#             # Get video properties
#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             min_edge = min(width, height)
#             aspect_ratio = width / height
            
#             # Calculate new dimensions
#             if min_edge > target_size:
#                 new_width = int(target_size * aspect_ratio)
#                 new_height = target_size if width < height else int(target_size / aspect_ratio)
#             else:
#                 new_width, new_height = width, height
           
#             # Define video writer with appropriate settings
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (new_width, new_height))

#             # Read and resize frames
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 resized_frame = cv2.resize(frame, (new_width, new_height))
#                 out.write(resized_frame)

#             # Release resources
#             cap.release()
#             out.release()
#             print(f"Resized video saved: {output_path}")

# # Parameters
# input_folder = "/app/input/outputliftfolder"
# output_folder = "/app/input/resized_videos"
# target_size = 256  # Short edge size
    
# # Run the function
# resize_videos(input_folder, output_folder, target_size)

# csv_file_path = '/app/input/attributes.csv'
# video_folder_path = '/app/input/outputliftfolder'
# json_file_path = '/app/input/output.json'
# output_dir = "/app/input/data1/caelen/dataset"
# videos_path = "/app/input/resized_videos"

# # Columns to exclude for CSV output only
# exclude_columns_csv = {'video source','time start','time end', 'stroke number'}

# # Read and filter CSV data
# data = []
# with open(csv_file_path, mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     for row in csv_reader:
#         data.append(row)  # Keep all columns for JSON

# # Process video files
# video_files = [f for f in os.listdir(video_folder_path) if f.endswith('.mp4')]
# results = []

# for video_file in video_files:
#     # Extract video ID and stroke number
#     parts = video_file.split('_lift_')
#     video_source = parts[0] + '.mp4'
#     stroke_number = parts[1].split('.')[0]

#     # Find matching feedback from CSV
#     for row in data:
#         if row.get('video source') == video_source:
#             video_id = video_file.replace('.mp4', '')
#             feedback = row.get('feedback', '')  # Default to empty string if 'feedback' is missing
            
#             # Prepare the result row for JSON
#             result_row = {"videoID": video_id, **row}  # Include all fields from the row
#             results.append(result_row)
#             break

# # Save JSON output with all fields
# os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
# with open(json_file_path, 'w') as json_file:
#     json.dump(results, json_file, indent=2)

# print("JSON file created successfully.")

# # Functions to load and save JSON
# def load_json(file_path):
#     with open(file_path, 'r') as file:
#         return json.load(file)

# def save_json(data, file_path):
#     with open(file_path, 'w') as file:
#         json.dump(data, file, indent=4)

# # Split data into train, validation, and test sets
# def split_data(data, train_ratio=0.8, val_ratio=0.1):
#     random.shuffle(data)
#     total = len(data)
#     train_end = int(total * train_ratio)
#     val_end = train_end + int(total * val_ratio)
#     return data[:train_end], data[train_end:val_end], data[val_end:]

# # Load JSON and split datasets
# data = load_json(json_file_path)
# train_data, val_data, test_data = split_data(data, train_ratio=0.8, val_ratio=0.1)

# # Save split datasets
# save_json(train_data, "train.json")
# save_json(val_data, "val.json")
# save_json(test_data, "test.json")

# print(f"Train set size: {len(train_data)}")
# print(f"Validation set size: {len(val_data)}")
# print(f"Test set size: {len(test_data)}")
# print("Files saved: train.json, val.json, test.json")

# # Create CSV files for splits
# os.makedirs(output_dir, exist_ok=True)
# csv_files = {
#     "train": os.path.join(output_dir, "train.csv"),
#     "validation": os.path.join(output_dir, "val.csv"),
#     "test": os.path.join(output_dir, "test.csv")
# }

# # Write split datasets to CSV
# for split, dataset in zip(["train", "validation", "test"], [train_data, val_data, test_data]):
#     print(f"Processing {split} dataset...")
#     with open(csv_files[split], mode='w', newline='') as file:
#         # Get fieldnames excluding fields for CSV
#         fieldnames = [
#             key if key != "videoID" else "video_path" 
#             for key in dataset[0].keys() 
#             if key not in exclude_columns_csv
#         ]
#         csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
#         csv_writer.writeheader()
        
#         for record in dataset:
#             # Replace videoID with video_path
#             record["video_path"] = os.path.join(videos_path, f"{record['videoID']}.mp4")
#             record.pop("videoID", None)  # Remove videoID field for CSV
#             # Remove excluded fields for CSV
#             for column in exclude_columns_csv:
#                 record.pop(column, None)
#             csv_writer.writerow(record)

# print("CSV files created for train, validation, and test splits.")


# train_csv = "/app/input/data1/caelen/dataset/train.csv"
# column_names = ["video_path", "squat_position", "racquet_carriage", "leg_stance", "lunge_length","non_racquet_hand","knee_alignment","time_of_lift","racquet_hand","movement","overall lift"]

# # Load dataset
# train_dataset = load_dataset(
#     'csv',
#     data_files={'sample': train_csv},
#     delimiter=',',
#     column_names=column_names,
#     skiprows=1
# )
# val_dataset = train_dataset
# # Debugging
# print("Sample data from the dataset:")
# print(train_dataset["sample"][:5])

# # Load models
# encoder = "facebook/timesformer-base-finetuned-k600"
# decoder = "gpt2"
# image_processor = AutoImageProcessor.from_pretrained(encoder)
# tokenizer = AutoTokenizer.from_pretrained(decoder)
# tokenizer.pad_token = tokenizer.eos_token

# # Load VisionEncoderDecoderModel
# base_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder, decoder).to('cuda')


# # Dynamically extract class columns and their unique values
# class_columns = {col: list(set(train_dataset["sample"][col])) for col in column_names if col != "video_path"}

# # Initialize label encoders for each class column
# label_encoders = {}
# for col, classes in class_columns.items():
#     encoder = LabelEncoder()
#     encoder.fit(classes)
#     label_encoders[col] = encoder

# # Print column names and their unique classes
# print("Extracted column names and their unique classes:")
# for col, classes in class_columns.items():
#     print(f"{col}: {classes}")

# class ModifiedModel(PreTrainedModel):
#     def __init__(self, model, class_columns):
#         super(ModifiedModel, self).__init__(model.config)
#         self.encoder = model.encoder
#         self.decoder = model.decoder
        
#         hidden_size = model.encoder.config.hidden_size
#         self.classifiers = nn.ModuleDict({
#             col: nn.Linear(hidden_size, len(set(class_columns[col])))
#             for col in class_columns
#         })
        
#         # Store class mappings for easier reference
#         self.class_mappings = class_columns

#     def forward(self, pixel_values, labels=None, combined_labels=None, video_paths=None, **kwargs):
#         # Forward pass through the encoder
#         encoder_outputs = self.encoder(pixel_values)
#         hidden_states = encoder_outputs.last_hidden_state
        
#         logits_dict = {}
#         probs_dict = {}
#         pred_dict = {}
#         mapped_preds = {}  # To store predictions mapped back to class labels

#         for col, classifier in self.classifiers.items():
#             # Classifier predictions
#             logits = classifier(hidden_states[:, -1, :])
#             probs = F.softmax(logits, dim=-1)
#             pred = torch.argmax(probs, dim=-1)
            
#             # Map predictions back to class labels
#             class_labels = self.class_mappings[col]
#             mapped_pred = [class_labels[idx] for idx in pred.cpu().tolist()]

#             logits_dict[f"{col}_logits"] = logits
#             probs_dict[f"{col}_probs"] = probs
#             pred_dict[f"{col}_pred"] = pred
#             mapped_preds[f"{col}_mapped_pred"] = mapped_pred

#         generated_text = None
#         loss = None
#         if labels is not None:
#             # Forward pass through the decoder
#             decoder_outputs = self.decoder(
#                 input_ids=labels,
#                 encoder_hidden_states=hidden_states,
#                 attention_mask=kwargs.get("attention_mask", None),
#             )
#             logits = decoder_outputs.logits

#             loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
#             loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
#         else:
#             logits = None

#         return {
#             **logits_dict,
#             **probs_dict,
#             **pred_dict,
#             **mapped_preds,  # Include mapped predictions in the output
#             "generated_text": generated_text,
#             "logits": logits,
#             "loss": loss,
#         }


# # Initialize the modified model
# model = ModifiedModel(base_model, class_columns).to('cuda')

# class SaveBestCheckpointCallback(TrainerCallback):
#     def __init__(self, output_dir, class_columns):
#         super().__init__()
#         self.output_dir = output_dir
#         self.class_columns = class_columns

#     def on_train_end(self, args, state, control, **kwargs):
#         if state.best_model_checkpoint:
#             print(f"Saving best model from {state.best_model_checkpoint}...")
#             model = kwargs["model"]
#             tokenizer = kwargs["tokenizer"]

#             # Save the encoder and decoder
#             model.encoder.save_pretrained(self.output_dir)
#             model.decoder.save_pretrained(self.output_dir)
            
#             # Save classifiers
#             for col in self.class_columns:
#                 classifier_name = f"{col}_classifier.pth"
#                 classifier = model.classifiers[col]
#                 torch.save(classifier.state_dict(), os.path.join(self.output_dir, classifier_name))
            
#             # Save tokenizer and config
#             tokenizer.save_pretrained(self.output_dir)
#             model.config.save_pretrained(self.output_dir)
#             print(f"Model and classification heads saved to {self.output_dir}")

# # Data preprocessing setup
# mean = image_processor.image_mean
# std = image_processor.image_std
# height, width = image_processor.size.get("height", 224), image_processor.size.get("width", 224)
# resize_to = (height, width)

# transform = Compose([
#     Lambda(lambda x: x / 255.0),
#     Normalize(mean, std),
#     Resize(resize_to),
# ])


# def collate_fn(examples):
#     batch_video_frames = []
#     labels = []
#     combined_labels = []
#     video_paths = []  # Initialize an empty list for video paths

#     for example in examples:
#         video_path = example["video_path"]
#         try:
#             video_paths.append(video_path)  # Store the video path
#             if not os.path.exists(video_path):
#                 raise FileNotFoundError(f"Video file not found: {video_path}")

#             sample = read_video(video_path)

#             if len(sample) == 2:
#                 video_frames, _ = sample
#             elif len(sample) == 3:
#                 video_frames, _, _ = sample
#             else:
#                 raise ValueError(f"Unexpected elements from read_video for {video_path}")

#             if video_frames.shape[0] < 16:
#                 print(f"Skipping video {video_path} due to insufficient frames.")
#                 continue

#             video_frames = video_frames[:16]
#             video_frames = video_frames.permute(3, 0, 1, 2)

#             frames_list = [transform(video_frames[:, i, :, :]).unsqueeze(0) for i in range(video_frames.shape[1])]
#             video_frames = torch.cat(frames_list).unsqueeze(0)

#             batch_video_frames.append(video_frames)

#             encoded_labels = [label_encoders[col].transform([example[col]])[0] for col in class_columns]
#             combined_labels.append(encoded_labels)

#             prompt_text = ", ".join([f"{col}: {label}" for col, label in zip(class_columns.keys(), encoded_labels)])
#             label = tokenizer(prompt_text, truncation=True, padding="max_length", max_length=50, return_tensors="pt")
#             labels.append(label.input_ids.squeeze(0))
#         except Exception as e:
#             print(f"Error processing video {video_path}: {e}")
#             continue

#     if not batch_video_frames:
#         print("No valid video frames found. Skipping...")
#         return {}

#     pixel_values = torch.stack(batch_video_frames).squeeze(1)
#     labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id) if labels else torch.empty(0)
#     combined_labels = torch.tensor(combined_labels) if combined_labels else torch.empty(0)

#     return {
#         "pixel_values": pixel_values,
#         "labels": labels,
#         "combined_labels": combined_labels,
#     }
# # Training arguments
# output_dir = "/app/output"
# training_args = Seq2SeqTrainingArguments(
#     output_dir=output_dir,
#     tf32=False,
#     predict_with_generate=True,
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     save_total_limit=1,
#     logging_strategy="epoch",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     remove_unused_columns=False,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     dataloader_num_workers=1,
#     num_train_epochs=50,
#     learning_rate=5e-5,
#     weight_decay=0.01,
#     report_to="wandb",
# )

# # Trainer setup
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset["sample"],
#     eval_dataset=val_dataset["sample"],
#     data_collator=collate_fn,
#     tokenizer=tokenizer,
#     callbacks=[SaveBestCheckpointCallback(output_dir=output_dir, class_columns=class_columns)],  # Pass class_columns here
# )

# # Train the model
# train_results = trainer.train()




##inference######
import torch
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor, GPT2LMHeadModel, GPT2Tokenizer
import os
import cv2
from PIL import Image
from torchvision import transforms
from datasets import load_dataset

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load Vision Encoder-Decoder model
model_path = "/app/output"  # Update this path as per your setup
loaded_model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)

# Load tokenizer and image processor
loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

# Load the GPT-2 model and tokenizer
gpt2_model_name = "gpt2"
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

# Dataset configuration
train_csv = "/app/input/data1/caelen/dataset/train.csv"
column_names = ["video_path", "squat_position", "racquet_carriage", "leg_stance", "lunge_length","non_racquet_hand","knee_alignment","time_of_lift","racquet_hand","movement","overall lift"]

# Load dataset
train_dataset = load_dataset(
    'csv',
    data_files={'sample': train_csv},
    delimiter=',' ,
    column_names=column_names,
    skiprows=1
)

# Dynamically extract class columns and their unique values
class_columns = {col: list(set(train_dataset["sample"][col])) for col in column_names if col != "video_path"}

# Print column names and their unique classes
print("Extracted column names and their unique classes:")
for col, classes in class_columns.items():
    print(f"{col}: {classes}")

# Define the hidden size from the encoder
hidden_size = loaded_model.encoder.config.hidden_size

# Dynamically create classifiers and load weights
classifiers = {}

for col, classes in class_columns.items():
    num_classes = len(classes)  # Get the number of unique classes
    classifier = torch.nn.Linear(hidden_size, num_classes)  # Define the classifier
    weight_path = os.path.join(model_path, f"{col}_classifier.pth")  # Define weight path

    # Load weights if they exist
    if os.path.exists(weight_path):
        classifier.load_state_dict(torch.load(weight_path))
    else:
        print(f"Warning: No saved weights found for {col} classifier at {weight_path}")
    
    # Move the classifier to the device
    classifiers[col] = classifier.to(device)

# Print the created classifiers
print("\nCreated classifiers dynamically:")
for col, classifier in classifiers.items():
    print(f"{col}: {classifier}")

# Assuming `loaded_model` and `classifiers` are already defined
loaded_model.eval()

# Define preprocess transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size as per model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Frame extraction function
def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // num_frames)

    print(f"Processing video: {video_path}")
    print(f"Total frames in video: {frame_count}, Sampling every {step} frames.")

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print(f"Frame read failed at index {i}.")
            break
        if i % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = preprocess(frame)
            frames.append(frame)

    cap.release()

    # Handle padding or truncation to ensure consistent frame count
    if len(frames) < num_frames:
        print(f"Padding frames: Current count is {len(frames)}, required is {num_frames}.")
        while len(frames) < num_frames:
            frames.append(frames[-1])
    elif len(frames) > num_frames:
        print(f"Truncating frames: Current count is {len(frames)}, required is {num_frames}.")
        frames = frames[:num_frames]

    frames_tensor = torch.stack(frames)  # [num_frames, 3, 224, 224]
    print(f"Final tensor shape after stacking: {frames_tensor.shape}")
    return frames_tensor


# Process each video and update predictions
def process_video(video_path, num_frames=16):
    try:
        # Extract and preprocess frames
        video_frames = extract_frames(video_path, num_frames).to(device)
        print(f"Frames shape after preprocessing: {video_frames.shape}")

        # Add batch dimension
        video_frames = video_frames.unsqueeze(0)  # [1, num_frames, 3, height, width]
        print(f"Input shape after unsqueeze: {video_frames.shape}")

        # Pass frames through encoder
        encoder_outputs = loaded_model.encoder(pixel_values=video_frames)
        print(f"Encoder output shape: {encoder_outputs.last_hidden_state.shape}")

        # Pool encoder outputs
        pooled_output = encoder_outputs.last_hidden_state.mean(dim=1)
        print(f"Pooled output shape: {pooled_output.shape}")

        # Classify outputs
        predictions = {}
        for col, classifier in classifiers.items():
            logits = classifier(pooled_output)
            pred_idx = torch.argmax(logits, dim=-1).item()
            pred_label = class_columns[col][pred_idx]
            predictions[col] = pred_label
            print(f"{col} Prediction: {pred_label}")

        return predictions
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return {col: "Error" for col in classifiers.keys()}

# Process CSV and update predictions
def process_videos_from_csv(csv_path, output_csv_path, num_frames=16):
    df = pd.read_csv(csv_path)
    predictions = {col: [] for col in classifiers.keys()}

    for idx, row in df.iterrows():
        video_path = row['video_path']
        print(f"Processing video {idx+1}/{len(df)}: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            for col in classifiers.keys():
                predictions[col].append("Not Found")
            continue

        # Process video and get predictions
        video_predictions = process_video(video_path, num_frames=num_frames)
        for col, pred in video_predictions.items():
            predictions[col].append(pred)

    # Add predictions and ground truth to DataFrame
    for col in classifiers.keys():
        df[f'ground_truth_{col}'] = df[col]  # Assuming ground truth is already in the CSV
        df[f'pred_{col}'] = predictions[col]

    # Save updated CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved at {output_csv_path}")

# Example usage
csv_path = "/app/input/data1/caelen/dataset/val.csv"
output_csv_path = "/app/output/val_pred.csv"
process_videos_from_csv(csv_path, output_csv_path, num_frames=16)
