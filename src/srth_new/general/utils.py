import json
from typing import List

from dataclasses import dataclass
import os
import numpy as np
from pathlib import Path
import torch

from srth_new.general import constants

import logging
log = logging.getLogger(__name__)

@dataclass
class DatasetStats:
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    dataset_dir: str
    tissue_sample_ids_train: List[int]


def get_sorted_phases(tissue_dir: Path) -> List[str]:
    phases = [file_name for file_name in os.listdir(tissue_dir)]
    phases_ordered = sorted(phases, key=lambda x: int(x.split('_')[0]))
    return phases_ordered

def initialize_model_and_tokenizer(encoder: str):
    from transformers import (
        DistilBertTokenizer,
        DistilBertModel,
        CLIPTextModel,
        CLIPTokenizer,
    )

    if encoder == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    elif encoder == "clip":
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
    else:
        raise ValueError("Unknown encoder type. Please use 'distilbert' or 'clip'.")
    return tokenizer, model

def encode_text(text, encoder, tokenizer, model):
    if encoder == "distilbert":
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the representation of the [CLS] token
        return outputs.last_hidden_state[:, 0, :].numpy().tolist()
    elif encoder == "clip":
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=77
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # Average the embeddings across tokens for CLIP to get a sentence representation
        return outputs.last_hidden_state.mean(dim=1).numpy().tolist()

def generate_command_embeddings(
        unique_phase_folder_names, 
        encoder, 
        tokenizer, 
        model
    ):
    # Returns a dictionary containing the phase command as key and a tuple of 
    # the phase command and phase embedding as value
    phase_command_embeddings_dict = {}    
    for phase_folder_name in unique_phase_folder_names:
        # Extract the phase command from the folder name
        phase_command = phase_folder_name.split("_")[1]
        embedding = encode_text(phase_command, encoder, tokenizer, model)
        phase_command_embeddings_dict[phase_folder_name]= (phase_command, embedding)

    return phase_command_embeddings_dict

def get_valid_ep_start_end_indices(
        ep_dir_path, # path to the episode directory
        # before_phase_offset, 
        # after_phase_offset, 
        # use_kinematic_indices_flag=True
    ):
    # Load the start and end indices for the current demo as the valid range of the demo
    num_ep_frames = len(os.listdir(Path(ep_dir_path).joinpath(constants.THIRD_PERSON_CAM_DIR_NAME)))
    start, end = 0, num_ep_frames - 1
    demo_num_frames_valid = end - start + 1
    
    return start, end, demo_num_frames_valid