import torch

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
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs)
        # Use the representation of the [CLS] token
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()
    elif encoder == "clip":
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=77
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # Average the embeddings across tokens for CLIP to get a sentence representation
        return outputs.last_hidden_state.mean(dim=1).numpy().tolist()