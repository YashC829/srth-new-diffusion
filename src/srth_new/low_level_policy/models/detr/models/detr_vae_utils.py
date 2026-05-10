import numpy as np
import torch
import torch.nn as nn

from .detr_vae import DETRVAE, DETRVAE_Decoder
from .backbone import build_image_backbone
from .transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    build_transformer,
    build_transformer_decoder,
)

def build(args):

    state_dim = args.action_dim  # TODO hardcode
    print("model type", args.model_type)
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_image_backbone(args)
        backbones.append(backbone)

    if args.model_type == "ACT":
        transformer = build_transformer(args)

        encoder = build_encoder(args)

        model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=state_dim,
            num_queries=args.num_queries,
            camera_names=args.camera_names,
            use_language=args.use_language,
            use_film="film" in args.backbone,
        )
    elif args.model_type == "SRT":
        transformer_decoder = build_transformer_decoder(args)
        print("use language:", args.use_language)
        model = DETRVAE_Decoder(
            backbones,
            transformer_decoder,
            state_dim=state_dim,
            num_queries=args.num_queries,
            camera_names=args.camera_names,
            action_dim=args.action_dim,
            use_language=args.use_language,
            use_film="film" in args.backbone,
        )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model

def build_encoder(args):
    d_model = args.hidden_dim  # 256
    dropout = args.dropout  # 0.1
    nhead = args.nheads  # 8
    dim_feedforward = args.dim_feedforward  # 2048
    num_encoder_layers = args.enc_layers  # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(
        d_model, nhead, dim_feedforward, dropout, activation, normalize_before
    )
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder