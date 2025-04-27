# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple
import pdb

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
# from fairseq.modules.causal_lm_layer import CausalLayer


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('transformer_wait_info')
class TransformerWaitInfoModel(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True
        self.wait_predictor = nn.Linear(self.encoder.output_dim, 1)  # One logit for wait/no-wait
    
    @classmethod
    def build_model(cls, cfg, task):
        encoder = TransformerWaitInfoEncoder(
            cfg,
            task.source_dictionary,
            Embedding(len(task.source_dictionary), cfg.encoder_embed_dim, task.source_dictionary.pad())
        )
        decoder = TransformerWaitInfoDecoder(
            cfg,
            task.target_dictionary,
            Embedding(len(task.target_dictionary), cfg.decoder_embed_dim, task.target_dictionary.pad())
        )
        return cls(cfg, encoder, decoder)

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            train_waitk_lagging=None,
    ):
        encoder_out, attn, info = self.encoder(  # Adjust unpacking to match the number of return values
        src_tokens,
        src_lengths=src_lengths,
        return_all_hiddens=return_all_hiddens
    )

        # 2. Decode
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            train_waitk_lagging=train_waitk_lagging,
            src_info=attn,
        )
        decoder_out[1]["src_info"] = attn

        # 3. Predict wait logits
        encoder_hidden_states = encoder_out[0]  # (src_len, batch, dim)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)  # (batch, src_len, dim)

        wait_logits = self.wait_predictor(encoder_hidden_states)  # (batch, src_len, 1)

        # 4. Add wait_logits to decoder_out
        decoder_out[1]["wait_logits"] = wait_logits  # attach as extra output

        return decoder_out


class TransformerWaitInfoEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)
        self._future_mask = torch.empty(0)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.info_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.info_proj2 = nn.Linear(embed_dim, 1, bias=False)  # For wait prediction
        self.output_dim = args.encoder_embed_dim

    def build_encoder_layer(self, args):
        return TransformerEncoderLayer(args)

    def forward_embedding(
            self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
            self,
            src_tokens,
            src_lengths,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        x = x.transpose(0, 1)

        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        encoder_states = [] if return_all_hiddens else None

        # Calculate source info and wait logits
        src_info = 2 * torch.sigmoid(
            self.info_proj2(torch.tanh(self.info_proj(torch.tanh(x)))))  # (T, B, 1)

        # Wait logits for the wait-info strategy
        wait_logits = self.info_proj2(torch.tanh(self.info_proj(torch.tanh(x))))  # (T, B, 1)
        wait_logits = wait_logits.transpose(0, 1)  # (B, T, 1)

        # Process through layers
        for layer in self.layers:
            x, fc_result = layer(  # Adjust unpacking to match the number of return values
                x,
                encoder_padding_mask,
                attn_mask=self.buffered_future_mask(x),
                info=src_info,  # Pass the info argument here
            )
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # Return encoder output and wait_logits for decoder
        return (
            EncoderOut(
                encoder_out=x,  # T x B x C
                encoder_padding_mask=encoder_padding_mask,  # B x T
                encoder_embedding=encoder_embedding,  # B x T x C
                encoder_states=encoder_states,  # List[T x B x C]
                src_tokens=None,
                src_lengths=None,
            ),
            src_info,  # Source info for attention, etc.
            wait_logits,  # Pass wait logits to decoder
        )

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if self._future_mask.size(0) == 0 or not self._future_mask.device == tensor.device or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1)
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        encoder_padding_mask = encoder_out.encoder_padding_mask
        encoder_embedding = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = f"{name}.embed_positions.weights"
            if weights_key in state_dict:
                print(f"deleting {weights_key}")
                del state_dict[weights_key]
            state_dict[f"{name}.embed_positions._float_tensor"] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            self.layers[i].upgrade_state_dict_named(
                state_dict, f"{name}.layers.{i}"
            )
        version_key = f"{name}.version"
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            self.layer_norm = None
            state_dict[version_key] = torch.Tensor([1])
        return state_dict



class TransformerWaitInfoDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim**-0.5
            )

        self.info_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.info_proj2 = nn.Linear(embed_dim, 1, bias=False)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return TransformerDecoderLayer(args, no_encoder_attn)

    def forward(
        self,
        prev_output_tokens,
        train_waitk_lagging=None,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        test_waitk_lagging=None,
        src_info=None,
    ):
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            train_waitk_lagging=train_waitk_lagging,
            test_waitk_lagging=test_waitk_lagging,
            src_info=src_info,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        train_waitk_lagging=None,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        test_waitk_lagging=None,
        src_info=None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            train_waitk_lagging,
            test_waitk_lagging,
            src_info,
        )

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        train_waitk_lagging=None,
        test_waitk_lagging=None,
        src_info=None,
    ):
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        attn_list = []

        # target info
        tgt_info = 2 * torch.sigmoid(
            self.info_proj2(torch.tanh(self.info_proj(torch.tanh(x))))
        ).transpose(0, 1)
        tgt_info = tgt_info.masked_fill(
            (prev_output_tokens == 2).unsqueeze(-1), float(1.0)
        )

        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, extra = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                train_waitk_lagging=train_waitk_lagging,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=True,
                test_waitk_lagging=test_waitk_lagging,
                src_info=src_info,
                tgt_info=tgt_info,
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)
            attn_list.append(layer_attn)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {
            "attn": [attn],
            "inner_states": inner_states,
            "attention": torch.cat(attn_list, dim=0),
            "tgt_info": tgt_info,
        }

    def generate_seg_from_self_attn(self, attn):
        bsz = attn.size(0)
        src_len = attn.size(1)
        attn_to_self = (
            attn_to_self
            >= torch.cat(
                (attn_to_self.new_zeros(bsz, src_len, 1), attn_to_self[:, :, :-1]),
                dim=-1,
            )
        ).int()
        attn_to_self = torch.diagonal(attn, dim1=-2, dim2=-1)
        # attn_to_self=(attn_to_self>0.1+(1/torch.arange(1,src_len+1,device=attn.device)).unsqueeze(0)).int()
        # attn_to_self=(attn_to_self==attn.max(dim=-1,keepdim=False)[0]).int()
        # attn_to_self=(attn_to_self>=0.2).int()
        # pdb.set_trace()
        attn_to_self = torch.cat(
            (
                attn_to_self.new_zeros(bsz, 1),
                attn_to_self[:, 1:-1],
                attn_to_self.new_ones(bsz, 1),
            ),
            dim=1,
        )
        attn_to_self = attn_to_self * torch.arange(
            0, src_len, device=attn.device
        ).unsqueeze(0)
        attn_to_self = attn_to_self.masked_fill((attn_to_self == 0), 1024)
        src_info_seg = torch.cummin(attn_to_self.flip([1]), dim=1)[0].flip([1])
        # pdb.set_trace()
        return src_info_seg

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

@register_model_architecture('transformer_wait_info', 'transformer_wait_info_arch')
def transformer_waitinfo_architecture(args):
    # Encoder arguments
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_layerdrop = getattr(args, 'encoder_layerdrop', 0.0)

    # Add adaptive_input argument
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    # Add quant_noise_pq argument
    args.quant_noise_pq = getattr(args, 'quant_noise_pq', 0.0)

    # Add encoder_normalize_before argument
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)

    # Decoder arguments
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)

    # Add decoder_output_dim argument
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)

    # Add decoder_layerdrop argument
    args.decoder_layerdrop = getattr(args, 'decoder_layerdrop', 0.0)

    # Add decoder_learned_pos argument
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)

    # Add decoder_normalize_before argument
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)  # Add this line

    # Add adaptive_softmax_cutoff argument
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)  # Add this line

    # Dropout argument
    args.dropout = getattr(args, 'dropout', 0.3)

    # Label smoothing
    args.label_smoothing = getattr(args, 'label_smoothing', 0.1)

    # Share decoder input-output embedding
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)

    # Maximum source/target positions
    args.max_source_positions = getattr(args, 'max_source_positions', 1024)
    args.max_target_positions = getattr(args, 'max_target_positions', 1024)

    # Add no_scale_embedding argument
    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)

    # Add no_token_positional_embeddings argument
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)

    # Add encoder_learned_pos argument
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
