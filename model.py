"""Layernorm"""

from dataclasses import dataclass
from typing import Callable, Optional
import equinox as eqx
import equinox.nn as nn
import jax.experimental
import jax.numpy as jnp
import jax


@dataclass
class GPTConfig:
    block_size: int = 100
    enc_vocab_size: int = 50304
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )


class CausalSelfAttention(eqx.Module):
    c_attn: nn.Linear
    c_proj: nn.Linear

    attn_dropout: nn.Dropout
    resid_dropout: nn.Dropout

    mask: jax.Array

    def __init__(self, config, key=None):
        key1, key2 = jax.random.split(key)
        # # Projection for W_1, W_2, W_3 in a batch
        self.c_attn = nn.Linear(
            config.n_embd, config.n_embd * 3, use_bias=config.bias, key=key1
        )
        # Output proj
        self.c_proj = nn.Linear(
            config.n_embd, config.n_embd, use_bias=config.bias, key=key2
        )

        # # Regularisation
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # # # self.n_head = config.n_head
        # # # self.n_embd = config.n_embd

        self.mask = jnp.tril(jnp.ones((config.block_size, config.block_size)))

    @eqx.filter_jit
    def __call__(
        self,
        x,
        key: Optional[jax.Array] = None,
    ):
        (t, _) = x.shape
        # X is of shape (seq, n_embd)
        # Project into the head dim.
        # jax.debug.breakpoint()
        # print(x)
        qkv = jax.vmap(self.c_attn)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        # Dim of (seq, head_dim)

        kq = jnp.matmul(k, jnp.transpose(q))
        # Dim of (seq, seq), a matrix showing which tokens are interested in each other.
        # Mask to make causal
        # mask = jax.lax.stop_gradient(self.mask)

        kq = jnp.where(
            jnp.equal(jax.lax.stop_gradient(self.mask[:t, :t]), 0), -jnp.inf, kq
        )  # Trick to lower compute
        kq = jax.nn.softmax(kq, axis=-1)
        # Add att dropout
        k1, k2 = (None, None) if key is None else jax.random.split(key)
        kq = self.attn_dropout(kq, key=k1)
        outs = jnp.matmul(kq, v)
        outs = jax.vmap(self.c_proj)(outs)

        # Add residual dropout
        outs = self.resid_dropout(outs, key=k2)
        return outs


class CrossAttention(eqx.Module):
    dec_attn: nn.Linear
    enc_attn: nn.Linear
    c_proj: nn.Linear

    attn_dropout: nn.Dropout
    resid_dropout: nn.Dropout

    mask: jax.Array

    def __init__(self, config, key=None):
        key1, key2 = jax.random.split(key)
        # # Projection for W_1, W_2, W_3 in a batch
        self.dec_attn = nn.Linear(
            config.n_embd, config.n_embd, use_bias=config.bias, key=key1
        )
        self.enc_attn = nn.Linear(
            config.n_embd, config.n_embd * 2, use_bias=config.bias, key=key1
        )
        # Output proj
        self.c_proj = nn.Linear(
            config.n_embd, config.n_embd, use_bias=config.bias, key=key2
        )

        # # Regularisation
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # # # self.n_head = config.n_head
        # # # self.n_embd = config.n_embd

        self.mask = jnp.tril(jnp.ones((config.block_size, config.block_size)))

    @eqx.filter_jit
    def __call__(
        self,
        x,
        enc_x,
        key: Optional[jax.Array] = None,
    ):
        (t, _) = x.shape
        q = jax.vmap(self.dec_attn)(x)
        kv = jax.vmap(self.enc_attn)(enc_x)
        k, v = jnp.split(kv, 2, axis=-1)
        # Dim of (seq, head_dim)
        # Q: (block_size, n_embd)
        # K: (enc_block_size, n_embd)
        # V: (enc_block_size, n_embd)
        kq = jnp.matmul(k, jnp.transpose(q))
        # KQ: (enc_block_size, block_size)

        kq = jnp.where(
            jnp.equal(jax.lax.stop_gradient(self.mask[:t, :t]), 0), -jnp.inf, kq
        )  # Trick to lower compute
        kq = jax.nn.softmax(kq, axis=-1)
        # Add att dropout
        k1, k2 = (None, None) if key is None else jax.random.split(key)
        kq = self.attn_dropout(kq, key=k1)
        outs = jnp.matmul(kq, v)
        outs = jax.vmap(self.c_proj)(outs)

        # Add residual dropout
        outs = self.resid_dropout(outs, key=k2)
        return outs


class SelfAttention(eqx.Module):
    c_attn: nn.Linear
    c_proj: nn.Linear

    attn_dropout: nn.Dropout
    resid_dropout: nn.Dropout

    def __init__(self, config, key=None):
        key1, key2 = jax.random.split(key)
        # # Projection for W_1, W_2, W_3 in a batch
        self.c_attn = nn.Linear(
            config.n_embd, config.n_embd * 3, use_bias=config.bias, key=key1
        )
        # Output proj
        self.c_proj = nn.Linear(
            config.n_embd, config.n_embd, use_bias=config.bias, key=key2
        )

        # # Regularisation
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    @eqx.filter_jit
    def __call__(
        self,
        x,
        key: Optional[jax.Array] = None,
    ):
        qkv = jax.vmap(self.c_attn)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        kq = jnp.matmul(k, jnp.transpose(q))
        kq = jax.nn.softmax(kq, axis=-1)
        k1, k2 = (None, None) if key is None else jax.random.split(key)
        kq = self.attn_dropout(kq, key=k1)
        outs = jnp.matmul(kq, v)
        outs = jax.vmap(self.c_proj)(outs)
        outs = self.resid_dropout(outs, key=k2)
        return outs


class MLP(eqx.Module):
    c_fc: nn.Linear
    c_proj: nn.Linear

    dropout: nn.Dropout

    def __init__(self, config, key=None):
        key1, key2 = jax.random.split(key)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, key=key1)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, key=key2)
        self.dropout = nn.Dropout(config.dropout)

    @eqx.filter_jit
    def __call__(self, x, key=None):
        y = jax.vmap(self.c_fc)(x)
        y = jax.nn.gelu(y)
        y = jax.vmap(self.c_proj)(y)
        return self.dropout(y, key=key)


class Block(eqx.Module):
    mlp: MLP
    attn: CausalSelfAttention

    ln_1: nn.LayerNorm
    ln_2: nn.LayerNorm

    def __init__(self, config, key=None):
        key1, key2 = jax.random.split(key)
        self.mlp = MLP(config, key=key1)
        self.attn = CausalSelfAttention(config, key=key2)
        self.ln_1 = nn.LayerNorm(config.n_embd, use_bias=config.bias)
        self.ln_2 = nn.LayerNorm(config.n_embd, use_bias=config.bias)

    @eqx.filter_jit
    def __call__(self, x, key=None):
        key1, key2 = (None, None) if key is None else jax.random.split(key)
        y = jax.vmap(self.ln_1)(x)
        x = self.attn(y, key=key1) + x
        y = jax.vmap(self.ln_2)(x)
        x = self.mlp(y, key=key2) + x
        return x


class EncBlock(eqx.Module):
    mlp: MLP
    attn: SelfAttention

    ln_1: nn.LayerNorm
    ln_2: nn.LayerNorm

    def __init__(self, config, key=None):
        key1, key2 = jax.random.split(key)
        self.mlp = MLP(config, key=key1)
        self.attn = SelfAttention(config, key=key2)
        self.ln_1 = nn.LayerNorm(config.n_embd, use_bias=config.bias)
        self.ln_2 = nn.LayerNorm(config.n_embd, use_bias=config.bias)

    @eqx.filter_jit
    def __call__(self, x, key=None):
        key1, key2 = (None, None) if key is None else jax.random.split(key)
        y = jax.vmap(self.ln_1)(x)
        x = self.attn(y, key=key1) + x
        y = jax.vmap(self.ln_2)(x)
        x = self.mlp(y, key=key2) + x
        return x


class CrossBlock(eqx.Module):
    mlp: MLP
    attn: CrossAttention

    ln_1: nn.LayerNorm
    ln_2: nn.LayerNorm

    def __init__(self, config, key=None):
        key1, key2 = jax.random.split(key)
        self.mlp = MLP(config, key=key1)
        self.attn = CrossAttention(config, key=key2)
        self.ln_1 = nn.LayerNorm(config.n_embd, use_bias=config.bias)
        self.ln_2 = nn.LayerNorm(config.n_embd, use_bias=config.bias)

    @eqx.filter_jit
    def __call__(self, x, enc_x, key=None):
        key1, key2 = (None, None) if key is None else jax.random.split(key)
        y = jax.vmap(self.ln_1)(x)
        x = self.attn(y, enc_x, key=key1) + x
        y = jax.vmap(self.ln_2)(x)
        x = self.mlp(y, key=key2) + x
        return x


class GPT(eqx.Module):
    layers: list
    lm_head: nn.Linear
    wpe: nn.Embedding
    wte: nn.Embedding
    drop: nn.Dropout
    ln_f: nn.LayerNorm

    enc_wpe: nn.Embedding
    enc_wte: nn.Embedding
    enc_block: EncBlock
    enc_drop: nn.Dropout

    cross_block: CrossBlock

    config: GPTConfig = eqx.field(static=True)

    def __init__(self, config, key=None):
        key1, key2, key3, key4, key5, key6, key7 = jax.random.split(key, 7)
        self.config = config

        self.drop = nn.Dropout(config.dropout)
        self.ln_f = nn.LayerNorm(config.n_embd, use_bias=config.bias)

        # Use weight typing

        self.wte = nn.Embedding(config.vocab_size, config.n_embd, key=key1)
        self.wpe = nn.Embedding(config.vocab_size, config.n_embd, key=key2)

        # Slightly different init as don't have module dict in JAX/EQX
        self.layers = [
            Block(config, key=k) for k in jax.random.split(key3, config.n_layer)
        ]

        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, use_bias=False, key=key4
        )

        self.enc_wpe = nn.Embedding(config.enc_vocab_size, config.n_embd, key=key5)
        self.enc_wte = nn.Embedding(config.enc_vocab_size, config.n_embd, key=key6)
        self.enc_block = EncBlock(config, key=key7)
        self.enc_drop = nn.Dropout(config.dropout)

        self.cross_block = CrossBlock(config, key=key7)

        # where = lambda embed_and_lin: embed_and_lin[1].weight
        # get = lambda embed_and_lin: embed_and_lin[0].weight
        # self.wte_and_lmhead = eqx.nn.Shared((wte, lm_head), where, get)

    @eqx.filter_jit
    def __call__(
        self, x, enc_x, train_mode=True, key=None
    ):  # We don't assert seq length as jax needs static shapes. Check elsewhere.
        (enc_t,) = enc_x.shape
        enc_pos = jnp.arange(0, enc_t)
        enc_tok_emb = jax.vmap(self.enc_wte)(enc_x)
        enc_pos_emb = jax.vmap(self.enc_wpe)(enc_pos)
        enc_x = enc_tok_emb + enc_pos_emb
        enc_x = self.enc_drop(enc_x, key=key)
        enc_x = self.enc_block(enc_x, key)

        (t,) = x.shape
        pos = jnp.arange(0, t)
        tok_emb = jax.vmap(self.wte)(x)
        pos_emb = jax.vmap(self.wpe)(pos)
        x = tok_emb + pos_emb

        x = self.drop(x, key=key)
        x = self.cross_block(x, enc_x, key=key)
        for layer in self.layers:
            key, k = (None, None) if key is None else jax.random.split(key)
            x = layer(x, k)


        x = jax.vmap(self.ln_f)(x)

        if train_mode:
            return jax.vmap(self.lm_head)(x)
        else:
            return jax.vmap(self.lm_head)(
                x[[-1], :]
            )  # note: using list [-1] to preserve the time dim

    def get_num_params(self):
        n_params = sum(
            x.size for x in jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_array))
        )
        # Min the emb once as is shared
        n_params -= self.wte_and_lmhead()[0].weight.size
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()

        L, H, Q, T = (
            self.config.n_layer,
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.block_size,
        )

        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 5.1e12  # Mac pro m1 flops
        mfu = flops_achieved / flops_promised
        return mfu

    @staticmethod
    def _init_weights(model: eqx.Module, config: GPTConfig, key=None):
        def init_layer(model, is_layer: Callable, mean: float, std: float):
            get_weights = lambda m: [
                x.weight
                for x in jax.tree_util.tree_leaves(m, is_leaf=is_layer)
                if is_layer(x)
            ]
            weights = get_weights(model)

            new_weights = [
                (
                    jax.random.normal(k, weight.shape) * std + mean
                    if not isinstance(
                        weight, nn._shared.SharedNode
                    )  # SharedNode is a place holder value as we only have one matrix not two.
                    else weight
                )
                for weight, k in zip(weights, jax.random.split(key, len(weights)))
            ]

            return eqx.tree_at(get_weights, model, new_weights)

        def init_linear(model):
            is_linear = lambda x: isinstance(x, eqx.nn.Linear)

            model = init_layer(model, is_linear, mean=0.0, std=0.2)

            get_biases = lambda m: [
                x.bias
                for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                if is_linear(x) and x.bias is not None
            ]
            biases = get_biases(model)

            new_biases = [jnp.zeros_like(bias) for bias in biases]

            return eqx.tree_at(get_biases, model, new_biases)

        def init_embedding(model):
            is_embedding = lambda x: isinstance(x, eqx.nn.Embedding)

            return init_layer(model, is_embedding, mean=0.0, std=0.2)

        def init_c_proj_weights_with_normal(model, key):

            def hop(path, x):
                nonlocal key
                if "c_proj.weight" in jax.tree_util.keystr(path):
                    key, k = jax.random.split(key)
                    return jax.random.normal(k, x.shape) * 0.02
                return x

            return jax.tree_util.tree_map_with_path(hop, model)

        model = init_linear(model)
        model = init_embedding(model)
        # apply special scaled init to the residual projections, per GPT-2 paper
        model = init_c_proj_weights_with_normal(model, key)

        return model

    @staticmethod
    def create_instance(config, key):
        key1, key2 = jax.random.split(key, 2)

        inst = GPT(config, key1)
        new_inst = GPT._init_weights(inst, config, key2)

        return new_inst

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, key=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            print(idx)
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.shape[1] <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits = jax.vmap(self, in_axes=(0, None))(idx_cond, False)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = jax.lax.top_k(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            key, k = jax.random.split(key)
            idx_next = jax.random.categorical(k, logits)
            # idx_next = jax.numpy.argmax(logits, axis=-1)

            # append sampled index to the running sequence and continue
            idx = jnp.concat((idx, idx_next), axis=-1)

        return idx
