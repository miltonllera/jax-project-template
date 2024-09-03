import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float, Int


class TokenEmbedder(eqx.Module):
    token_embedder: eqx.nn.Embedding
    segment_embedder: eqx.nn.Embedding
    position_embedder: eqx.nn.Embedding
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        type_vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        dropout_rate: float,
        key: jax.Array,
    ):
        token_key, segment_key, position_key = jr.split(key, 3)

        self.token_embedder = eqx.nn.Embedding(
            num_embeddings=vocab_size, embedding_size=embedding_size, key=token_key
        )

        if type_vocab_size > 0:
            self.segment_embedder = eqx.nn.Embedding(
                num_embeddings=type_vocab_size,
                embedding_size=embedding_size,
                key=segment_key,
            )
        else:
            # Use this in case we have no segment information. I am not even sure what this is.
            self.segment_embedder = lambda x: jnp.zeros_like(x)  # type: ignore

        self.position_embedder = eqx.nn.Embedding(
            num_embeddings=max_length, embedding_size=embedding_size, key=position_key
        )
        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        token_ids: Int[Array, "S"],
        position_ids: Int[Array, "S"],
        segment_ids: Int[Array, "S"],
        enable_dropout: bool = False,
        key: jax.Array | None = None,
    ) -> Float[Array, "S E"]:
        tokens = jax.vmap(self.token_embedder)(token_ids)
        segments = jax.vmap(self.segment_embedder)(segment_ids)
        positions = jax.vmap(self.position_embedder)(position_ids)

        embedded_inputs = tokens + segments + positions

        embedded_inputs = jax.vmap(self.layernorm)(embedded_inputs)
        return self.dropout(embedded_inputs, inference=not enable_dropout, key=key)
