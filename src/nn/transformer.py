import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int

from src.nn.embedding import TokenEmbedder
from src.nn.attn import MultiheadAttention


class FeedForwardBlock(eqx.Module):
    """A single transformer feed forward block."""

    mlp: eqx.nn.Linear
    output: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout_rate: float,
        key: jax.Array,
    ):
        mlp_key, output_key = jax.random.split(key)
        self.mlp = eqx.nn.Linear(
            in_features=hidden_size, out_features=intermediate_size, key=mlp_key
        )
        self.output = eqx.nn.Linear(
            in_features=intermediate_size, out_features=hidden_size, key=output_key
        )

        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Float[Array, "H"],
        enable_dropout: bool = True,
        key: jax.Array | None = None,
    ) -> Float[Array, "H"]:
        # Feed-forward.
        hidden = self.mlp(inputs)
        hidden = jax.nn.gelu(hidden)

        # Project back to input size.
        output = self.output(hidden)
        output = self.dropout(output, inference=not enable_dropout, key=key)

        # Residual and layer norm.
        return self.layernorm(output + inputs)


class AttentionBlock(eqx.Module):
    """A single transformer attention block."""

    attention: eqx.nn.MultiheadAttention
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    num_heads: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: jax.Array,
    ):
        self.num_heads = num_heads
        self.attention = MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=attention_dropout_rate,  # type: ignore
            key=key,  # type: ignore
        )
        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Float[Array, "S E"],
        mask: Int[Array, "S"] | None,
        enable_dropout: bool = False,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, "S E"]:
        if mask is not None:
            mask = self.make_self_attention_mask(mask)
        attention_key, dropout_key = (
            (None, None) if key is None else jax.random.split(key)
        )

        attention_output, _ = self.attention(
            query=inputs,
            key_=inputs,
            value=inputs,
            mask=mask,
            inference=not enable_dropout,
            key=attention_key,
        )

        result = attention_output
        result = self.dropout(result, inference=not enable_dropout, key=dropout_key)
        return jax.vmap(self.layernorm)(result + inputs)

    def make_self_attention_mask(
        self, mask: Int[Array, "S"]
    ) -> Float[Array, "H S S"]:
        """Create self-attention mask from sequence-level mask."""
        mask = mask[..., None] * mask[..., None, :]
        mask = jnp.expand_dims(mask, axis=-3)
        mask = jnp.repeat(mask, repeats=self.num_heads, axis=-3)
        return mask.astype(jnp.float32)


class TransformerLayer(eqx.Module):
    """A single transformer layer."""

    attention_block: AttentionBlock
    ff_block: FeedForwardBlock

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: jax.Array,
    ):
        attention_key, ff_key = jax.random.split(key)

        self.attention_block = AttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            key=attention_key,
        )
        self.ff_block = FeedForwardBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate,
            key=ff_key,
        )

    def __call__(
        self,
        inputs: Float[Array, "S E"],
        mask: Int[Array, "S"] | None = None,
        *,
        enable_dropout: bool = False,
        key: jax.Array | None = None,
    ) -> Float[Array, "S E"]:
        attn_key, ff_key = (None, None) if key is None else jax.random.split(key)
        attention_output = self.attention_block(
            inputs, mask, enable_dropout=enable_dropout, key=attn_key
        )
        seq_len = inputs.shape[0]
        ff_keys = None if ff_key is None else jax.random.split(ff_key, num=seq_len)
        output = jax.vmap(self.ff_block, in_axes=(0, None, 0))(
            attention_output, enable_dropout, ff_keys
        )
        return output


class TransformerEncoder(eqx.Module):
    embedder_block: TokenEmbedder
    layers: list[TransformerLayer]

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        type_vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: jax.Array,
    ):
        embedder_key, layer_key = jax.random.split(key)
        self.embedder_block = TokenEmbedder(
            vocab_size=vocab_size,
            max_length=max_length,
            type_vocab_size=type_vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            key=embedder_key,
        )

        layer_keys = jax.random.split(layer_key, num=num_layers)
        self.layers = []
        for layer_key in layer_keys:
            self.layers.append(
                TransformerLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    key=layer_key,
                )
            )

    def __call__(
        self,
        token_ids: Int[Array, "S"],
        position_ids: Int[Array, "S"],
        segment_ids: Int[Array, "S"],
        *,
        enable_dropout: bool = False,
        key: jax.Array | None = None,
    ) -> Float[Array, "S E"]:
        emb_key, t_key = (None, None) if key is None else jax.random.split(key)

        embeddings = self.embedder_block(
            token_ids=token_ids,
            position_ids=position_ids,
            segment_ids=segment_ids,
            enable_dropout=enable_dropout,
            key=emb_key,
        )

        # We assume that all 0-values should be masked out.
        mask = jnp.asarray(token_ids != 0, dtype=jnp.int32)

        x = embeddings
        for layer in self.layers:
            l_key, t_key = (None, None) if t_key is None else jax.random.split(t_key)
            x = layer(x, mask, enable_dropout=enable_dropout, key=l_key)

        return x
