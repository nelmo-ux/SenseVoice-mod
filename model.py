
import time
import torch
from torch import nn
import torch.nn.functional as F
from typing import Iterable, Optional, Sequence, Tuple, Union

from funasr.register import tables
from funasr.models.ctc.ctc import CTC
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.paraformer.search import Hypothesis
from funasr.models.scama.chunk_utilis import overlap_chunk
from funasr.models.transformer.embedding import StreamSinusoidalPositionEncoder
from funasr.train_utils.device_funcs import force_gatherable
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.metrics.compute_acc import compute_accuracy, th_accuracy
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from utils.ctc_alignment import ctc_forced_align

class SinusoidalPositionEncoder(torch.nn.Module):
    """ """

    def __init__(self, d_model=80, dropout_rate=0.1):
        super().__init__()

    def encode(
        self, positions: torch.Tensor = None, depth: int = None, dtype: torch.dtype = torch.float32
    ):
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(torch.tensor([10000], dtype=dtype, device=device)) / (
            depth / 2 - 1
        )
        inv_timescales = torch.exp(
            torch.arange(depth / 2, device=device).type(dtype) * (-log_timescale_increment)
        )
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(
            inv_timescales, [1, 1, -1]
        )
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return encoding.type(dtype)

    def forward(self, x):
        batch_size, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device)[None, :]
        position_encoding = self.encode(positions, input_dim, x.dtype).to(x.device)

        return x + position_encoding


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class MultiHeadedAttentionSANM(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self,
        n_head,
        in_feat,
        n_feat,
        dropout_rate,
        kernel_size,
        sanm_shfit=0,
        lora_list=None,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        # self.linear_q = nn.Linear(n_feat, n_feat)
        # self.linear_k = nn.Linear(n_feat, n_feat)
        # self.linear_v = nn.Linear(n_feat, n_feat)

        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        # padding
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward_fsmn(self, inputs, mask, mask_shfit_chunk=None):
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

    def forward_qkv(self, x):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        b, t, d = x.size()
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q_h = torch.reshape(q, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time1, d_k)
        k_h = torch.reshape(k, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)
        v_h = torch.reshape(v, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)

        return q_h, k_h, v_h, v

    def forward_attention(self, value, scores, mask, mask_att_chunk_encoder=None):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            if mask_att_chunk_encoder is not None:
                mask = mask * mask_att_chunk_encoder

            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)

            min_value = -float(
                "inf"
            )  # float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, x, mask, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask, mask_shfit_chunk)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask, mask_att_chunk_encoder)
        return att_outs + fsmn_memory

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q_h, k_h, v_h, v = self.forward_qkv(x)
        if chunk_size is not None and look_back > 0 or look_back == -1:
            if cache is not None:
                k_h_stride = k_h[:, :, : -(chunk_size[2]), :]
                v_h_stride = v_h[:, :, : -(chunk_size[2]), :]
                k_h = torch.cat((cache["k"], k_h), dim=2)
                v_h = torch.cat((cache["v"], v_h), dim=2)

                cache["k"] = torch.cat((cache["k"], k_h_stride), dim=2)
                cache["v"] = torch.cat((cache["v"], v_h_stride), dim=2)
                if look_back != -1:
                    cache["k"] = cache["k"][:, :, -(look_back * chunk_size[1]) :, :]
                    cache["v"] = cache["v"][:, :, -(look_back * chunk_size[1]) :, :]
            else:
                cache_tmp = {
                    "k": k_h[:, :, : -(chunk_size[2]), :],
                    "v": v_h[:, :, : -(chunk_size[2]), :],
                }
                cache = cache_tmp
        fsmn_memory = self.forward_fsmn(v, None)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, None)
        return att_outs + fsmn_memory, cache


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask = mask.detach()

    return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


class EncoderLayerSANM(nn.Module):
    def __init__(
        self,
        in_size,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayerSANM, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate

    def forward(self, x, mask, cache=None, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.concat_after:
            x_concat = torch.cat(
                (
                    x,
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    ),
                ),
                dim=-1,
            )
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
            else:
                x = stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
            else:
                x = stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.in_size == self.size:
            attn, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)
            x = residual + attn
        else:
            x, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm2(x)

        return x, cache


@tables.register("encoder_classes", "SenseVoiceEncoderSmall")
class SenseVoiceEncoderSmall(nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    SCAMA: Streaming chunk-aware multihead attention for online end-to-end speech recognition
    https://arxiv.org/abs/2006.01713
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        tp_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=SinusoidalPositionEncoder,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        kernel_size: int = 11,
        sanm_shfit: int = 0,
        selfattention_layer_type: str = "sanm",
        chunk_size: Optional[Union[int, Sequence[int]]] = None,
        stride: Optional[Union[int, Sequence[int]]] = None,
        pad_left: Optional[Union[int, Sequence[int]]] = None,
        encoder_att_look_back_factor: Optional[Union[int, Sequence[int]]] = None,
        decoder_att_look_back_factor: Optional[Union[int, Sequence[int]]] = None,
        **kwargs,
    ):
        """Build the encoder.

        Args:
            chunk_size: Total chunk width(s) in frames used by the chunk-mask training
                path, i.e. ``pad_left + stride + pad_right``. ``None`` (the default)
                disables the chunk path completely and ``forward`` runs the unmodified
                full-attention computation. A tuple with several entries turns on
                dynamic chunk training: one entry is drawn uniformly at random per
                training step. An entry ``<= 0`` is the *full attention sentinel* -
                when that entry is drawn the step runs the plain full-attention path,
                which is how full-attention batches are mixed into chunk training.
                ``-1`` is the canonical spelling of the sentinel.
            stride: Committed frames per chunk, one entry per ``chunk_size`` entry (a
                single entry is broadcast). Defaults to ``chunk_size`` (no right
                context). Must satisfy ``0 < stride <= chunk_size``.
            pad_left: Left context frames per chunk, broadcast like ``stride``.
                Defaults to ``0``. ``pad_right`` is implied as
                ``chunk_size - stride - pad_left`` and must be ``>= 0``.
            encoder_att_look_back_factor: How many previous chunks the encoder
                self-attention may attend to. Defaults to ``(1,)``.
            decoder_att_look_back_factor: Unused by SenseVoice (there is no decoder);
                accepted and forwarded because ``overlap_chunk`` requires it.

        Note:
            The 3-tuple passed to :meth:`forward_chunk` via the cache has a *different*
            meaning: ``[pad_left, stride, pad_right]``. Use :meth:`init_chunk_cache` to
            derive it from the training configuration instead of writing it by hand.
        """
        super().__init__()
        self._output_size = output_size
        self._input_size = input_size

        self.embed = SinusoidalPositionEncoder()

        self.normalize_before = normalize_before

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
        )

        encoder_selfattn_layer = MultiHeadedAttentionSANM
        encoder_selfattn_layer_args0 = (
            attention_heads,
            input_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )

        self.encoders0 = nn.ModuleList(
            [
                EncoderLayerSANM(
                    input_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(1)
            ]
        )
        self.encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(num_blocks - 1)
            ]
        )

        self.tp_encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(tp_blocks)
            ]
        )

        self.after_norm = LayerNorm(output_size)

        self.tp_norm = LayerNorm(output_size)

        # Streaming position encoder used only by ``forward_chunk``. It holds no
        # parameters and no buffers, so ``state_dict()`` is byte-for-byte unchanged
        # and published checkpoints still load with ``strict=True``.
        self.chunk_embed = StreamSinusoidalPositionEncoder()

        self.chunk_size, self.stride, self.pad_left = None, None, None
        self.overlap_chunk_cls = None
        if chunk_size is not None:
            self.chunk_size, self.stride, self.pad_left = self._normalize_chunk_args(
                chunk_size, stride, pad_left
            )
            self.overlap_chunk_cls = overlap_chunk(
                chunk_size=self.chunk_size,
                stride=self.stride,
                pad_left=self.pad_left,
                shfit_fsmn=(kernel_size - 1) // 2,
                encoder_att_look_back_factor=self._as_int_tuple(
                    encoder_att_look_back_factor, (1,)
                ),
                decoder_att_look_back_factor=self._as_int_tuple(
                    decoder_att_look_back_factor, (1,)
                ),
            )

    @staticmethod
    def _as_int_tuple(
        value: Optional[Union[int, Sequence[int]]],
        default: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        """Coerce a scalar / sequence / omitted chunk argument to a tuple of ints.

        Args:
            value: An int, any sequence of ints (including a Hydra ``ListConfig``),
                or ``None``.
            default: Returned when ``value`` is ``None``.

        Returns:
            The coerced tuple.
        """
        if value is None:
            return default
        if isinstance(value, int):
            return (value,)
        return tuple(int(v) for v in value)

    @classmethod
    def _normalize_chunk_args(
        cls,
        chunk_size: Union[int, Sequence[int]],
        stride: Optional[Union[int, Sequence[int]]],
        pad_left: Optional[Union[int, Sequence[int]]],
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        """Validate and broadcast the chunk-mask training arguments.

        ``overlap_chunk`` broadcasts ``pad_left`` and the look-back factors itself but
        *not* ``stride``, so every tuple is expanded to ``len(chunk_size)`` here and the
        per-entry geometry is checked up front rather than failing deep inside mask
        generation.

        Args:
            chunk_size: Total chunk width(s); entries ``<= 0`` are full-attention
                sentinels and skip geometry validation.
            stride: Committed frames per chunk, or ``None`` to default to ``chunk_size``.
            pad_left: Left context frames, or ``None`` to default to ``0``.

        Returns:
            ``(chunk_size, stride, pad_left)`` as equal-length tuples of ints.

        Raises:
            ValueError: If a tuple length cannot be broadcast, or if any non-sentinel
                entry violates ``0 < stride <= chunk_size`` or
                ``0 <= pad_left <= chunk_size - stride``.
        """
        chunk_size = cls._as_int_tuple(chunk_size, ())
        if not chunk_size:
            raise ValueError("chunk_size must contain at least one entry")

        def _broadcast(value: Tuple[int, ...], name: str) -> Tuple[int, ...]:
            if len(value) == 1:
                return value * len(chunk_size)
            if len(value) != len(chunk_size):
                raise ValueError(
                    f"{name} has {len(value)} entries but chunk_size has "
                    f"{len(chunk_size)}; pass one entry per chunk_size entry or a "
                    "single entry to broadcast"
                )
            return value

        stride = _broadcast(cls._as_int_tuple(stride, chunk_size), "stride")
        pad_left = _broadcast(cls._as_int_tuple(pad_left, (0,)), "pad_left")

        for i, total in enumerate(chunk_size):
            if total <= 0:
                continue  # full-attention sentinel: geometry is not used
            if not 0 < stride[i] <= total:
                raise ValueError(
                    f"stride[{i}]={stride[i]} must satisfy 0 < stride <= "
                    f"chunk_size[{i}]={total}"
                )
            if not 0 <= pad_left[i] <= total - stride[i]:
                raise ValueError(
                    f"pad_left[{i}]={pad_left[i]} must satisfy 0 <= pad_left <= "
                    f"chunk_size[{i}] - stride[{i}] = {total - stride[i]}"
                )
        return chunk_size, stride, pad_left

    def _check_chunk_ind(self, ind: Optional[int], name: str) -> None:
        """Range-check an index into the chunk configuration tuples.

        ``overlap_chunk.random_choice`` and ``overlap_chunk.get_chunk_size`` do no
        bounds checking, so an out-of-range index surfaces as a bare ``IndexError``
        deep inside mask generation.

        Args:
            ind: Index to check. ``None`` is accepted and means "not supplied".
            name: Argument name to quote in the error message.

        Raises:
            ValueError: If the chunk path is disabled or ``ind`` is out of range.
        """
        if ind is None:
            return
        if self.overlap_chunk_cls is None:
            raise ValueError(
                f"{name}={ind} was given but this encoder was built without chunk_size"
            )
        if not 0 <= ind < len(self.chunk_size):
            raise ValueError(
                f"{name}={ind} is out of range for {len(self.chunk_size)} configured "
                "chunk size(s)"
            )

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        decoding_ind: Optional[int] = None,
        **kwargs,
    ):
        """Embed positions in tensor.

        Args:
            xs_pad: Input features (batch, time, input_size). Scaled in place, as
                before.
            ilens: Input lengths (batch,).
            decoding_ind: Chunk configuration index to use when not training. Ignored
                unless the chunk path is enabled; ``None`` uses index 0. During
                training the index is drawn at random from the configured tuple
                (dynamic chunk training).
            **kwargs: Ignored; accepted so callers can pass extra keywords.

        Returns:
            ``(xs_pad, olens)`` with the same shape and length semantics as the
            full-attention path. When the chunk path is active the chunk-expanded
            sequence is folded back to the original time axis before returning, so the
            CTC head and loss need no changes.

        Note:
            When the chunk path is active the returned time axis is truncated to
            ``ilens.max()``; the full-attention path keeps ``xs_pad.size(1)``. These
            agree for the usual case of a batch padded exactly to its longest item.
        """
        masks = sequence_mask(ilens, device=ilens.device)[:, None, :]

        xs_pad *= self.output_size() ** 0.5

        xs_pad = self.embed(xs_pad)

        self._check_chunk_ind(decoding_ind, "decoding_ind")
        chunk_outs, olens_chunk = None, None
        mask_shfit_chunk, mask_att_chunk_encoder = None, None
        if self.overlap_chunk_cls is not None:
            ind = self.overlap_chunk_cls.random_choice(self.training, decoding_ind)
            # A non-positive entry is the full-attention sentinel: leave every chunk
            # tensor as None so this step runs the plain full-attention computation.
            if self.chunk_size[ind] > 0:
                chunk_ilens = masks.squeeze(1).sum(1).to(torch.int64)
                chunk_outs = self.overlap_chunk_cls.gen_chunk_mask(chunk_ilens, ind)
                xs_pad, olens_chunk = self.overlap_chunk_cls.split_chunk(
                    xs_pad, chunk_ilens, chunk_outs=chunk_outs
                )
                masks = sequence_mask(
                    olens_chunk, maxlen=xs_pad.size(1), device=xs_pad.device
                )[:, None, :]
                mask_shfit_chunk = self.overlap_chunk_cls.get_mask_shfit_chunk(
                    chunk_outs, xs_pad.device, xs_pad.size(0), dtype=xs_pad.dtype
                )
                mask_att_chunk_encoder = self.overlap_chunk_cls.get_mask_att_chunk_encoder(
                    chunk_outs, xs_pad.device, xs_pad.size(0), dtype=xs_pad.dtype
                )

        # forward encoder1
        for layer_idx, encoder_layer in enumerate(self.encoders0):
            encoder_outs = encoder_layer(
                xs_pad, masks, None, mask_shfit_chunk, mask_att_chunk_encoder
            )
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        for layer_idx, encoder_layer in enumerate(self.encoders):
            encoder_outs = encoder_layer(
                xs_pad, masks, None, mask_shfit_chunk, mask_att_chunk_encoder
            )
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.after_norm(xs_pad)

        # forward encoder2
        olens = masks.squeeze(1).sum(1).int()

        for layer_idx, encoder_layer in enumerate(self.tp_encoders):
            encoder_outs = encoder_layer(
                xs_pad, masks, None, mask_shfit_chunk, mask_att_chunk_encoder
            )
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.tp_norm(xs_pad)

        if chunk_outs is not None:
            # Fold the chunk-expanded sequence back onto the original time axis so the
            # CTC loss is computed at full length, unlike funasr's SANMEncoderChunkOpt
            # which returns the expanded sequence. remove_chunk masks its input in
            # place, so hand it a clone and keep tp_norm's output untouched for
            # autograd.
            xs_pad, olens = self.overlap_chunk_cls.remove_chunk(
                xs_pad.clone(), olens_chunk, chunk_outs
            )
            olens = olens.int()
        return xs_pad, olens

    def init_chunk_cache(
        self,
        pad_left: Optional[int] = None,
        stride: Optional[int] = None,
        pad_right: Optional[int] = None,
        encoder_chunk_look_back: int = 0,
        batch_size: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        ind: int = 0,
    ) -> dict:
        """Build a fresh streaming cache for :meth:`forward_chunk`.

        Args:
            pad_left: Left context frames carried over from the previous chunk.
            stride: Committed frames per :meth:`forward_chunk` call. This is how many
                new frames the caller feeds each step.
            pad_right: Lookahead frames withheld from the output until the next call.
            encoder_chunk_look_back: Number of previous chunks the encoder
                self-attention may attend to through the per-layer key/value cache.
                ``0`` (the default) keeps attention inside the current window, ``-1``
                keeps every past chunk.
            batch_size: Batch size of the stream. Streaming decode is normally 1.
            device: Device for the cached tensors. Defaults to CPU.
            dtype: Dtype for the cached tensors.
            ind: Index into the encoder's training chunk configuration, used only when
                ``stride`` is omitted.

        Returns:
            The cache dict. It is mutated in place by :meth:`forward_chunk`, so hold on
            to it and pass the same object every call. Keys:

            - ``start_idx`` (int): absolute frame offset consumed so far, used to keep
              the sinusoidal position encoding continuous across calls.
            - ``chunk_size`` (list[int]): ``[pad_left, stride, pad_right]``.
            - ``encoder_chunk_look_back`` (int): as above.
            - ``opt`` (list | None): per-layer attention key/value caches, one entry per
              encoder layer in ``encoders0 + encoders + tp_encoders``. Each entry is
              ``None`` or ``{"k": (batch, head, time, d_k), "v": (batch, head, time,
              d_k)}``. Stays ``None`` when ``encoder_chunk_look_back == 0``.
            - ``feats`` (Tensor): ``(batch, kept, input_size)`` overlap of already
              scaled and position-encoded frames prepended to the next chunk. Starts
              with ``pad_left`` zero frames and holds ``pad_left + pad_right`` frames
              afterwards. That initial length deviates from upstream funasr; see the
              note below.
            - ``tail_chunk`` (bool): set to ``True`` by the caller before the final
              call to flush the withheld lookahead frames.

        Raises:
            ValueError: If the geometry is invalid, or if ``stride`` is omitted while
                the encoder was built without ``chunk_size``.

        Note:
            When ``stride`` is omitted the geometry is derived from the training
            configuration at ``ind``: ``pad_left = self.pad_left[ind]``,
            ``stride = self.stride[ind]``,
            ``pad_right = self.chunk_size[ind] - stride - pad_left``.

        Note:
            ``encoder_chunk_look_back != 0`` requires ``pad_right >= 1`` because
            :meth:`MultiHeadedAttentionSANM.forward_chunk` trims the lookahead with
            ``k_h[:, :, :-pad_right, :]``, which would empty the cache at
            ``pad_right == 0``. Combining it with ``pad_left > 0`` also re-appends the
            left-context frames to that cache (upstream funasr behaviour), so prefer
            ``pad_left = 0`` whenever look-back is enabled.

        Note:
            ``feats`` is seeded with ``pad_left`` zero frames, where upstream funasr
            (``funasr/models/scama/model.py``, ``init_cache``) seeds
            ``pad_left + pad_right``. On the very first call nothing has been withheld
            yet, so upstream's extra ``pad_right`` frames stand for a lookahead that
            does not exist. They are not merely wasted compute: they enter
            self-attention and the FSMN convolution, so the first call emits ``stride``
            frames instead of ``stride - pad_right`` and the stream carries
            ``pad_right`` phantom output frames for the whole utterance, breaking the
            one-to-one input/output frame alignment CTC decoding depends on.
            ``tests/test_chunk_streaming_equivalence.py`` pins this. The raw first
            layer-0 windows cannot be compared directly, since upstream's is exactly
            ``pad_right`` frames longer; those surplus leading frames are exactly
            ``0.0``, and after realigning (dropping them) the two windows are
            bit-identical for every geometry tested. The resulting output corruption is
            confined to the first emitted chunk -- it peaks at ``1.514661`` for
            ``(pad_left, stride, pad_right) = (0, 10, 5)``, and every frame from index
            ``stride`` onward is bit-identical.
        """
        if stride is None:
            if self.overlap_chunk_cls is None:
                raise ValueError(
                    "stride must be given because this encoder was built without "
                    "chunk_size, so there is no training configuration to derive from"
                )
            self._check_chunk_ind(ind, "ind")
            if self.chunk_size[ind] <= 0:
                raise ValueError(
                    f"chunk_size[{ind}]={self.chunk_size[ind]} is the full-attention "
                    "sentinel and has no streaming geometry; pick another ind or pass "
                    "stride explicitly"
                )
            pad_left = self.pad_left[ind]
            stride = self.stride[ind]
            pad_right = self.chunk_size[ind] - stride - pad_left
        else:
            pad_left = 0 if pad_left is None else int(pad_left)
            stride = int(stride)
            pad_right = 0 if pad_right is None else int(pad_right)

        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if pad_left < 0 or pad_right < 0:
            raise ValueError(
                f"pad_left and pad_right must be >= 0, got {pad_left} and {pad_right}"
            )
        if encoder_chunk_look_back != 0 and pad_right < 1:
            raise ValueError(
                "encoder_chunk_look_back != 0 requires pad_right >= 1; the attention "
                "cache is built by dropping the last pad_right frames and would be "
                "empty otherwise"
            )

        return {
            "start_idx": 0,
            "chunk_size": [pad_left, stride, pad_right],
            "encoder_chunk_look_back": encoder_chunk_look_back,
            "opt": None,
            "feats": torch.zeros(
                (batch_size, pad_left, self._input_size), dtype=dtype, device=device
            ),
            "tail_chunk": False,
        }

    def _add_overlap_chunk(self, feats: torch.Tensor, cache: dict) -> torch.Tensor:
        """Prepend the cached overlap to the incoming frames and refresh the cache.

        The SANM memory block has no streaming cache of its own; its context is
        supplied by overlapping the chunks instead, which is what this does.

        Args:
            feats: Scaled and position-encoded new frames (batch, time, input_size).
            cache: Cache dict from :meth:`init_chunk_cache`.

        Returns:
            The concatenated window (batch, kept + time, input_size).
        """
        cached = cache.get("feats")
        if cached is None:
            return feats
        overlap_feats = torch.cat(
            (cached.to(device=feats.device, dtype=feats.dtype), feats), dim=1
        )
        keep = cache["chunk_size"][0] + cache["chunk_size"][2]
        cache["feats"] = overlap_feats[:, max(0, overlap_feats.size(1) - keep) :, :]
        return overlap_feats

    def forward_chunk(
        self,
        xs_pad: torch.Tensor,
        cache: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Encode one streaming chunk.

        Args:
            xs_pad: New frames for this step, ``(batch, stride, input_size)``. The last
                real chunk may be shorter. Not modified in place, unlike
                :meth:`forward`. Ignored when ``cache["tail_chunk"]`` is set.
            cache: Cache dict from :meth:`init_chunk_cache`, mutated in place. When
                ``None`` a default cache is built from the training configuration.
            **kwargs: ``trim`` (bool, default ``True``) returns only the frames that
                are final at this step. Pass ``trim=False`` for the raw
                ``pad_left + time + pad_right`` window, matching funasr's
                ``SANMEncoderChunkOpt.forward_chunk``.

        Returns:
            ``(xs_pad, cache)``. With ``trim=True`` the output frames line up one to
            one with the input frames, delayed by ``pad_right``: the first call returns
            ``time - pad_right`` frames, later calls return ``time`` frames, and the
            ``tail_chunk`` call flushes the final ``pad_right`` frames. Concatenating
            every call's output reproduces the whole utterance in order.

        Note:
            Run this under ``torch.no_grad()`` with the module in ``eval()`` mode; the
            per-layer caches assume a single monotonic pass over one stream.

        Note:
            The caller owns the SenseVoice prompt frames. Prepend the four
            ``[language, event, emo, textnorm]`` frames to the first chunk so they take
            absolute positions 1-4, exactly as ``SenseVoiceSmall.encode`` does.
        """
        if cache is None:
            cache = self.init_chunk_cache(
                batch_size=xs_pad.size(0), device=xs_pad.device, dtype=xs_pad.dtype
            )
        pad_left, _, pad_right = cache["chunk_size"]
        is_tail = cache.get("tail_chunk", False)

        if is_tail:
            # The cached overlap is already scaled and position-encoded; re-running it
            # is what flushes the frames whose lookahead never arrived.
            xs_pad = cache["feats"].to(device=xs_pad.device, dtype=xs_pad.dtype)
        else:
            xs_pad = xs_pad * self.output_size() ** 0.5
            xs_pad = self.chunk_embed(xs_pad, cache)
            xs_pad = self._add_overlap_chunk(xs_pad, cache)

        if cache["opt"] is None:
            cache_layer_num = (
                len(self.encoders0) + len(self.encoders) + len(self.tp_encoders)
            )
            new_cache = [None] * cache_layer_num
        else:
            new_cache = cache["opt"]

        chunk_size, look_back = cache["chunk_size"], cache["encoder_chunk_look_back"]
        offset = 0
        for layer_idx, encoder_layer in enumerate(self.encoders0):
            idx = offset + layer_idx
            xs_pad, new_cache[idx] = encoder_layer.forward_chunk(
                xs_pad, new_cache[idx], chunk_size, look_back
            )

        offset += len(self.encoders0)
        for layer_idx, encoder_layer in enumerate(self.encoders):
            idx = offset + layer_idx
            xs_pad, new_cache[idx] = encoder_layer.forward_chunk(
                xs_pad, new_cache[idx], chunk_size, look_back
            )

        xs_pad = self.after_norm(xs_pad)

        offset += len(self.encoders)
        for layer_idx, encoder_layer in enumerate(self.tp_encoders):
            idx = offset + layer_idx
            xs_pad, new_cache[idx] = encoder_layer.forward_chunk(
                xs_pad, new_cache[idx], chunk_size, look_back
            )

        xs_pad = self.tp_norm(xs_pad)

        if look_back > 0 or look_back == -1:
            cache["opt"] = new_cache

        if kwargs.get("trim", True):
            end = xs_pad.size(1) if is_tail else xs_pad.size(1) - pad_right
            xs_pad = xs_pad[:, pad_left : max(0, end), :]
        return xs_pad, cache


@tables.register("model_classes", "SenseVoiceSmall")
class SenseVoiceSmall(nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        specaug: str = None,
        specaug_conf: dict = None,
        normalize: str = None,
        normalize_conf: dict = None,
        encoder: str = None,
        encoder_conf: dict = None,
        ctc_conf: dict = None,
        input_size: int = 80,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        length_normalized_loss: bool = False,
        **kwargs,
    ):

        super().__init__()

        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)
        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()

        if ctc_conf is None:
            ctc_conf = {}
        ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)

        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.error_calculator = None

        self.ctc = ctc

        self.length_normalized_loss = length_normalized_loss
        self.encoder_output_size = encoder_output_size

        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.textnorm_int_dict = {25016: 14, 25017: 15}
        self.embed = torch.nn.Embedding(7 + len(self.lid_dict) + len(self.textnorm_dict), input_size)
        self.emo_dict = {"unk": 25009, "happy": 25001, "sad": 25002, "angry": 25003, "neutral": 25004}
        
        self.criterion_att = LabelSmoothingLoss(
            size=self.vocab_size,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
        )
    
    @staticmethod
    def from_pretrained(model:str=None, **kwargs):
        from funasr import AutoModel
        model, kwargs = AutoModel.build_model(model=model, trust_remote_code=True, **kwargs)
        
        return model, kwargs

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        # import pdb;
        # pdb.set_trace()
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, text)

        loss_ctc, cer_ctc = None, None
        loss_rich, acc_rich = None, None
        stats = dict()

        loss_ctc, cer_ctc = self._calc_ctc_loss(
            encoder_out[:, 4:, :], encoder_out_lens - 4, text[:, 4:], text_lengths - 4
        )

        loss_rich, acc_rich = self._calc_rich_ce_loss(
            encoder_out[:, :4, :], text[:, :4]
        )

        loss = loss_ctc + loss_rich
        # Collect total loss stats
        stats["loss_ctc"] = torch.clone(loss_ctc.detach()) if loss_ctc is not None else None
        stats["loss_rich"] = torch.clone(loss_rich.detach()) if loss_rich is not None else None
        stats["loss"] = torch.clone(loss.detach()) if loss is not None else None
        stats["acc_rich"] = acc_rich

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        **kwargs,
    ):
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """

        # Data augmentation
        if self.specaug is not None and self.training:
            speech, speech_lengths = self.specaug(speech, speech_lengths)

        # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        if self.normalize is not None:
            speech, speech_lengths = self.normalize(speech, speech_lengths)


        lids = torch.LongTensor([[self.lid_int_dict[int(lid)] if torch.rand(1) > 0.2 and int(lid) in self.lid_int_dict else 0 ] for lid in text[:, 0]]).to(speech.device)
        language_query = self.embed(lids)
        
        styles = torch.LongTensor([[self.textnorm_int_dict[int(style)]] for style in text[:, 3]]).to(speech.device)
        style_query = self.embed(styles)
        speech = torch.cat((style_query, speech), dim=1)
        speech_lengths += 1

        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(speech.size(0), 1, 1)
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)

        return encoder_out, encoder_out_lens

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rich_ce_loss(
        self,
        encoder_out: torch.Tensor,
        ys_pad: torch.Tensor,
    ):
        decoder_out = self.ctc.ctc_lo(encoder_out)
        # 2. Compute attention loss
        loss_rich = self.criterion_att(decoder_out, ys_pad.contiguous())
        acc_rich = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_pad.contiguous(),
            ignore_label=self.ignore_id,
        )

        return loss_rich, acc_rich


    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = ["wav_file_tmp_name"],
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):


        meta_data = {}
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = (
                speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        language = kwargs.get("language", "auto")
        language_query = self.embed(
            torch.LongTensor(
                [[self.lid_dict[language] if language in self.lid_dict else 0]]
            ).to(speech.device)
        ).repeat(speech.size(0), 1, 1)
        
        use_itn = kwargs.get("use_itn", False)
        output_timestamp = kwargs.get("output_timestamp", False)

        textnorm = kwargs.get("text_norm", None)
        if textnorm is None:
            textnorm = "withitn" if use_itn else "woitn"
        textnorm_query = self.embed(
            torch.LongTensor([[self.textnorm_dict[textnorm]]]).to(speech.device)
        ).repeat(speech.size(0), 1, 1)
        speech = torch.cat((textnorm_query, speech), dim=1)
        speech_lengths += 1

        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
            speech.size(0), 1, 1
        )
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        # Encoder
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # c. Passed the encoder result and the beam search
        ctc_logits = self.ctc.log_softmax(encoder_out)
        if kwargs.get("ban_emo_unk", False):
            ctc_logits[:, :, self.emo_dict["unk"]] = -float("inf")

        results = []
        b, n, d = encoder_out.size()
        if isinstance(key[0], (list, tuple)):
            key = key[0]
        if len(key) < b:
            key = key * b
        for i in range(b):
            x = ctc_logits[i, : encoder_out_lens[i].item(), :]
            yseq = x.argmax(dim=-1)
            yseq = torch.unique_consecutive(yseq, dim=-1)

            ibest_writer = None
            if kwargs.get("output_dir") is not None:
                if not hasattr(self, "writer"):
                    self.writer = DatadirWriter(kwargs.get("output_dir"))
                ibest_writer = self.writer[f"1best_recog"]

            mask = yseq != self.blank_id
            token_int = yseq[mask].tolist()

            # Change integer-ids to tokens
            text = tokenizer.decode(token_int)
            if ibest_writer is not None:
                ibest_writer["text"][key[i]] = text

            if output_timestamp:
                from itertools import groupby
                timestamp = []
                tokens = tokenizer.text2tokens(text)[4:]

                logits_speech = self.ctc.softmax(encoder_out)[i, 4:encoder_out_lens[i].item(), :]

                pred = logits_speech.argmax(-1).cpu()
                logits_speech[pred==self.blank_id, self.blank_id] = 0

                align = ctc_forced_align(
                    logits_speech.unsqueeze(0).float(),
                    torch.Tensor(token_int[4:]).unsqueeze(0).long().to(logits_speech.device),
                    (encoder_out_lens-4).long()[i],
                    torch.tensor(len(token_int)-4).unsqueeze(0).long().to(logits_speech.device),
                    ignore_id=self.ignore_id,
                )

                pred = groupby(align[0, :encoder_out_lens[0]])
                _start = 0
                token_id = 0
                ts_max = encoder_out_lens[i] - 4
                for pred_token, pred_frame in pred:
                    _end = _start + len(list(pred_frame))
                    if pred_token != 0 and token_id < len(tokens):
                        ts_left = max((_start*60-30)/1000, 0)
                        ts_right = min((_end*60-30)/1000, (ts_max*60-30)/1000)
                        timestamp.append([tokens[token_id], ts_left, ts_right])
                        token_id += 1
                    _start = _end

                result_i = {"key": key[i], "text": text, "timestamp": timestamp}
                results.append(result_i)
            else:
                result_i = {"key": key[i], "text": text}
                results.append(result_i)
        return results, meta_data

    def export(self, **kwargs):
        from export_meta import export_rebuild_model

        if "max_seq_len" not in kwargs:
            kwargs["max_seq_len"] = 512
        models = export_rebuild_model(model=self, **kwargs)
        return models
