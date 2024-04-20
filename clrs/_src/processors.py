# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""JAX implementation of baseline processor networks."""

import abc
from typing import Any, Callable, List, Optional, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from clrs._src.global_config import latents_config


_Array = chex.Array
_Fn = Callable[..., Any]
BIG_NUMBER = 1e6
PROCESSOR_TAG = 'clrs_processor'


class Processor(hk.Module):
  """Processor abstract base class."""

  def __init__(self, name: str):
    if not name.endswith(PROCESSOR_TAG):
      name = name + '_' + PROCESSOR_TAG
    super().__init__(name=name)

  @abc.abstractmethod
  def __call__(
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **kwargs,
  ) -> Tuple[_Array, Optional[_Array]]:
    """Processor inference step.

    Args:
      node_fts: Node features.
      edge_fts: Edge features.
      graph_fts: Graph features.
      adj_mat: Graph adjacency matrix.
      hidden: Hidden features.
      **kwargs: Extra kwargs.

    Returns:
      Output of processor inference step as a 2-tuple of (node, edge)
      embeddings. The edge embeddings can be None.
    """
    pass

  @property
  def inf_bias(self):
    return False

  @property
  def inf_bias_edge(self):
    return False


class GAT(Processor):
  """Graph Attention Network (Velickovic et al., ICLR 2018)."""

  def __init__(
      self,
      out_size: int,
      nb_heads: int,
      activation: Optional[_Fn] = jax.nn.relu,
      residual: bool = True,
      use_ln: bool = False,
      name: str = 'gat_aggr',
  ):
    super().__init__(name=name)
    self.out_size = out_size
    self.nb_heads = nb_heads
    if out_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the width!')
    self.head_size = out_size // nb_heads
    self.activation = activation
    self.residual = residual
    self.use_ln = use_ln

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """GAT inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    m = hk.Linear(self.out_size)
    skip = hk.Linear(self.out_size)

    bias_mat = (adj_mat - 1.0) * 1e9
    bias_mat = jnp.tile(bias_mat[..., None],
                        (1, 1, 1, self.nb_heads))     # [B, N, N, H]
    bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

    a_1 = hk.Linear(self.nb_heads)
    a_2 = hk.Linear(self.nb_heads)
    a_e = hk.Linear(self.nb_heads)
    a_g = hk.Linear(self.nb_heads)

    values = m(z)                                      # [B, N, H*F]
    values = jnp.reshape(
        values,
        values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, H, F]
    values = jnp.transpose(values, (0, 2, 1, 3))              # [B, H, N, F]

    att_1 = jnp.expand_dims(a_1(z), axis=-1)
    att_2 = jnp.expand_dims(a_2(z), axis=-1)
    att_e = a_e(edge_fts)
    att_g = jnp.expand_dims(a_g(graph_fts), axis=-1)

    logits = (
        jnp.transpose(att_1, (0, 2, 1, 3)) +  # + [B, H, N, 1]
        jnp.transpose(att_2, (0, 2, 3, 1)) +  # + [B, H, 1, N]
        jnp.transpose(att_e, (0, 3, 1, 2)) +  # + [B, H, N, N]
        jnp.expand_dims(att_g, axis=-1)       # + [B, H, 1, 1]
    )                                         # = [B, H, N, N]
    coefs = jax.nn.softmax(jax.nn.leaky_relu(logits) + bias_mat, axis=-1)
    ret = jnp.matmul(coefs, values)  # [B, H, N, F]
    ret = jnp.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]
    ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, H*F]

    if self.residual:
      ret += skip(z)

    if self.activation is not None:
      ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    return ret, None  # pytype: disable=bad-return-type  # numpy-scalars


class GATFull(GAT):
  """Graph Attention Network with full adjacency matrix."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class GATv2(Processor):
  """Graph Attention Network v2 (Brody et al., ICLR 2022)."""

  def __init__(
      self,
      out_size: int,
      nb_heads: int,
      mid_size: Optional[int] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      residual: bool = True,
      use_ln: bool = False,
      name: str = 'gatv2_aggr',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.nb_heads = nb_heads
    if out_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the width!')
    self.head_size = out_size // nb_heads
    if self.mid_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the message!')
    self.mid_head_size = self.mid_size // nb_heads
    self.activation = activation
    self.residual = residual
    self.use_ln = use_ln

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """GATv2 inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    m = hk.Linear(self.out_size)
    skip = hk.Linear(self.out_size)

    bias_mat = (adj_mat - 1.0) * 1e9
    bias_mat = jnp.tile(bias_mat[..., None],
                        (1, 1, 1, self.nb_heads))     # [B, N, N, H]
    bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

    w_1 = hk.Linear(self.mid_size)
    w_2 = hk.Linear(self.mid_size)
    w_e = hk.Linear(self.mid_size)
    w_g = hk.Linear(self.mid_size)

    a_heads = []
    for _ in range(self.nb_heads):
      a_heads.append(hk.Linear(1))

    values = m(z)                                      # [B, N, H*F]
    values = jnp.reshape(
        values,
        values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, H, F]
    values = jnp.transpose(values, (0, 2, 1, 3))              # [B, H, N, F]

    pre_att_1 = w_1(z)
    pre_att_2 = w_2(z)
    pre_att_e = w_e(edge_fts)
    pre_att_g = w_g(graph_fts)

    pre_att = (
        jnp.expand_dims(pre_att_1, axis=1) +     # + [B, 1, N, H*F]
        jnp.expand_dims(pre_att_2, axis=2) +     # + [B, N, 1, H*F]
        pre_att_e +                              # + [B, N, N, H*F]
        jnp.expand_dims(pre_att_g, axis=(1, 2))  # + [B, 1, 1, H*F]
    )                                            # = [B, N, N, H*F]

    pre_att = jnp.reshape(
        pre_att,
        pre_att.shape[:-1] + (self.nb_heads, self.mid_head_size)
    )  # [B, N, N, H, F]

    pre_att = jnp.transpose(pre_att, (0, 3, 1, 2, 4))  # [B, H, N, N, F]

    # This part is not very efficient, but we agree to keep it this way to
    # enhance readability, assuming `nb_heads` will not be large.
    logit_heads = []
    for head in range(self.nb_heads):
      logit_heads.append(
          jnp.squeeze(
              a_heads[head](jax.nn.leaky_relu(pre_att[:, head])),
              axis=-1)
      )  # [B, N, N]

    logits = jnp.stack(logit_heads, axis=1)  # [B, H, N, N]

    coefs = jax.nn.softmax(logits + bias_mat, axis=-1)
    ret = jnp.matmul(coefs, values)  # [B, H, N, F]
    ret = jnp.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]
    ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, H*F]

    if self.residual:
      ret += skip(z)

    if self.activation is not None:
      ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    return ret, None  # pytype: disable=bad-return-type  # numpy-scalars


class GATv2FullD2(GATv2):
  """Graph Attention Network v2 with full adjacency matrix and D2 symmetry."""

  def d2_forward(self,
                 node_fts: List[_Array],
                 edge_fts: List[_Array],
                 graph_fts: List[_Array],
                 adj_mat: _Array,
                 hidden: _Array,
                 **unused_kwargs) -> List[_Array]:
    num_d2_actions = 4

    d2_inverses = [
        0, 1, 2, 3  # All members of D_2 are self-inverses!
    ]

    d2_multiply = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]

    assert len(node_fts) == num_d2_actions
    assert len(edge_fts) == num_d2_actions
    assert len(graph_fts) == num_d2_actions

    ret_nodes = []
    adj_mat = jnp.ones_like(adj_mat)

    for g in range(num_d2_actions):
      emb_values = []
      for h in range(num_d2_actions):
        gh = d2_multiply[d2_inverses[g]][h]
        node_features = jnp.concatenate(
            (node_fts[g], node_fts[gh]),
            axis=-1)
        edge_features = jnp.concatenate(
            (edge_fts[g], edge_fts[gh]),
            axis=-1)
        graph_features = jnp.concatenate(
            (graph_fts[g], graph_fts[gh]),
            axis=-1)
        cell_embedding = super().__call__(
            node_fts=node_features,
            edge_fts=edge_features,
            graph_fts=graph_features,
            adj_mat=adj_mat,
            hidden=hidden
        )
        emb_values.append(cell_embedding[0])
      ret_nodes.append(
          jnp.mean(jnp.stack(emb_values, axis=0), axis=0)
      )

    return ret_nodes


class GATv2Full(GATv2):
  """Graph Attention Network v2 with full adjacency matrix."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


def get_triplet_msgs(z, edge_fts, graph_fts, nb_triplet_fts):
  """Triplet messages, as done by Dudzik and Velickovic (2022)."""
  t_1 = hk.Linear(nb_triplet_fts)
  t_2 = hk.Linear(nb_triplet_fts)
  t_3 = hk.Linear(nb_triplet_fts)
  t_e_1 = hk.Linear(nb_triplet_fts)
  t_e_2 = hk.Linear(nb_triplet_fts)
  t_e_3 = hk.Linear(nb_triplet_fts)
  t_g = hk.Linear(nb_triplet_fts)

  tri_1 = t_1(z)
  tri_2 = t_2(z)
  tri_3 = t_3(z)
  tri_e_1 = t_e_1(edge_fts)
  tri_e_2 = t_e_2(edge_fts)
  tri_e_3 = t_e_3(edge_fts)
  tri_g = t_g(graph_fts)

  return (
      jnp.expand_dims(tri_1, axis=(2, 3))    +  #   (B, N, 1, 1, H)
      jnp.expand_dims(tri_2, axis=(1, 3))    +  # + (B, 1, N, 1, H)
      jnp.expand_dims(tri_3, axis=(1, 2))    +  # + (B, 1, 1, N, H)
      jnp.expand_dims(tri_e_1, axis=3)       +  # + (B, N, N, 1, H)
      jnp.expand_dims(tri_e_2, axis=2)       +  # + (B, N, 1, N, H)
      jnp.expand_dims(tri_e_3, axis=1)       +  # + (B, 1, N, N, H)
      jnp.expand_dims(tri_g, axis=(1, 2, 3))    # + (B, 1, 1, 1, H)
  )                                             # = (B, N, N, N, H)

# class HierarchicalGraphProcessor(Processor):
#   """Hierarchical Graph Processor."""

#   def __init__(self,
#                out_size: int,
#                nb_hgp_levels: int,
#                nb_heads: int,
#                use_skip_connection: bool,
#                reducer: str = 'max',
#                activation_fn: Optional[_Fn] = jax.nn.relu,
#                dropout_rate: Optional[float] = 0.0,
#                use_ln: bool = False,
#                name: str = 'hierarchical_graph_processor'):
#     super().__init__(name=name)
#     self.out_size = out_size
#     self.nb_hgp_levels = nb_hgp_levels
#     self.reducer = reducer
#     self.activation_fn = activation_fn
#     self.nb_heads = nb_heads
#     self.dropout_rate = dropout_rate
#     self.use_skip_connection = use_skip_connection

#     self.use_ln = use_ln

#   def __call__(self,
#                node_fts: _Array,
#                edge_fts: _Array,
#                graph_fts: _Array,
#                adj_mat: _Array,
#                hidden: _Array,
#                **unused_kwargs):
#     """Hierarchical graph processor inference step."""

#     b, n, _ = node_fts.shape
#     assert edge_fts.shape[:-1] == (b, n, n)
#     assert graph_fts.shape[:-1] == (b,)
#     assert adj_mat.shape == (b, n, n)

#     node_fts = jnp.concatenate([node_fts, hidden], axis=-1)

#     # Perform hierarchical message passing
#     for level in range(self.nb_hgp_levels):
#       node_fts = self.update_node_fts(level, node_fts, edge_fts, adj_mat)

#     # Perform final update to get output node features
#     output_node_fts = hk.Linear(self.out_size)(node_fts)
#     if self.activation_fn is not None:
#       output_node_fts = self.activation_fn(output_node_fts)

#     if self.use_ln:
#       output_node_fts = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(output_node_fts)

#     return output_node_fts, None, None  # pytype: disable=bad-return-type  # numpy-scalars
  
#   def compute_attention(self, query, key, value, edge_fts, graph_fts, adj_mat):
#     """Compute attention scores with graph-level features and adjacency masking."""
#     # Compute attention scores based on node features
#     node_query = hk.Linear(self.out_size)(query)
#     node_key = hk.Linear(self.out_size)(key)
#     node_attention_scores = jnp.einsum('bhid,bhjd->bhij', node_query, node_key)

#     # Compute attention scores based on edge features
#     edge_query = hk.Linear(self.out_size)(query)
#     edge_key = hk.Linear(self.out_size)(edge_fts)
#     edge_attention_scores = jnp.einsum('bhid,bhijd->bhij', edge_query, edge_key)

#     # Compute attention scores based on graph-level features
#     graph_query = hk.Linear(self.out_size)(query)
#     graph_key = hk.Linear(self.out_size)(graph_fts)
#     graph_attention_scores = jnp.einsum('bhid,bhd->bhij', graph_query, graph_key)

#     # Combine node, edge, and graph attention scores
#     attention_scores = node_attention_scores + edge_attention_scores + graph_attention_scores

#     # Mask attention scores based on adjacency matrix
#     mask = -1e9 * (1.0 - adj_mat)
#     attention_scores = jnp.where(adj_mat, attention_scores, mask)
#     attention_scores = jax.nn.softmax(attention_scores, axis=-1)

#     # Compute attended values
#     attended_values = jnp.einsum('bhij,bhjd->bhid', attention_scores, value)

#     return attended_values

#   # def compute_attention(self, query, key, value, edge_fts, adj_mat):
#   #   """Compute attention scores with adjacency masking."""
#   #   # Compute attention scores based on node features
#   #   node_attention_scores = jnp.dot(query, key.transpose(0, 2, 1))

#   #   # Compute attention scores based on edge features
#   #   edge_query = hk.Linear(self.out_size)(query)
#   #   edge_key = hk.Linear(self.out_size)(edge_fts)
#   #   edge_attention_scores = jnp.einsum('bhid,bhijd->bhij', edge_query, edge_key)

#   #   # Combine node and edge attention scores
#   #   attention_scores = node_attention_scores + edge_attention_scores

#   #   # Mask attention scores based on adjacency matrix
#   #   mask = -1e9 * (1.0 - adj_mat)
#   #   attention_scores = jnp.where(adj_mat, attention_scores, mask)
#   #   attention_scores = jax.nn.softmax(attention_scores, axis=-1)
#   #   attended_values = jnp.einsum('bhij,bhjd->bhid', attention_scores, value)
#   #   return attended_values

#   def aggregate_level(self, level_node_fts, level_edge_fts, level_adj_mat, b, n):
#     """Aggregate information at a single level."""
#     if self.nb_heads > 0:
#       # Implement multi-head attention
#       head_size = self.out_size // self.nb_heads
#       query = hk.Linear(self.out_size)(level_node_fts)
#       key = hk.Linear(self.out_size)(level_node_fts)
#       level_node_fts = jnp.expand_dims(level_node_fts, axis=2)
#       level_node_fts = jnp.repeat(level_node_fts, level_edge_fts.shape[2], axis=2)
#       value = jnp.concatenate([level_node_fts, level_edge_fts], axis=-1)
#       value = hk.Linear(self.out_size)(value)

#       query = jnp.reshape(query, (b, n, self.nb_heads, head_size))
#       key = jnp.reshape(key, (b, n, self.nb_heads, head_size))
#       value = jnp.reshape(value, (b, n, self.nb_heads, head_size))

#       query = jnp.transpose(query, (0, 2, 1, 3))  # (b, h, n, d)
#       key = jnp.transpose(key, (0, 2, 1, 3))  # (b, h, n, d)
#       value = jnp.transpose(value, (0, 2, 1, 3))  # (b, h, n, d)

#       attended_values = jax.vmap(self.compute_attention, in_axes=(0, 0, 0, None))(
#           query, key, value, level_adj_mat)  # (b, h, n, d)

#       attended_values = jnp.transpose(attended_values, (0, 2, 1, 3))  # (b, n, h, d)
#       attended_values = jnp.reshape(attended_values, (b, n, self.out_size))
#       if self.reducer == 'max':
#         aggregated_fts = jnp.max(attended_values, axis=1)
#       elif self.reducer == 'sum':
#         aggregated_fts = jnp.sum(attended_values, axis=1)
#       elif self.reducer == 'mean':
#         aggregated_fts = jnp.mean(attended_values, axis=1)
#       else:
#         raise ValueError(f"Unsupported reducer: {self.reducer}")
        
#     else:
#       level_edge_fts = jnp.max(level_node_fts[:, None, :, :] +
#                               level_node_fts[:, :, None, :] +
#                               level_edge_fts, axis=-1, keepdims=True)
#       level_edge_fts = level_edge_fts * level_adj_mat[..., None]
#       if self.reducer == 'max':
#         aggregated_fts = jnp.max(level_edge_fts, axis=-2)
#       elif self.reducer == 'sum':
#         aggregated_fts = jnp.sum(level_edge_fts, axis=-2)
#       elif self.reducer == 'mean':
#         aggregated_fts = jnp.mean(level_edge_fts, axis=-2)
#       else:
#         raise ValueError(f"Unsupported reducer: {self.reducer}")
    
#     return aggregated_fts

#   def update_node_fts(self, level, node_fts, edge_fts, adj_mat):
#     """Update node features at a single level."""
#     level_node_fts = hk.Linear(self.out_size, name=f"level_{level}_linear")(node_fts)
#     # level_node_fts = hk.Linear(self.out_size)(node_fts)
#     if self.activation_fn is not None:
#       level_node_fts = self.activation_fn(level_node_fts)
#     b, n, _ = node_fts.shape
#     aggregated_fts = self.aggregate_level(level_node_fts, edge_fts, adj_mat, b, n)
#     if self.use_skip_connection:
#       aggregated_fts += node_fts
#     hk.dropout(hk.next_rng_key(), self.dropout_rate, node_fts)
#     return aggregated_fts
  
class HierarchicalGraphProcessor(Processor):
  """Hierarchical Graph Processor."""

  def __init__(self,
               out_size: int,
               nb_hgp_levels: int,
               nb_heads: int,
               use_skip_connection: bool,
               reducer: str = 'max',
               activation_fn: Optional[_Fn] = jax.nn.relu,
               dropout_rate: Optional[float] = 0.0,
               use_ln: bool = False,
               name: str = 'hierarchical_graph_processor'):
    super().__init__(name=name)
    self.out_size = out_size
    self.nb_hgp_levels = nb_hgp_levels
    self.reducer = reducer
    self.activation_fn = activation_fn
    self.nb_heads = nb_heads
    self.dropout_rate = dropout_rate
    self.use_skip_connection = use_skip_connection
    self.use_ln = use_ln

  def __call__(self,
               node_fts: _Array,
               edge_fts: _Array,
               graph_fts: _Array,
               adj_mat: _Array,
               hidden: _Array,
               **unused_kwargs):
    """Hierarchical graph processor inference step."""
    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    node_fts = jnp.concatenate([node_fts, hidden], axis=-1)

    # Perform hierarchical message passing
    for level in range(self.nb_hgp_levels):
      node_fts = self.update_node_fts(level, node_fts, edge_fts, graph_fts, adj_mat)

    # Perform final update to get output node features
    output_node_fts = hk.Linear(self.out_size)(node_fts)
    if self.activation_fn is not None:
      output_node_fts = self.activation_fn(output_node_fts)
    if self.use_ln:
      output_node_fts = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(output_node_fts)

    return output_node_fts, None, None
  
  def compute_attention(self, query, key, value, edge_fts, graph_fts, adj_mat):
    """Compute attention scores with graph-level features and adjacency masking."""

    # DEBUGGING PRINT SHAPES
    print("Inside compute_attention:")
    print("query shape: ", query.shape)
    print("key shape: ", key.shape)
    print("value shape: ", value.shape)
    print("edge_fts shape: ", edge_fts.shape)
    print("graph_fts shape: ", graph_fts.shape)
    print("adj_mat shape: ", adj_mat.shape)


    # 1. Linear Transformations for Node Attention
    node_query = hk.Linear(self.out_size)(query)  # (b, h, n, d)
    node_key = hk.Linear(self.out_size)(key)    # (b, h, n, d)

    # DEBUGGING PRINT SHAPES
    print("After Linear Transformations for Node Attention:")
    print("node_query shape: ", node_query.shape)
    print("node_key shape: ", node_key.shape)

    # 2. Node-to-Node Attention Scores
    node_attention_scores = jnp.einsum('hnd,hmd->hnm', node_query, node_key)  # (b, h, n, n)

    # DEBUGGING PRINT SHAPES
    print("After Node-to-Node Attention Scores:")
    print("node_attention_scores shape: ", node_attention_scores.shape)

    # 3. Linear Transformations for Edge Attention
    edge_query = hk.Linear(self.out_size)(query)  # (b, h, n, d)
    edge_key = hk.Linear(self.out_size)(edge_fts)  # (b, h, n, n, d)

    # DEBUGGING PRINT SHAPES
    print("After Linear Transformations for Edge Attention:")
    print("edge_query shape: ", edge_query.shape)
    print("edge_key shape: ", edge_key.shape)

    # 4. Edge Attention Scores
    edge_attention_scores = jnp.einsum('hnd,hnmd->hnm', edge_query, edge_key)  # (b, h, n, n)

    # DEBUGGING PRINT SHAPES
    print("After Edge Attention Scores:")
    print("edge_attention_scores shape: ", edge_attention_scores.shape)

    # # 5. Choose Approach for Graph Attention (Global Context or Node-Graph)
    # if self.use_global_context:
    #   # 5a. Global Context Vector Attention
    #   graph_query = hk.Linear(self.out_size)(query)  # (b, h, n, d)
    #   context_vector = hk.get_parameter("context_vector", 
    #                                     shape=(1, self.nb_heads, 1, self.head_size), 
    #                                     init=hk.initializers.RandomNormal())  # (1, h, 1, d)
    #   graph_attention_scores = jnp.einsum('hnd,hd->hn', graph_query, context_vector)  # (b, h, n)
    #   graph_attention_scores = jnp.expand_dims(graph_attention_scores, axis=-1)  # (b, h, n, 1)

    # else:
    # 5b. Node-Graph Attention
    graph_query = hk.Linear(self.out_size)(query)  # (b, h, n, d)
    graph_key = hk.Linear(self.out_size)(graph_fts)  # (b, h, d)
    graph_key = jnp.expand_dims(jnp.expand_dims(graph_key, axis=2), axis=2)  # (b, h, 1, 1, d)
    graph_attention_scores = jnp.einsum('hnd,hd->hn', graph_query, graph_key)  # (b, h, n, 1)

    # DEBUGGING PRINT SHAPES
    print("After Linear Transformations for Graph Attention:")
    print("graph_query shape: ", graph_query.shape)
    print("graph_key shape: ", graph_key.shape)
    print("graph_attention_scores shape: ", graph_attention_scores.shape)
    

    # 6. Combine Attention Scores
    attention_scores = node_attention_scores + edge_attention_scores + graph_attention_scores

    # 7. Masking based on Adjacency Matrix
    mask = -1e9 * (1.0 - adj_mat[0])  # Assuming adj_mat has shape (b, n, n)
    attention_scores = jnp.where(adj_mat[0], attention_scores, mask)

    # 8. Softmax for Attention Coefficients
    attention_coefs = jax.nn.softmax(attention_scores, axis=-1)  # (b, h, n, n)

    # 9. Compute Attended Values
    attended_values = jnp.einsum('bhnm,bhmd->bhnd', attention_coefs, value)

    return attended_values
  
  def aggregate_level(self, level_node_fts, level_edge_fts, level_graph_fts, level_adj_mat, b, n):
    """Aggregate information at a single level."""
    if self.nb_heads > 0:
      # Implement multi-head attention
      head_size = self.out_size // self.nb_heads
      query = hk.Linear(self.out_size)(level_node_fts)
      key = hk.Linear(self.out_size)(level_node_fts)
      value = hk.Linear(self.out_size)(level_node_fts)

      # DEBUGGING PRINT SHAPES
      print("Inside aggregate_level:")
      print("level_node_fts shape: ", level_node_fts.shape)
      print("level_edge_fts shape: ", level_edge_fts.shape)
      print("level_graph_fts shape: ", level_graph_fts.shape)
      print("level_adj_mat shape: ", level_adj_mat.shape)
      print("query shape: ", query.shape)
      print("key shape: ", key.shape)
      print("value shape: ", value.shape)

      # Reshape for multi-head attention
      query = jnp.reshape(query, (b, n, self.nb_heads, head_size))
      key = jnp.reshape(key, (b, n, self.nb_heads, head_size))
      value = jnp.reshape(value, (b, n, self.nb_heads, head_size))
      query = jnp.transpose(query, (0, 2, 1, 3))  # (b, h, n, d)
      key = jnp.transpose(key, (0, 2, 1, 3))  # (b, h, n, d)
      value = jnp.transpose(value, (0, 2, 1, 3))  # (b, h, n, d)

      # DEBUGGING PRINT SHAPES
      print("After reshaping for multi-head attention:")
      print("query shape: ", query.shape)
      print("key shape: ", key.shape)
      print("value shape: ", value.shape)

      # Reshape edge features for multi-head attention
      level_edge_fts = jnp.reshape(level_edge_fts, (b, n, n, self.nb_heads, head_size))
      level_edge_fts = jnp.transpose(level_edge_fts, (0, 3, 1, 2, 4))  # (b, h, n, n, d)

      # DEBUGGING PRINT SHAPES
      print("After reshaping edge features for multi-head attention:")
      print("level_edge_fts shape: ", level_edge_fts.shape)


      # # Adjust graph features based on chosen approach
      # if self.use_global_context:
      #   # No need to pass graph features for global context
      #   graph_fts = None 
      # else:
      #   # Reshape graph features for node-graph attention 
      graph_fts = jnp.reshape(level_graph_fts, (b, self.nb_heads, 1, 1, self.out_size))

      # DEBUGGING PRINT SHAPES
      print("After reshaping graph features:")
      print("graph_fts shape: ", graph_fts.shape)

      # Apply compute_attention with jax.vmap over the batch dimension
      attended_values = jax.vmap(self.compute_attention, in_axes=(0, 0, 0, 0, 0, None))(
        query, key, value, level_edge_fts, graph_fts, level_adj_mat)

      # Compute attention and aggregate
      attended_values = jax.vmap(self.compute_attention, in_axes=(0, 0, 0, 0, 0, None))(
          query, key, value, level_edge_fts, graph_fts, level_adj_mat)  # (b, h, n, d)
      
      if self.reducer == 'max':
        aggregated_fts = jnp.max(attended_values, axis=1)
      elif self.reducer == 'sum':
        aggregated_fts = jnp.sum(attended_values, axis=1)
      elif self.reducer == 'mean':
        aggregated_fts = jnp.mean(attended_values, axis=1)
      else:
        raise ValueError(f"Unsupported reducer: {self.reducer}")
 
    else:
      # Aggregate without attention
      level_edge_fts = jnp.max(level_node_fts[:, None, :, :] +
                               level_node_fts[:, :, None, :] +
                               level_edge_fts, axis=-1, keepdims=True)
      level_edge_fts = level_edge_fts * level_adj_mat[..., None]
      if self.reducer == 'max':
        aggregated_fts = jnp.max(level_edge_fts, axis=-2)
      elif self.reducer == 'sum':
        aggregated_fts = jnp.sum(level_edge_fts, axis=-2)
      elif self.reducer == 'mean':
        aggregated_fts = jnp.mean(level_edge_fts, axis=-2)
      else:
        raise ValueError(f"Unsupported reducer: {self.reducer}")

    return aggregated_fts

  # def compute_attention(self, query, key, value, edge_fts, graph_fts, adj_mat):
  #   """Compute attention scores with graph-level features and adjacency masking."""

  #   # DEBUGGING PRINT SHAPES
  #   print("Inside compute_attention:")
  #   print("query shape: ", query.shape)
  #   print("key shape: ", key.shape)
  #   print("value shape: ", value.shape)
  #   print("edge_fts shape: ", edge_fts.shape)
  #   print("graph_fts shape: ", graph_fts.shape)
  #   print("adj_mat shape: ", adj_mat.shape)
    

  #   # Compute attention scores based on node features
  #   node_query = hk.Linear(self.out_size)(query)
  #   node_key = hk.Linear(self.out_size)(key)
  #   node_attention_scores = jnp.einsum('hnd,hmd->hnm', node_query, node_key)

  #   # Compute attention scores based on edge features
  #   edge_query = hk.Linear(self.out_size)(query)
  #   edge_key = hk.Linear(self.out_size)(edge_fts)
  #   edge_attention_scores = jnp.einsum('hnd,hnmd->hnm', edge_query, edge_key)

  #   # Compute attention scores based on graph-level features
  #   graph_query = hk.Linear(self.out_size)(query)
  #   graph_key = hk.Linear(self.out_size)(graph_fts)
  #   graph_attention_scores = jnp.einsum('hnd,hd->hn', graph_query, graph_key)
  #   graph_attention_scores = jnp.expand_dims(graph_attention_scores, axis=-1)  # (1, 4, 1)
  #   graph_attention_scores = jnp.tile(graph_attention_scores, [1, 1, 4])  # (1, 4, 4)

  #   # DEBUGGING PRINT SHAPES
  #   print("After computing attention scores:")
  #   print("node_attention_scores shape: ", node_attention_scores.shape)
  #   print("edge_attention_scores shape: ", edge_attention_scores.shape)
  #   print("graph_attention_scores shape: ", graph_attention_scores.shape)


  #   # Combine node, edge, and graph attention scores
  #   attention_scores = node_attention_scores + edge_attention_scores + graph_attention_scores

  #   # Mask attention scores based on adjacency matrix
  #   adj_mat = adj_mat[0]  # (4, 4)
  #   mask = -1e9 * (1.0 - adj_mat)
  #   attention_scores = jnp.where(adj_mat, attention_scores, mask)
  #   attention_scores = jax.nn.softmax(attention_scores, axis=-1)

  #   # Compute attended values
  #   attended_values = jnp.einsum('hnm,hmd->hnd', attention_scores, value)
  #   return attended_values

  # def aggregate_level(self, level_node_fts, level_edge_fts, level_graph_fts, level_adj_mat, b, n):
  #   """Aggregate information at a single level."""
  #   if self.nb_heads > 0:
  #     # Implement multi-head attention
  #     head_size = self.out_size // self.nb_heads
  #     query = hk.Linear(self.out_size)(level_node_fts)
  #     key = hk.Linear(self.out_size)(level_node_fts)
  #     value = hk.Linear(self.out_size)(level_node_fts)

  #     # DEBUGGGING PRINT SHAPES
  #     print("Before reshaping for multi-head attention:")
  #     print("query shape: ", query.shape)
  #     print("key shape: ", key.shape)
  #     print("value shape: ", value.shape)
  #     print("level_edge_fts shape: ", level_edge_fts.shape)
  #     print("level_graph_fts shape: ", level_graph_fts.shape)
  #     print("level_adj_mat shape: ", level_adj_mat.shape)

  #     # Reshape for multi-head attention
  #     query = jnp.reshape(query, (b, n, self.nb_heads, head_size))
  #     key = jnp.reshape(key, (b, n, self.nb_heads, head_size))
  #     value = jnp.reshape(value, (b, n, self.nb_heads, head_size))
  #     query = jnp.transpose(query, (0, 2, 1, 3))  # (b, h, n, d)
  #     key = jnp.transpose(key, (0, 2, 1, 3))  # (b, h, n, d)
  #     value = jnp.transpose(value, (0, 2, 1, 3))  # (b, h, n, d)

  #     # Reshape edge and graph features for multi-head attention
  #     level_edge_fts = jnp.reshape(level_edge_fts, (b, n, n, self.nb_heads, head_size))
  #     level_edge_fts = jnp.transpose(level_edge_fts, (0, 3, 1, 2, 4))  # (b, h, n, n, d)
  #     level_graph_fts = jnp.repeat(level_graph_fts[:, None, :], self.nb_heads, axis=1)  # (b, h, d)

  #     # DEBUGGING PRINT SHAPES
  #     print("After reshaping for multi-head attention:")
  #     print("query shape: ", query.shape)
  #     print("key shape: ", key.shape)
  #     print("value shape: ", value.shape)
  #     print("level_edge_fts shape: ", level_edge_fts.shape)
  #     print("level_graph_fts shape: ", level_graph_fts.shape)
  #     print("level_adj_mat shape: ", level_adj_mat.shape)

  #     # Compute attention and aggregate
  #     attended_values = jax.vmap(self.compute_attention, in_axes=(0, 0, 0, 0, 0, None))(
  #         query, key, value, level_edge_fts, level_graph_fts, level_adj_mat)  # (b, h, n, d)
  #     attended_values = jnp.transpose(attended_values, (0, 2, 1, 3))  # (b, n, h, d)
  #     attended_values = jnp.reshape(attended_values, (b, n, self.out_size))

  #     # DEBUGGING PRINT SHAPES
  #     print("After computing attention and reshaping:")
  #     print("attended_values shape: ", attended_values.shape)


  #     if self.reducer == 'max':
  #       aggregated_fts = jnp.max(attended_values, axis=1)
  #     elif self.reducer == 'sum':
  #       aggregated_fts = jnp.sum(attended_values, axis=1)
  #     elif self.reducer == 'mean':
  #       aggregated_fts = jnp.mean(attended_values, axis=1)
  #     else:
  #       raise ValueError(f"Unsupported reducer: {self.reducer}")
 
  #   else:
  #     # Aggregate without attention
  #     level_edge_fts = jnp.max(level_node_fts[:, None, :, :] +
  #                              level_node_fts[:, :, None, :] +
  #                              level_edge_fts, axis=-1, keepdims=True)
  #     level_edge_fts = level_edge_fts * level_adj_mat[..., None]
  #     if self.reducer == 'max':
  #       aggregated_fts = jnp.max(level_edge_fts, axis=-2)
  #     elif self.reducer == 'sum':
  #       aggregated_fts = jnp.sum(level_edge_fts, axis=-2)
  #     elif self.reducer == 'mean':
  #       aggregated_fts = jnp.mean(level_edge_fts, axis=-2)
  #     else:
  #       raise ValueError(f"Unsupported reducer: {self.reducer}")

  #   return aggregated_fts

  def update_node_fts(self, level, node_fts, edge_fts, graph_fts, adj_mat):
    """Update node features at a single level."""

    # DEBUGGING PRINT SHAPES
    print("Inside update_node_fts:")
    print("level: ", level)
    print("node_fts shape: ", node_fts.shape)
    print("edge_fts shape: ", edge_fts.shape)
    print("graph_fts shape: ", graph_fts.shape)
    print("adj_mat shape: ", adj_mat.shape)

    level_node_fts = hk.Linear(self.out_size, name=f"level_{level}_linear")(node_fts)
    if self.activation_fn is not None:
      level_node_fts = self.activation_fn(level_node_fts)

    # Compute level-specific edge features
    level_edge_fts = hk.Linear(self.out_size, name=f"level_{level}_edge_linear")(edge_fts)
    if self.activation_fn is not None:
      level_edge_fts = self.activation_fn(level_edge_fts)

    # Compute level-specific graph features
    level_graph_fts = hk.Linear(self.out_size, name=f"level_{level}_graph_linear")(graph_fts)
    if self.activation_fn is not None:
      level_graph_fts = self.activation_fn(level_graph_fts)

    # DEBUGGING PRINT SHAPES
    print("After computing level-specific features:")
    print("level_node_fts shape: ", level_node_fts.shape)
    print("level_edge_fts shape: ", level_edge_fts.shape)
    print("level_graph_fts shape: ", level_graph_fts.shape)

    b, n, _ = node_fts.shape
    aggregated_fts = self.aggregate_level(level_node_fts, level_edge_fts, level_graph_fts, adj_mat, b, n)
    if self.use_skip_connection:
      aggregated_fts += node_fts
    hk.dropout(hk.next_rng_key(), self.dropout_rate, node_fts)

    return aggregated_fts

class PGN(Processor):
  """Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      mid_act: Optional[_Fn] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      reduction: _Fn = jnp.max,
      msgs_mlp_sizes: Optional[List[int]] = None,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      name: str = 'mpnn_aggr',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.mid_act = mid_act
    self.activation = activation
    self.reduction = reduction
    self._msgs_mlp_sizes = msgs_mlp_sizes
    self.use_ln = use_ln
    self.use_triplets = use_triplets
    self.nb_triplet_fts = nb_triplet_fts
    self.gated = gated

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    m_1 = hk.Linear(self.mid_size)
    m_2 = hk.Linear(self.mid_size)
    m_e = hk.Linear(self.mid_size)
    m_g = hk.Linear(self.mid_size)

    o1 = hk.Linear(self.out_size)
    o2 = hk.Linear(self.out_size)

    msg_1 = m_1(z)
    msg_2 = m_2(z)
    msg_e = m_e(edge_fts)
    msg_g = m_g(graph_fts)

    tri_msgs = None

    if self.use_triplets:
      # Triplet messages, as done by Dudzik and Velickovic (2022)
      triplets = get_triplet_msgs(z, edge_fts, graph_fts, self.nb_triplet_fts)

      o3 = hk.Linear(self.out_size)
      tri_msgs = o3(jnp.max(triplets, axis=1))  # (B, N, N, H)

      if self.activation is not None:
        tri_msgs = self.activation(tri_msgs)

    msgs = (
        jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
        msg_e + jnp.expand_dims(msg_g, axis=(1, 2)))

    if self._msgs_mlp_sizes is not None:
      msgs = hk.nets.MLP(self._msgs_mlp_sizes)(jax.nn.relu(msgs))

    if self.mid_act is not None:
      msgs = self.mid_act(msgs)

    if self.reduction == jnp.mean:
      msgs = jnp.sum(msgs * jnp.expand_dims(adj_mat, -1), axis=1)
      msgs = msgs / jnp.sum(adj_mat, axis=-1, keepdims=True)
    elif self.reduction == jnp.max:
      maxarg = jnp.where(jnp.expand_dims(adj_mat, -1),
                         msgs,
                         -BIG_NUMBER)
      msgs = jnp.max(maxarg, axis=1)
    else:
      msgs = self.reduction(msgs * jnp.expand_dims(adj_mat, -1), axis=1)

    h_1 = o1(z)
    h_2 = o2(msgs)

    ret = h_1 + h_2

    if self.activation is not None:
      ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    if self.gated:
      gate1 = hk.Linear(self.out_size)
      gate2 = hk.Linear(self.out_size)
      gate3 = hk.Linear(self.out_size, b_init=hk.initializers.Constant(-3))
      gate = jax.nn.sigmoid(gate3(jax.nn.relu(gate1(z) + gate2(msgs))))
      ret = ret * gate + hidden * (1-gate)
    
    if latents_config.save_latents:
      potential_latents = {'z': z, 'msgs': msgs, 'tri_msgs': tri_msgs, 'ret': ret}
      saved_latents = {key: value for key, value in potential_latents.items() 
                          if key in latents_config.save_latents}
    else:
      saved_latents = None

    return ret, tri_msgs, saved_latents  # pytype: disable=bad-return-type  # numpy-scalars


class DeepSets(PGN):
  """Deep Sets (Zaheer et al., NeurIPS 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    assert adj_mat.ndim == 3
    adj_mat = jnp.ones_like(adj_mat) * jnp.eye(adj_mat.shape[-1])
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class MPNN(PGN):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class PGNMask(PGN):
  """Masked Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

  @property
  def inf_bias(self):
    return True

  @property
  def inf_bias_edge(self):
    return True


class MemNetMasked(Processor):
  """Implementation of End-to-End Memory Networks.

  Inspired by the description in https://arxiv.org/abs/1503.08895.
  """

  def __init__(
      self,
      vocab_size: int,
      sentence_size: int,
      linear_output_size: int,
      embedding_size: int = 16,
      memory_size: Optional[int] = 128,
      num_hops: int = 1,
      nonlin: Callable[[Any], Any] = jax.nn.relu,
      apply_embeddings: bool = True,
      init_func: hk.initializers.Initializer = jnp.zeros,
      use_ln: bool = False,
      name: str = 'memnet') -> None:
    """Constructor.

    Args:
      vocab_size: the number of words in the dictionary (each story, query and
        answer come contain symbols coming from this dictionary).
      sentence_size: the dimensionality of each memory.
      linear_output_size: the dimensionality of the output of the last layer
        of the model.
      embedding_size: the dimensionality of the latent space to where all
        memories are projected.
      memory_size: the number of memories provided.
      num_hops: the number of layers in the model.
      nonlin: non-linear transformation applied at the end of each layer.
      apply_embeddings: flag whether to aply embeddings.
      init_func: initialization function for the biases.
      use_ln: whether to use layer normalisation in the model.
      name: the name of the model.
    """
    super().__init__(name=name)
    self._vocab_size = vocab_size
    self._embedding_size = embedding_size
    self._sentence_size = sentence_size
    self._memory_size = memory_size
    self._linear_output_size = linear_output_size
    self._num_hops = num_hops
    self._nonlin = nonlin
    self._apply_embeddings = apply_embeddings
    self._init_func = init_func
    self._use_ln = use_ln
    # Encoding part: i.e. "I" of the paper.
    self._encodings = _position_encoding(sentence_size, embedding_size)

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MemNet inference step."""

    del hidden
    node_and_graph_fts = jnp.concatenate([node_fts, graph_fts[:, None]],
                                         axis=1)
    edge_fts_padded = jnp.pad(edge_fts * adj_mat[..., None],
                              ((0, 0), (0, 1), (0, 1), (0, 0)))
    nxt_hidden = jax.vmap(self._apply, (1), 1)(node_and_graph_fts,
                                               edge_fts_padded)

    # Broadcast hidden state corresponding to graph features across the nodes.
    nxt_hidden = nxt_hidden[:, :-1] + nxt_hidden[:, -1:]
    return nxt_hidden, None  # pytype: disable=bad-return-type  # numpy-scalars

  def _apply(self, queries: _Array, stories: _Array) -> _Array:
    """Apply Memory Network to the queries and stories.

    Args:
      queries: Tensor of shape [batch_size, sentence_size].
      stories: Tensor of shape [batch_size, memory_size, sentence_size].

    Returns:
      Tensor of shape [batch_size, vocab_size].
    """
    if self._apply_embeddings:
      query_biases = hk.get_parameter(
          'query_biases',
          shape=[self._vocab_size - 1, self._embedding_size],
          init=self._init_func)
      stories_biases = hk.get_parameter(
          'stories_biases',
          shape=[self._vocab_size - 1, self._embedding_size],
          init=self._init_func)
      memory_biases = hk.get_parameter(
          'memory_contents',
          shape=[self._memory_size, self._embedding_size],
          init=self._init_func)
      output_biases = hk.get_parameter(
          'output_biases',
          shape=[self._vocab_size - 1, self._embedding_size],
          init=self._init_func)

      nil_word_slot = jnp.zeros([1, self._embedding_size])

    # This is "A" in the paper.
    if self._apply_embeddings:
      stories_biases = jnp.concatenate([stories_biases, nil_word_slot], axis=0)
      memory_embeddings = jnp.take(
          stories_biases, stories.reshape([-1]).astype(jnp.int32),
          axis=0).reshape(list(stories.shape) + [self._embedding_size])
      memory_embeddings = jnp.pad(
          memory_embeddings,
          ((0, 0), (0, self._memory_size - jnp.shape(memory_embeddings)[1]),
           (0, 0), (0, 0)))
      memory = jnp.sum(memory_embeddings * self._encodings, 2) + memory_biases
    else:
      memory = stories

    # This is "B" in the paper. Also, when there are no queries (only
    # sentences), then there these lines are substituted by
    # query_embeddings = 0.1.
    if self._apply_embeddings:
      query_biases = jnp.concatenate([query_biases, nil_word_slot], axis=0)
      query_embeddings = jnp.take(
          query_biases, queries.reshape([-1]).astype(jnp.int32),
          axis=0).reshape(list(queries.shape) + [self._embedding_size])
      # This is "u" in the paper.
      query_input_embedding = jnp.sum(query_embeddings * self._encodings, 1)
    else:
      query_input_embedding = queries

    # This is "C" in the paper.
    if self._apply_embeddings:
      output_biases = jnp.concatenate([output_biases, nil_word_slot], axis=0)
      output_embeddings = jnp.take(
          output_biases, stories.reshape([-1]).astype(jnp.int32),
          axis=0).reshape(list(stories.shape) + [self._embedding_size])
      output_embeddings = jnp.pad(
          output_embeddings,
          ((0, 0), (0, self._memory_size - jnp.shape(output_embeddings)[1]),
           (0, 0), (0, 0)))
      output = jnp.sum(output_embeddings * self._encodings, 2)
    else:
      output = stories

    intermediate_linear = hk.Linear(self._embedding_size, with_bias=False)

    # Output_linear is "H".
    output_linear = hk.Linear(self._linear_output_size, with_bias=False)

    for hop_number in range(self._num_hops):
      query_input_embedding_transposed = jnp.transpose(
          jnp.expand_dims(query_input_embedding, -1), [0, 2, 1])

      # Calculate probabilities.
      probs = jax.nn.softmax(
          jnp.sum(memory * query_input_embedding_transposed, 2))

      # Calculate output of the layer by multiplying by C.
      transposed_probs = jnp.transpose(jnp.expand_dims(probs, -1), [0, 2, 1])
      transposed_output_embeddings = jnp.transpose(output, [0, 2, 1])

      # This is "o" in the paper.
      layer_output = jnp.sum(transposed_output_embeddings * transposed_probs, 2)

      # Finally the answer
      if hop_number == self._num_hops - 1:
        # Please note that in the TF version we apply the final linear layer
        # in all hops and this results in shape mismatches.
        output_layer = output_linear(query_input_embedding + layer_output)
      else:
        output_layer = intermediate_linear(query_input_embedding + layer_output)

      query_input_embedding = output_layer
      if self._nonlin:
        output_layer = self._nonlin(output_layer)

    # This linear here is "W".
    ret = hk.Linear(self._vocab_size, with_bias=False)(output_layer)

    if self._use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    return ret


class MemNetFull(MemNetMasked):
  """Memory Networks with full adjacency matrix."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


ProcessorFactory = Callable[[int], Processor]


def get_processor_factory(kind: str,
                          use_ln: bool,
                          nb_triplet_fts: int,
                          nb_heads: int,
                          nb_hgp_levels: int,
                          use_skip_connection: bool,
                          dropout_rate: Optional[float] = 0.0) -> ProcessorFactory:
  """Returns a processor factory.

  Args:
    kind: One of the available types of processor.
    use_ln: Whether the processor passes the output through a layernorm layer.
    nb_triplet_fts: How many triplet features to compute.
    nb_heads: Number of attention heads for GAT processors.
  Returns:
    A callable that takes an `out_size` parameter (equal to the hidden
    dimension of the network) and returns a processor instance.
  """
  def _factory(out_size: int):
    if kind == 'deepsets':
      processor = DeepSets(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=0
      )
    elif kind == 'gat':
      processor = GAT(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln,
      )
    elif kind == 'gat_full':
      processor = GATFull(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'gatv2':
      processor = GATv2(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'gatv2_full':
      processor = GATv2Full(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'hgp':
      processor = HierarchicalGraphProcessor(
          out_size=out_size,
          nb_hgp_levels=nb_hgp_levels,
          use_skip_connection=use_skip_connection,
          dropout_rate=dropout_rate,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'memnet_full':
      processor = MemNetFull(
          vocab_size=out_size,
          sentence_size=out_size,
          linear_output_size=out_size,
      )
    elif kind == 'memnet_masked':
      processor = MemNetMasked(
          vocab_size=out_size,
          sentence_size=out_size,
          linear_output_size=out_size,
      )
    elif kind == 'mpnn':
      processor = MPNN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=0,
      )
    elif kind == 'pgn':
      processor = PGN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=0,
      )
    elif kind == 'pgn_mask':
      processor = PGNMask(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=0,
      )
    elif kind == 'triplet_mpnn':
      processor = MPNN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
      )
    elif kind == 'triplet_pgn':
      processor = PGN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
      )
    elif kind == 'triplet_pgn_mask':
      processor = PGNMask(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
      )
    elif kind == 'gpgn':
      processor = PGN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'gpgn_mask':
      processor = PGNMask(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'gmpnn':
      processor = MPNN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'triplet_gpgn':
      processor = PGN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'triplet_gpgn_mask':
      processor = PGNMask(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'triplet_gmpnn':
      processor = MPNN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    else:
      raise ValueError('Unexpected processor kind ' + kind)

    return processor

  return _factory


def _position_encoding(sentence_size: int, embedding_size: int) -> np.ndarray:
  """Position Encoding described in section 4.1 [1]."""
  encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
  ls = sentence_size + 1
  le = embedding_size + 1
  for i in range(1, le):
    for j in range(1, ls):
      encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
  encoding = 1 + 4 * encoding / embedding_size / sentence_size
  return np.transpose(encoding)
