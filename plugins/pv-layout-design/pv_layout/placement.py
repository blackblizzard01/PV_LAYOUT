"""
Inverter and transformer placement via k-means facility location, and cable
length calculation.

Algorithm
---------
1. Group rows into DC blocks whose capacity ≈ one inverter's rated power.
2. k = ceil(total_kWp / inverter_kW).  Run weighted k-means on block
   centroids to place k inverters (Weber problem / weighted facility
   location).  Reference: Drezner & Hamacher, *Facility Location:
   Applications and Theory* (2002), Ch. 2; sklearn KMeans (BSD-3 licence).
3. Transformer position = capacity-weighted centroid of inverter positions
   (single-transformer case).
4. Cable lengths = straight-line Euclidean distances (block→inverter DC;
   inverter→transformer AC).  Real trench routing is out of scope for this
   pass (see task brief §2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class InverterSpec:
    rated_kw: float  # rated AC output power, kW


@dataclass(frozen=True)
class TransformerSpec:
    rated_kva: float          # rated apparent power, kVA
    p_no_load_kw: float       # core/iron loss (constant), kW — from nameplate
    p_load_rated_kw: float    # copper/winding loss at rated load, kW — from nameplate


@dataclass
class PlacementResult:
    # DC side — grouped string blocks
    block_centroids: np.ndarray    # (n_blocks, 2)  (x, y) metres
    block_kwp: np.ndarray          # (n_blocks,)    DC capacity per block
    block_to_inverter: np.ndarray  # (n_blocks,)    index into inverter_positions

    # Inverters
    n_inverters: int
    inverter_positions: np.ndarray  # (n_inverters, 2)

    # Transformer(s) — currently always 1
    n_transformers: int
    transformer_positions: np.ndarray      # (n_transformers, 2)
    inverter_to_transformer: np.ndarray    # (n_inverters,) index into transformer_positions

    # Cable lengths
    dc_cable_lengths_m: np.ndarray  # (n_blocks,)    block centroid → assigned inverter
    ac_cable_lengths_m: np.ndarray  # (n_inverters,) inverter → assigned transformer
    total_dc_cable_m: float
    total_ac_cable_m: float


def _group_rows_into_blocks(
    row_centroids: np.ndarray,
    n_per_row: int,
    module_power_wp: float,
    inverter_rated_kw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Group consecutive rows into blocks whose DC capacity is approximately
    one inverter's rated output.  Returns (block_centroids, block_kwp).

    Grouping consecutive rows (south → north) keeps each block's centroid
    close to the modules it represents, which is important for the cable-
    length minimisation that follows.
    """
    row_kwp = n_per_row * module_power_wp / 1000.0
    rows_per_block = max(1, round(inverter_rated_kw / row_kwp))

    blocks: list[np.ndarray] = []
    kwps: list[float] = []
    n_rows = len(row_centroids)

    for start in range(0, n_rows, rows_per_block):
        end = min(start + rows_per_block, n_rows)
        chunk = row_centroids[start:end]
        blocks.append(chunk.mean(axis=0))
        kwps.append((end - start) * row_kwp)

    return np.array(blocks), np.array(kwps)


def place_equipment(
    row_centroids: np.ndarray,
    n_per_row: int,
    total_kwp: float,
    module_power_wp: float,
    inverter: InverterSpec,
    transformer: TransformerSpec,
    random_state: int = 42,
) -> PlacementResult:
    """
    Place inverters and transformer to minimise total cable length.

    Steps (following task brief §4):
      1. Group rows into DC blocks sized ≈ inverter_rated_kw.
      2. k = ceil(total_kwp / inverter.rated_kw).  Weighted k-means on block
         centroids (weight = block kWp) → inverter positions.
      3. Capacity-weighted centroid of inverter positions → transformer.
      4. Cable lengths: Euclidean straight-line.
    """
    block_centroids, block_kwp = _group_rows_into_blocks(
        row_centroids, n_per_row, module_power_wp, inverter.rated_kw
    )
    n_blocks = len(block_centroids)

    n_inverters = max(1, min(math.ceil(total_kwp / inverter.rated_kw), n_blocks))

    kmeans = KMeans(n_clusters=n_inverters, random_state=random_state, n_init="auto")
    kmeans.fit(block_centroids, sample_weight=block_kwp)
    labels = kmeans.labels_
    inverter_positions = kmeans.cluster_centers_

    # Transformer: capacity-weighted centroid of inverter positions.
    # Weight each inverter by the total kWp of blocks assigned to it.
    inv_kwp = np.array([
        block_kwp[labels == i].sum() for i in range(n_inverters)
    ])
    inv_kwp = np.maximum(inv_kwp, 1e-9)  # guard against empty cluster
    transformer_pos = np.average(inverter_positions, axis=0, weights=inv_kwp)
    transformer_positions = transformer_pos.reshape(1, 2)

    # Cable lengths
    dc_cable_lengths = np.linalg.norm(
        block_centroids - inverter_positions[labels], axis=1
    )
    ac_cable_lengths = np.linalg.norm(
        inverter_positions - transformer_positions[0], axis=1
    )

    return PlacementResult(
        block_centroids=block_centroids,
        block_kwp=block_kwp,
        block_to_inverter=labels,
        n_inverters=n_inverters,
        inverter_positions=inverter_positions,
        n_transformers=1,
        transformer_positions=transformer_positions,
        inverter_to_transformer=np.zeros(n_inverters, dtype=int),
        dc_cable_lengths_m=dc_cable_lengths,
        ac_cable_lengths_m=ac_cable_lengths,
        total_dc_cable_m=float(dc_cable_lengths.sum()),
        total_ac_cable_m=float(ac_cable_lengths.sum()),
    )
