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
3. Transformer placement (Layer 2 — F7):
   - n_transformers = 1 (default, Layer 1):  capacity-weighted centroid of
     all inverter positions.
   - n_transformers > 1:  run a second k-means on inverter positions,
     weighted by each inverter's kWp, to place multiple transformers.
     Source: RatedPower Design_Methodology §5.3.4 (power station sizing).
4. Cable lengths = straight-line Euclidean distances (block→inverter DC;
   inverter→transformer AC).
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
    block_centroids: np.ndarray    # (n_blocks, 2)
    block_kwp: np.ndarray          # (n_blocks,)
    block_to_inverter: np.ndarray  # (n_blocks,) index into inverter_positions

    # Inverters
    n_inverters: int
    inverter_positions: np.ndarray  # (n_inverters, 2)

    # Transformers — default 1 (Layer 1 compatible); F7 allows n > 1
    n_transformers: int
    transformer_positions: np.ndarray      # (n_transformers, 2)
    inverter_to_transformer: np.ndarray    # (n_inverters,) index into transformer_positions

    # Cable lengths
    dc_cable_lengths_m: np.ndarray  # (n_blocks,)
    ac_cable_lengths_m: np.ndarray  # (n_inverters,)
    total_dc_cable_m: float
    total_ac_cable_m: float


def _group_rows_into_blocks(
    row_centroids: np.ndarray,
    n_per_row: int,
    module_power_wp: float,
    inverter_rated_kw: float,
    n_per_row_array: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Group consecutive rows into blocks whose DC capacity ≈ inverter_rated_kw.

    If n_per_row_array is provided (polygon sites), each row's kWp is
    computed from its actual module count rather than the uniform n_per_row.
    """
    if n_per_row_array is not None:
        row_kwp_arr = n_per_row_array * module_power_wp / 1000.0
        avg_row_kwp = float(row_kwp_arr.mean())
    else:
        avg_row_kwp = n_per_row * module_power_wp / 1000.0
        row_kwp_arr = np.full(len(row_centroids), avg_row_kwp)

    rows_per_block = max(1, round(inverter_rated_kw / avg_row_kwp))

    blocks: list[np.ndarray] = []
    kwps: list[float] = []
    n_rows = len(row_centroids)

    for start in range(0, n_rows, rows_per_block):
        end = min(start + rows_per_block, n_rows)
        chunk = row_centroids[start:end]
        blocks.append(chunk.mean(axis=0))
        kwps.append(float(row_kwp_arr[start:end].sum()))

    return np.array(blocks), np.array(kwps)


def place_equipment(
    row_centroids: np.ndarray,
    n_per_row: int,
    total_kwp: float,
    module_power_wp: float,
    inverter: InverterSpec,
    transformer: TransformerSpec,
    random_state: int = 42,
    n_transformers: int = 1,
    n_per_row_array: np.ndarray | None = None,
) -> PlacementResult:
    """
    Place inverters and transformer(s) to minimise total cable length.

    Layer 1 parameters (unchanged):
      row_centroids, n_per_row, total_kwp, module_power_wp, inverter,
      transformer, random_state

    Layer 2 (F7) — n_transformers:
      Number of transformers to place.  Default = 1 (Layer 1 behaviour).
      If > 1, a second weighted k-means groups inverters into transformer
      zones.  Source: RatedPower Design_Methodology §5.3.4 (multiple power
      stations; the software reduces inverters per PS by one until filled).

    Layer 3 (polygon sites) — n_per_row_array:
      Per-row module counts from place_panels_polygon; enables accurate
      block kWp for irregular-width rows.
    """
    block_centroids, block_kwp = _group_rows_into_blocks(
        row_centroids, n_per_row, module_power_wp, inverter.rated_kw,
        n_per_row_array=n_per_row_array,
    )
    n_blocks = len(block_centroids)
    n_inverters = max(1, min(math.ceil(total_kwp / inverter.rated_kw), n_blocks))

    kmeans = KMeans(n_clusters=n_inverters, random_state=random_state, n_init="auto")
    kmeans.fit(block_centroids, sample_weight=block_kwp)
    labels = kmeans.labels_
    inverter_positions = kmeans.cluster_centers_

    inv_kwp = np.array([
        block_kwp[labels == i].sum() for i in range(n_inverters)
    ])
    inv_kwp = np.maximum(inv_kwp, 1e-9)

    # ── Transformer placement (F7) ────────────────────────────────────────────
    n_tx = max(1, min(n_transformers, n_inverters))

    if n_tx == 1:
        # Layer 1 path: single transformer at capacity-weighted centroid
        tx_pos = np.average(inverter_positions, axis=0, weights=inv_kwp)
        transformer_positions = tx_pos.reshape(1, 2)
        inverter_to_tx = np.zeros(n_inverters, dtype=int)
    else:
        # F7: multiple transformers via second k-means on inverter positions
        # Source: RatedPower Design_Methodology §5.3.4 — multiple power stations
        tx_kmeans = KMeans(n_clusters=n_tx, random_state=random_state, n_init="auto")
        tx_kmeans.fit(inverter_positions, sample_weight=inv_kwp)
        inverter_to_tx = tx_kmeans.labels_
        transformer_positions = tx_kmeans.cluster_centers_

    # ── Cable lengths ─────────────────────────────────────────────────────────
    dc_cable_lengths = np.linalg.norm(
        block_centroids - inverter_positions[labels], axis=1
    )
    ac_cable_lengths = np.linalg.norm(
        inverter_positions - transformer_positions[inverter_to_tx], axis=1
    )

    return PlacementResult(
        block_centroids=block_centroids,
        block_kwp=block_kwp,
        block_to_inverter=labels,
        n_inverters=n_inverters,
        inverter_positions=inverter_positions,
        n_transformers=n_tx,
        transformer_positions=transformer_positions,
        inverter_to_transformer=inverter_to_tx,
        dc_cable_lengths_m=dc_cable_lengths,
        ac_cable_lengths_m=ac_cable_lengths,
        total_dc_cable_m=float(dc_cable_lengths.sum()),
        total_ac_cable_m=float(ac_cable_lengths.sum()),
    )
