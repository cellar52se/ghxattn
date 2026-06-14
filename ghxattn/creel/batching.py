from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from ghxattn.creel.bundles import VENDORS

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class EyeRecord:
    eye_id: int
    site: int
    vendor: int
    race: int
    has_oct: bool
    vf_seq: FloatArray
    dt: FloatArray
    oct_raw: FloatArray
    vf_target: FloatArray
    slope: float


@dataclass(frozen=True)
class EyeBatch:
    vf_seq: torch.Tensor
    dt: torch.Tensor
    visit_mask: torch.Tensor
    oct_raw: torch.Tensor
    has_oct: torch.Tensor
    vf_target: torch.Tensor
    slope: torch.Tensor
    site: torch.Tensor
    vendor: torch.Tensor
    race: torch.Tensor
    eye_id: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.vf_seq.shape[0])

    def to(self, device: torch.device) -> EyeBatch:
        return EyeBatch(
            vf_seq=self.vf_seq.to(device),
            dt=self.dt.to(device),
            visit_mask=self.visit_mask.to(device),
            oct_raw=self.oct_raw.to(device),
            has_oct=self.has_oct.to(device),
            vf_target=self.vf_target.to(device),
            slope=self.slope.to(device),
            site=self.site.to(device),
            vendor=self.vendor.to(device),
            race=self.race.to(device),
            eye_id=self.eye_id.to(device),
        )


def vendor_bucket(vendor: int) -> int:
    if vendor < 0:
        return len(VENDORS)
    return vendor


def collate_eyes(records: Sequence[EyeRecord]) -> EyeBatch:
    batch = len(records)
    max_t = max(record.vf_seq.shape[0] for record in records)
    locations = records[0].vf_seq.shape[1]
    n_oct = records[0].oct_raw.shape[0]
    raw_dim = records[0].oct_raw.shape[1]

    vf_seq = torch.zeros(batch, max_t, locations, dtype=torch.float32)
    dt = torch.zeros(batch, max_t, dtype=torch.float32)
    visit_mask = torch.zeros(batch, max_t, dtype=torch.bool)
    oct_raw = torch.zeros(batch, n_oct, raw_dim, dtype=torch.float32)
    has_oct = torch.zeros(batch, dtype=torch.bool)
    vf_target = torch.zeros(batch, locations, dtype=torch.float32)
    slope = torch.zeros(batch, dtype=torch.float32)
    site = torch.zeros(batch, dtype=torch.long)
    vendor = torch.zeros(batch, dtype=torch.long)
    race = torch.zeros(batch, dtype=torch.long)
    eye_id = torch.zeros(batch, dtype=torch.long)

    for index, record in enumerate(records):
        visits = record.vf_seq.shape[0]
        vf_seq[index, :visits] = torch.from_numpy(record.vf_seq)
        dt[index, :visits] = torch.from_numpy(record.dt)
        visit_mask[index, :visits] = True
        oct_raw[index] = torch.from_numpy(record.oct_raw)
        has_oct[index] = record.has_oct
        vf_target[index] = torch.from_numpy(record.vf_target)
        slope[index] = record.slope
        site[index] = record.site
        vendor[index] = vendor_bucket(record.vendor)
        race[index] = record.race
        eye_id[index] = record.eye_id

    return EyeBatch(
        vf_seq=vf_seq,
        dt=dt,
        visit_mask=visit_mask,
        oct_raw=oct_raw,
        has_oct=has_oct,
        vf_target=vf_target,
        slope=slope,
        site=site,
        vendor=vendor,
        race=race,
        eye_id=eye_id,
    )
