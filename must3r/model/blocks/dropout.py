# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
import torch.nn as nn


class MemoryDropoutSelector(nn.Module):
    def __init__(self, p=0.0) -> None:
        super().__init__()
        self.p = p

    def sel(self, N, protected=0, device='cuda', p=None):
        p = self.p if p is None else p
        N_x = N - protected
        if N_x > 0:
            if p < 1:
                tokens_to_drop = torch.sum(torch.rand(N_x, device=device) < p)
            else:
                tokens_to_drop = max(0, min(N - p, N_x))

            sel = torch.randperm(N_x, device=device)
            sel = torch.sort(sel[tokens_to_drop:]).values
            not_sel = torch.sort(sel[:tokens_to_drop]).values
            if protected > 0:
                sel = sel + protected
                not_sel = not_sel + protected
                protected_labels = torch.arange(protected, device=device)
                sel = torch.cat([protected_labels, sel], dim=-1)
        else:
            sel = torch.arange(N, device=device)
            not_sel = torch.zeros((0, ), device=device, dtype=torch.int)
        return sel, not_sel

    def forward(self, Nm, nimgs, N, protected=0, device='cuda', p=None):
        p = self.p if p is None else p
        if p == 0.0:
            return None, None
        assert nimgs > 0
        sel, not_sel = [], []
        sel0 = torch.arange(Nm, device=device)  # initialization or already dropped out at the previous iteration
        not_sel0 = torch.arange(0, device=device)
        sel.append(sel0)
        not_sel.append(not_sel0)

        for i in range(nimgs):
            sel_prev = sel[-1]
            not_sel_prev = not_sel[-1]

            N_prev = len(sel_prev)

            seli, not_seli = self.sel(N_prev + N, protected, device, p=p)
            keep_new_vals = seli >= N_prev
            discard_new_vals = not_seli >= N_prev

            old_keep = sel_prev[seli[~keep_new_vals]]
            old_discard = sel_prev[not_seli[~discard_new_vals]]

            offset = (Nm + i * N) - N_prev
            seli = torch.concatenate([old_keep, seli[keep_new_vals] + offset])
            not_seli = torch.concatenate(
                [not_sel_prev, old_discard, not_seli[discard_new_vals] + offset]
            )
            sel.append(seli)
            not_sel.append(not_seli)
        return sel, not_sel


class TemporaryMemoryDropoutSelector(MemoryDropoutSelector):
    def __init__(self, p=0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, Nm, nimgs, N, protected=0, device='cuda', p=None):
        p = self.p if p is None else p
        if p == 0.0:
            return None, None
        sel, not_sel = [], []
        for i in range(nimgs):
            mem_cnt = Nm + (i * N)
            seli, not_seli = self.sel(mem_cnt, protected, device, p=p)
            new_vals = torch.arange(N, device=device) + mem_cnt
            seli = torch.concatenate([seli, new_vals])
            sel.append(seli)
            not_sel.append(not_seli)
        return sel, not_sel
