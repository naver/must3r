# Copyright (C) 2025-present Naver Corporation. All rights reserved.

def get_random_choice(rng, possibilities):
    if len(possibilities) > 0:
        return rng.choice(sorted(possibilities))
    else:
        return None
