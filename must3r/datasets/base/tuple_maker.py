# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import copy
from must3r.datasets.base.random import get_random_choice


def select_tuple_from_pairs(pairs_getter, view_getter, num_views, memory_num_views, rng, idx1, idx2):
    selected_idx = [idx1, idx2]
    selected_idx_set = set(selected_idx)

    possibilities = pairs_getter(idx1).union(pairs_getter(idx2)).difference(selected_idx_set)

    for _ in range(2, num_views):
        if len(possibilities) == 0:
            break
        # pick a random value
        new_idx = rng.choice(sorted(possibilities))
        assert new_idx not in selected_idx_set
        selected_idx.append(new_idx)
        selected_idx_set.add(new_idx)
        if len(selected_idx) <= memory_num_views:
            possibilities = possibilities.union(pairs_getter(new_idx))
        possibilities = possibilities.difference(selected_idx_set)

    views = []
    for view_idx in selected_idx:
        views.append(view_getter(view_idx, rng))

    return fill_views(views, num_views)


def select_tuple_from_360_scene(is_valid_getter, is_valid_check, view_getter,
                                nimg_per_scene, num_views, rng, idx):
    views = []
    possibilities = set(range(nimg_per_scene))
    img_idx = idx
    if img_idx not in possibilities:
        img_idx = get_random_choice(rng, possibilities)
    while len(views) < num_views and img_idx is not None:  # some images (few) have zero depth
        possibilities.remove(img_idx)
        if not is_valid_getter(img_idx):  # make sure that img_idx is valid
            img_idx = get_random_choice(rng, possibilities)
            continue
        view = view_getter(img_idx, rng)  # get the view
        if not is_valid_check(view, img_idx):
            img_idx = get_random_choice(rng, possibilities)
            continue
        views.append(view)
        img_idx = get_random_choice(rng, possibilities)  # select new idx for the next loop
    return fill_views(views, num_views)


def fill_views(views, num_views):
    if len(views) < num_views:
        # somehow failed to add all views: there wasn't enough valid
        while len(views) != num_views:
            views = views + copy.deepcopy(views)
            views = views[:num_views]
    return views
