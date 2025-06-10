#!/usr/bin/env python3
# Copyright (C) 2025-present Naver Corporation. All rights reserved.
from must3r.engine.train import get_args_parser, train


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    train(args)
