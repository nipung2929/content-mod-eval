# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility wrapper for the real content moderation environment."""

try:
    from ..env.environment import ContentModEnvironment
except ImportError:
    from env.environment import ContentModEnvironment

__all__ = ["ContentModEnvironment"]
