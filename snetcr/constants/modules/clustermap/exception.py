#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class ClusterMapException(
        Exception
        ):
    """Base class for exceptions."""

class ClusterMapError(
        ClusterMapException
        ):
    """Exception for serious errors."""