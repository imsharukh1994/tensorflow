# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Python module for Session ops, vars, and functions exported by # Disable pylint invalid name warnings for legacy functions.
# pylint: disable=invalid-name
# Disable pylint undefined-variable for variables exported in shared object via pybind11.
# pylint: disable=undefined-variable

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client._pywrap_tf_session import (
    _TF_SetTarget,
    _TF_SetConfig,
    _TF_NewSessionOptions,
    TF_Reset_wrapper,  # Ensure this function is defined somewhere
    TF_DeleteSessionOptions  # Ensure this function is defined somewhere
)
from tensorflow.python.util import tf_stack

# Version and build information for compatibility
__version__ = str(get_version())
__git_version__ = str(get_git_version())
__compiler_version__ = str(get_compiler_version())
__cxx11_abi_flag__ = get_cxx11_abi_flag()
__cxx_version__ = get_cxx_version()
__monolithic_build__ = get_monolithic_build()

# Session-related constants (you can adjust the comments for these as needed)
GRAPH_DEF_VERSION = get_graph_def_version()
GRAPH_DEF_VERSION_MIN_CONSUMER = get_graph_def_version_min_consumer()
GRAPH_DEF_VERSION_MIN_PRODUCER = get_graph_def_version_min_producer()
TENSOR_HANDLE_KEY = get_tensor_handle_key()


def TF_NewSessionOptions(target=None, config=None):
    """Create new session options with optional target and config."""
    opts = _TF_NewSessionOptions()
    if target:
        _TF_SetTarget(opts, target)
    if config:
        config_str = config.SerializeToString()
        _TF_SetConfig(opts, config_str)
    return opts


def TF_Reset(target, containers=None, config=None):
    """Reset the TensorFlow session."""
    opts = TF_NewSessionOptions(target=target, config=config)
    try:
        TF_Reset_wrapper(opts, containers)
    finally:
        TF_DeleteSessionOptions(opts)
    TF_Reset_wrapper(opts, containers)
  finally:
    TF_DeleteSessionOptions(opts)
