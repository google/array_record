# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""array_record_data_source module.

Warning: this is an experimental module. The interface might change in the
future without backwards compatibility.

Data source is an abstraction that is responsible for retrieving data records
from storage backend in ML workloads (e.g. a set of files, a database). It
implements a simple Python interface to query ArrayRecord files:

```
class RandomAccessDataSource(Protocol, Generic[T]):
  def __len__(self) -> int:
    ...

  def __getitem__(self, record_keys: Sequence[int]) -> Sequence[T]:
    ...
```
"""

import bisect
from concurrent import futures
import dataclasses
import hashlib
import itertools
import os
import pathlib
import re
import typing
from typing import Any, Callable, List, Mapping, Protocol, Sequence, Tuple, TypeVar, Union

from absl import flags
from absl import logging
from etils import epath

from array_record.python.array_record_module import ArrayRecordReader


# TODO(jolesiak): Decide what to do with these flags, e.g., remove them (could
# be appropriate if we decide to use asyncio) or move them somewhere else and
# pass the number of threads as an argument. For now, since we experiment, it's
# convenient to have them.
_GRAIN_NUM_THREADS_COMPUTING_NUM_RECORDS = flags.DEFINE_integer(
    "grain_num_threads_computing_num_records",
    64,
    (
        "The number of threads used to fetch file instructions (i.e., the max"
        " number of Array Record files opened while calculating the total"
        " number of records)."
    ),
)
_GRAIN_NUM_THREADS_FETCHING_RECORDS = flags.DEFINE_integer(
    "grain_num_threads_fetching_records",
    64,
    (
        "The number of threads used to fetch records from Array Record files. "
        "(i.e., the max number of Array Record files opened while fetching "
        "records)."
    ),
)

T = TypeVar("T")


def _run_in_parallel(
    function: Callable[..., T],
    list_of_kwargs_to_function: Sequence[Mapping[str, Any]],
    num_workers: int,
) -> List[T]:
  """Runs `function` in parallel threads with given keyword arguments.

  This is useful for performing IO in parallel. CPU bound functions will likely
  not be faster.

  Args:
    function: The function to execute in parallel.
    list_of_kwargs_to_function: A list of dicts mapping from string to argument
      value. These will be passed into `function` as kwargs.
    num_workers: Number of threads in the thread pool.

  Returns:
    list of return values from function, in the same order as the arguments in
    list_of_kwargs_to_function.
  """
  if num_workers < 1:
    raise ValueError("num_workers must be >=1 for parallelism.")
  thread_futures = []
  with futures.ThreadPoolExecutor(num_workers) as executor:
    for kwargs in list_of_kwargs_to_function:
      future = executor.submit(function, **kwargs)
      thread_futures.append(future)
    futures_as_completed = futures.as_completed(thread_futures)
    for completed_future in futures_as_completed:
      if completed_future.exception():
        # Cancel all remaining futures, if possible. In Python>3.8, you can call
        # `executor.shutdown(cancel_futures=True)`.
        for remaining_future in thread_futures:
          remaining_future.cancel()
        raise completed_future.exception()
  return [future.result() for future in thread_futures]


@dataclasses.dataclass(frozen=True)
class _ReadInstruction:
  """Internal class used to keep track of files and records to read from them."""

  filename: str
  start: int
  end: int
  num_records: int = dataclasses.field(init=False)

  def __post_init__(self):
    object.__setattr__(self, "num_records", self.end - self.start)


@typing.runtime_checkable
class FileInstruction(Protocol):
  """Protocol with same interface as FileInstruction returned by TFDS.

  ArrayRecordDataSource would accept objects implementing this protocol without
  depending on TFDS.
  """

  filename: str
  skip: int
  take: int
  examples_in_shard: int


PathLikeOrFileInstruction = Union[epath.PathLike, FileInstruction]


def _get_read_instructions(
    paths: Sequence[PathLikeOrFileInstruction],
) -> Sequence[_ReadInstruction]:
  """Constructs ReadInstructions for given paths."""

  def get_read_instruction(path: PathLikeOrFileInstruction) -> _ReadInstruction:
    if isinstance(path, FileInstruction):
      start = path.skip
      end = path.skip + path.take
      path = os.fspath(path.filename)
    else:
      path = os.fspath(path)
      reader = ArrayRecordReader(path)
      start = 0  # Using whole file.
      end = reader.num_records()
      reader.close()
    return _ReadInstruction(path, start, end)

  num_threads = _get_flag_value(_GRAIN_NUM_THREADS_COMPUTING_NUM_RECORDS)
  num_workers = min(len(paths), num_threads)
  return _run_in_parallel(
      function=get_read_instruction,
      list_of_kwargs_to_function=[{"path": path} for path in paths],
      num_workers=num_workers,
  )


def _check_group_size(
    filename: epath.PathLike, reader: ArrayRecordReader
) -> None:
  """Logs an error if the group size of the underlying file is not 1."""
  options = reader.writer_options_string()
  # The ArrayRecord Python API does not include methods to parse the options.
  # We will likely move this to C++ soon. In the meantime, we just test if
  # 'group_size:1' is in the options string.
  # The string might be empty for old files written before October 2022.
  if not options:
    return
  group_size = re.search(r"group_size:(\d+),", options)
  if not group_size:
    raise ValueError(
        f"Couldn't detect group_size for {filename}. Extracted writer options:"
        f" {options}."
    )
  if group_size[1] != "1":
    logging.error(
        (
            "File %s was created with group size %s. Grain requires group size"
            " 1 for good performance. Please re-generate your ArrayRecord files"
            " with 'group_size:1'."
        ),
        filename,
        group_size[1],
    )


class ArrayRecordDataSource:
  """Datasource for ArrayRecord files."""

  def __init__(
      self,
      paths: Union[
          PathLikeOrFileInstruction, Sequence[PathLikeOrFileInstruction]
      ],
  ):
    """Creates a new ArrayRecordDataSource object.

    Note on the terminology:
    * record_key: This is the global key of a record in a list of files.
    * position: position of a record within a specific file.

    For example, assume we have two files: my_file-00000-of-00002 and
    my_file-00001-of-00002. If both files have 100 records each, then we can
    read keys in [0, 199] (record_keys can be anywhere in that range).
    record_key 40 will map to the record at position 40 in
    my_file-00000-of-00002 and key 121 would map to the record at position 21
    in my_file-00001-of-00002.

    Args:
      paths: This can be a single path/FileInstruction or list of
        paths/FileInstructions. When you want to read subsets or have a large
        number of files prefer to pass FileInstructions. This makes the
        initialization faster.
    """
    if isinstance(paths, (str, pathlib.Path, FileInstruction)):
      paths = [paths]
    elif isinstance(paths, Sequence):
      # Validate correct format of a sequence path
      if len(paths) <= 0:
        raise ValueError("Paths sequence can not be of 0 length")
      elif not all(
          isinstance(path, (str, pathlib.Path, FileInstruction))
          for path in paths
      ):
        raise ValueError(
            "All elements in a path sequence must be of type: String,"
            " pathlib.Path, or FileInstruction."
        )
    else:
      raise ValueError(
          "Unsupported path format was used. Path format must be "
          "a Sequence, String, pathlib.Path or FileInstruction."
      )
    self._read_instructions = _get_read_instructions(paths)
    self._paths = [ri.filename for ri in self._read_instructions]
    # We open readers lazily when we need to read from them.
    self._readers = [None] * len(self._read_instructions)
    self._num_records = sum(
        map(lambda x: x.num_records, self._read_instructions)
    )
    records_per_instruction = map(
        lambda x: x.num_records, self._read_instructions
    )
    self._prefix_sums = list(itertools.accumulate(records_per_instruction))

  def __enter__(self):
    logging.debug("__enter__ for ArrayRecordDataSource is called.")
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    logging.debug("__exit__ for ArrayRecordDataSource is called.")
    for reader in self._readers:
      if reader:
        reader.close()
    self._readers = [None] * len(self._read_instructions)

  def __len__(self) -> int:
    return self._num_records

  def _reader_idx_and_position(self, record_key: int) -> Tuple[int, int]:
    """Computes reader idx and position of given record key."""
    if record_key < 0 or record_key >= self._num_records:
      raise ValueError("Record key should be in [0, num_records)")
    reader_idx = bisect.bisect_right(self._prefix_sums, record_key)
    records_in_previous_instructions = 0
    if reader_idx > 0:
      records_in_previous_instructions = self._prefix_sums[reader_idx - 1]
    return (
        reader_idx,
        record_key
        - records_in_previous_instructions
        + self._read_instructions[reader_idx].start,
    )

  def _split_keys_per_reader(
      self, record_keys: Sequence[int]
  ) -> Mapping[int, Sequence[Tuple[int, int]]]:
    """Splits record_keys among readers."""
    positions_and_indices = {}
    for idx, record_key in enumerate(record_keys):
      reader_idx, position = self._reader_idx_and_position(record_key)
      if reader_idx in positions_and_indices:
        positions_and_indices[reader_idx].append((position, idx))
      else:
        positions_and_indices[reader_idx] = [(position, idx)]
    return positions_and_indices

  def _ensure_reader_exists(self, reader_idx: int) -> None:
    """Threadsafe method to create corresponding reader if it doesn't exist."""
    if self._readers[reader_idx] is not None:
      return
    filename = self._read_instructions[reader_idx].filename
    reader = ArrayRecordReader(
        filename,
        options="readahead_buffer_size:0",
        file_reader_buffer_size=32768,
    )
    _check_group_size(filename, reader)
    self._readers[reader_idx] = reader

  def __getitem__(self, record_key: int) -> bytes:
    if not isinstance(record_key, int):
      logging.error(
          "Calling ArrayRecordDataSource.__getitem__() with sequence "
          "of record keys (%s) is deprecated. Either pass a single "
          "integer or switch to __getitems__().",
          record_key,
      )
      return self.__getitems__(record_key)
    reader_idx, position = self._reader_idx_and_position(record_key)
    self._ensure_reader_exists(reader_idx)
    return self._readers[reader_idx].read([position])[0]

  def __getitems__(self, record_keys: Sequence[int]) -> Sequence[bytes]:
    def read_records(
        reader_idx: int, reader_positions_and_indices: Sequence[Tuple[int, int]]
    ) -> Sequence[Tuple[Any, int]]:
      """Reads records using the given reader keeping track of the indices."""
      # Initialize readers lazily when we need to read from them.
      self._ensure_reader_exists(reader_idx)
      positions, indices = list(zip(*reader_positions_and_indices))
      records = self._readers[reader_idx].read(positions)  # pytype: disable=attribute-error
      return list(zip(records, indices))

    positions_and_indices = self._split_keys_per_reader(record_keys)
    num_threads = _get_flag_value(_GRAIN_NUM_THREADS_FETCHING_RECORDS)
    num_workers = min(len(positions_and_indices), num_threads)
    list_of_kwargs_to_read_records = []
    for (
        reader_idx,
        reader_positions_and_indices,
    ) in positions_and_indices.items():
      list_of_kwargs_to_read_records.append({
          "reader_idx": reader_idx,
          "reader_positions_and_indices": reader_positions_and_indices,
      })
    records_with_indices: Sequence[Sequence[Tuple[Any, int]]] = (
        _run_in_parallel(
            function=read_records,
            list_of_kwargs_to_function=list_of_kwargs_to_read_records,
            num_workers=num_workers,
        )
    )

    sorted_records = [b""] * len(record_keys)
    for single_reader_records_with_indices in records_with_indices:
      for record, index in single_reader_records_with_indices:
        sorted_records[index] = record
    return sorted_records

  def __getstate__(self):
    logging.debug("__getstate__ for ArrayRecordDataSource is called.")
    state = self.__dict__.copy()
    del state["_readers"]
    return state

  def __setstate__(self, state):
    logging.debug("__setstate__ for ArrayRecordDataSource is called.")
    self.__dict__.update(state)
    # We open readers lazily when we need to read from them. Thus, we don't
    # need to re-open the same files as before pickling.
    self._readers = [None] * len(self._read_instructions)

  def __repr__(self) -> str:
    """Storing a hash of paths since paths can be a very long list."""
    h = hashlib.sha1()
    for p in self._paths:
      h.update(p.encode())
    return f"ArrayRecordDataSource(hash_of_paths={h.hexdigest()})"


def _get_flag_value(flag: flags.FlagHolder[int]) -> int:
  """Retrieves the flag value or the default if run outside of absl."""
  try:
    return flag.value
  except flags.UnparsedFlagAccessError:
    return flag.default
