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
import threading
import typing
from typing import Any, Callable, Iterator, List, Mapping, Protocol, Sequence, SupportsIndex, Tuple, TypeVar, Union

from absl import flags
from absl import logging
from etils import epath

from . import array_record_module

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
  if num_workers == 1 or len(list_of_kwargs_to_function) == 1:
    return [function(**kwargs) for kwargs in list_of_kwargs_to_function]
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
    elif m := re.fullmatch(r"(.*)\[(\d+):(\d+)\]", os.fspath(path)):
      path = m.group(1)
      start = int(m.group(2))
      end = int(m.group(3))
    else:
      path = os.fspath(path)
      reader = array_record_module.ArrayRecordReader(path)
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


def _create_reader(filename: epath.PathLike, additional_reader_options: str):
  """Returns an ArrayRecordReader for the given filename."""
  reader_options = f"readahead_buffer_size:0,{additional_reader_options}"
  return array_record_module.ArrayRecordReader(
      filename,
      options=reader_options,
      file_reader_buffer_size=32768,
  )


def _check_group_size(
    filename: epath.PathLike, reader: array_record_module.ArrayRecordReader
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


class BoundedReaderPool:
  """A pool of readers for a single shard with a dynamic upper bound."""

  def __init__(self, filename: str, options_string: str, max_size: int):
    self._filename = filename
    self._options_string = options_string
    self._max_size = max_size
    self._readers = []
    self._created_count = 0
    self._condition = threading.Condition()
    self._group_size_checked = False
    self.has_read_method = None

  def get(self) -> Any:
    """Gets a reader from the pool, blocking if the cap is reached."""
    create_new = False
    with self._condition:
      # Wait if no idle readers and we reached the cap
      while not self._readers and self._created_count >= self._max_size:
        self._condition.wait()

      if self._readers:
        return self._readers.pop()

      self._created_count += 1
      create_new = True

    if create_new:
      reader = _create_reader(self._filename, self._options_string)
      if self.has_read_method is None:
        self.has_read_method = hasattr(reader, "read")
      if not self._group_size_checked:
        _check_group_size(self._filename, reader)
        self._group_size_checked = True
      return reader

  def put(self, reader: Any) -> None:
    """Returns a reader to the pool."""
    with self._condition:
      self._readers.append(reader)
      self._condition.notify()

  def close_all(self) -> None:
    """Closes all pooled readers."""
    with self._condition:
      for reader in self._readers:
        if reader:
          reader.close()
      self._readers.clear()

  def peek_readers(self) -> List[Any]:
    """Returns the list of readers currently in the pool."""
    with self._condition:
      return list(self._readers)


class ArrayRecordDataSource:
  """Datasource for ArrayRecord files."""

  def __init__(
      self,
      paths: Union[
          PathLikeOrFileInstruction, Sequence[PathLikeOrFileInstruction]
      ],
      reader_options: dict[str, str] | None = None,
      reader_pool_size: int = 1,
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
      reader_options: string of comma-separated options to be passed when
        creating a reader.
      reader_pool_size: Number of readers to pre-allocate in the pool for each
        shard. Default is 1.
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
    if reader_options is None:
      self._reader_options_string = ""
    else:
      reader_options = dict(reader_options)
      if "reader_pool_size" in reader_options:
        reader_pool_size = int(reader_options.pop("reader_pool_size"))
      self._reader_options_string = ",".join(
          [f"{k}:{v}" for k, v in reader_options.items()]
      )
    self._read_instructions = _get_read_instructions(paths)
    self._paths = [ri.filename for ri in self._read_instructions]
    self._reader_pool_size = max(reader_pool_size, 1)
    # We maintain a pool of readers for each shard to ensure thread safety
    # while allowing concurrent reads.
    self._shard_pools = [
        BoundedReaderPool(
            ri.filename, self._reader_options_string, self._reader_pool_size
        )
        for ri in self._read_instructions
    ]
    # We open readers lazily when we need to read from them.

    self._num_records = sum(
        map(lambda x: x.num_records, self._read_instructions)
    )
    records_per_instruction = map(
        lambda x: x.num_records, self._read_instructions
    )
    self._prefix_sums = list(itertools.accumulate(records_per_instruction))
    self._thread_local = threading.local()

  def __enter__(self):
    logging.debug("__enter__ for ArrayRecordDataSource is called.")
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    logging.debug("__exit__ for ArrayRecordDataSource is called.")
    # Clear thread-local cache for the current thread.
    if hasattr(self._thread_local, "reader"):
      if self._thread_local.reader is not None:
        self._thread_local.reader.close()
      self._thread_local.reader = None
      self._thread_local.pool_idx = -1

    for pool in self._shard_pools:
      pool.close_all()

  def __len__(self) -> int:
    return self._num_records

  def __iter__(self) -> Iterator[bytes]:
    for index in range(self._num_records):
      yield self[index]

  def _pool_idx_and_position(
      self, record_key: SupportsIndex
  ) -> Tuple[int, int]:
    """Computes pool idx and position of given record key."""
    record_key = record_key.__index__()
    if record_key < 0 or record_key >= self._num_records:
      raise ValueError("Record key should be in [0, num_records)")
    pool_idx = bisect.bisect_right(self._prefix_sums, record_key)
    records_in_previous_instructions = 0
    if pool_idx > 0:
      records_in_previous_instructions = self._prefix_sums[pool_idx - 1]
    return (
        pool_idx,
        record_key
        - records_in_previous_instructions
        + self._read_instructions[pool_idx].start,
    )

  def _split_keys_per_pool(
      self, record_keys: Sequence[SupportsIndex]
  ) -> Mapping[int, Sequence[Tuple[int, int]]]:
    """Splits record_keys among pools."""
    positions_and_indices = {}
    for idx, record_key in enumerate(record_keys):
      pool_idx, position = self._pool_idx_and_position(record_key)
      if pool_idx in positions_and_indices:
        positions_and_indices[pool_idx].append((position, idx))
      else:
        positions_and_indices[pool_idx] = [(position, idx)]
    return positions_and_indices

  def _get_reader(self, pool_idx: int) -> Any:
    """Gets a reader from the single-slot thread-local cache or the pool."""
    if not hasattr(self._thread_local, "pool_idx"):
      self._thread_local.pool_idx = -1
      self._thread_local.reader = None

    if self._thread_local.pool_idx == pool_idx:
      return self._thread_local.reader

    # Return previous reader to pool if we have one
    if self._thread_local.reader is not None:
      self._shard_pools[self._thread_local.pool_idx].put(
          self._thread_local.reader
      )

    # Get new reader and cache it
    reader = self._shard_pools[pool_idx].get()
    self._thread_local.pool_idx = pool_idx
    self._thread_local.reader = reader
    return reader

  def _release_reader(self, pool_idx: int, reader: Any) -> None:
    """No-op: we keep it cached in _get_reader until the next request."""
    pass

  def _read_record(
      self, reader: Any, pool: BoundedReaderPool, position: int
  ) -> bytes:
    """Helper to read a record using the best available method."""
    if pool.has_read_method:
      return reader.read([position])[0]
    if hasattr(reader, "read_record"):
      return reader.read_record(position)
    return reader[position]

  def __getitem__(self, record_key: SupportsIndex) -> bytes:
    pool_idx, position = self._pool_idx_and_position(record_key)
    reader = self._get_reader(pool_idx)
    try:
      return self._read_record(reader, self._shard_pools[pool_idx], position)
    finally:
      self._release_reader(pool_idx, reader)

  def __getitems__(
      self, record_keys: Sequence[SupportsIndex]
  ) -> Sequence[bytes]:

    def read_records(
        pool_idx: int, reader_positions_and_indices: Sequence[Tuple[int, int]]
    ) -> Sequence[Tuple[Any, int]]:
      """Reads records using the given reader keeping track of the indices."""
      reader = self._get_reader(pool_idx)
      pool = self._shard_pools[pool_idx]
      try:
        records = []
        for position, _ in reader_positions_and_indices:
          records.append(self._read_record(reader, pool, position))
        indices = [idx for _, idx in reader_positions_and_indices]
        return list(zip(records, indices))
      finally:
        self._release_reader(pool_idx, reader)

    # Group record keys by pool/shard to maximize reader reuse.
    positions_and_indices = self._split_keys_per_pool(record_keys)
    num_threads = _get_flag_value(_GRAIN_NUM_THREADS_FETCHING_RECORDS)
    # Parallelize reads across shards using the available threads.
    num_workers = min(len(positions_and_indices), num_threads)
    list_of_kwargs_to_read_records = []
    for (
        pool_idx,
        reader_positions_and_indices,
    ) in positions_and_indices.items():
      list_of_kwargs_to_read_records.append({
          "pool_idx": pool_idx,
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
    state.pop("_shard_pools", None)
    state.pop("_reader_pools", None)
    state.pop("_thread_local", None)
    return state

  def __setstate__(self, state):
    logging.debug("__setstate__ for ArrayRecordDataSource is called.")
    self.__dict__.update(state)
    self._shard_pools = [
        BoundedReaderPool(
            ri.filename,
            self._reader_options_string,
            getattr(self, "_reader_pool_size", 1),
        )
        for ri in self._read_instructions
    ]
    self._thread_local = threading.local()
    # We open readers lazily when we need to read from them.

  def _peek_readers(self) -> List[Any]:
    """Returns a list of readers (one per shard) or None (for testing only)."""
    readers = []
    for i, pool in enumerate(self._shard_pools):
      reader = None
      if (
          hasattr(self._thread_local, "pool_idx")
          and self._thread_local.pool_idx == i
      ):
        reader = self._thread_local.reader
      if reader is None:
        pooled_readers = pool.peek_readers()
        reader = pooled_readers[-1] if pooled_readers else None
      readers.append(reader)
    return readers

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
