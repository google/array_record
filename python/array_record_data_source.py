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

  def __getitem__(self, record_key: SupportsIndex) -> T:
    ...

  def __getitems__(self, record_keys: Sequence[SupportsIndex]) -> Sequence[T]:
    ...
```
"""

import bisect
import collections
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

T = TypeVar("T")


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
ArrayRecordDataSourcePaths = Union[
    PathLikeOrFileInstruction, Sequence[PathLikeOrFileInstruction]
]


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
_ARRAY_RECORD_READER_POOL_SIZE = flags.DEFINE_integer(
    "array_record_reader_pool_size",
    None,
    "The default reader pool size per shard in ArrayRecordDataSource.",
)
_ARRAY_RECORD_GCS_READAHEAD_BUFFER_SIZE_BYTES = flags.DEFINE_integer(
    "array_record_gcs_readahead_buffer_size_bytes",
    4 * 1024 * 1024,
    "Default ArrayRecord readahead_buffer_size in bytes for gs:// paths.",
)


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
        raise completed_future.exception()  # pyrefly: ignore[bad-raise]
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


def _parse_options_string(options_string: str) -> dict[str, str]:
  """Parses a comma-separated 'key:value' options string into a dict."""
  parsed: dict[str, str] = {}
  if not options_string:
    return parsed
  for item in options_string.split(","):
    if not item:
      continue
    if ":" in item:
      key, value = item.split(":", 1)
      parsed[key] = value
    else:
      parsed[item] = ""
  return parsed


def _format_options_dict(options: Mapping[str, str]) -> str:
  """Formats a dict of reader options into a comma-separated string."""
  return ",".join(
      f"{key}:{value}" if value else key for key, value in options.items()
  )


class _StatefulArrayRecordReader:
  """Adapter that routes sequential or small-shard reads to stateful readahead.

  Uses stateful `seek(position)` + `read()` (readahead) when:
  - the entire shard fits within the readahead buffer
  (`_is_likely_small_shard`),
  - `position` falls within the currently cached readahead window
    (`_buf_start <= position < _buf_end`), or
  - `position` is part of a sequential scan (`position == _last_pos + 1`,
    initial single-record read, or sequential batch stream `_seq_mode`).
  Otherwise (non-sequential random reads on shards larger than the readahead
  buffer), routes to stateless `read([position])` / `read(positions)` so point
  reads release the Python GIL, avoid 4 MiB read amplification, and do not
  evict an active readahead buffer in `state_->current_decoders`.
  """

  def __init__(self, reader: Any, readahead_bytes: int = 4 * 1024 * 1024):
    self._reader = reader
    self._readahead_bytes = readahead_bytes
    self._num_records: int | None = None
    self._whole_shard_fits: bool | None = None
    self._records_per_buf: int = 1
    self._last_pos: int | None = None
    self._buf_range: range | None = None
    self._seq_mode: bool = False

  def _init_shard_stats(self, sample_record_len: int) -> None:
    """Initializes shard record count and readahead buffer capacity."""
    if self._whole_shard_fits is not None:
      return
    num_records = self._reader.num_records()
    if isinstance(num_records, int) and num_records > 0:
      self._num_records = num_records
      rec_len = max(1, sample_record_len)
      self._records_per_buf = max(1, self._readahead_bytes // rec_len)
      self._whole_shard_fits = self._num_records * rec_len <= int(
          self._readahead_bytes * 1.1
      )
    else:
      self._whole_shard_fits = False

  def _is_likely_small_shard(self) -> bool:
    """Returns True if the shard is estimated to fit in the readahead buffer."""
    if self._whole_shard_fits is not None:
      return self._whole_shard_fits
    num_records = self._reader.num_records()
    if isinstance(num_records, int) and 0 < num_records <= max(
        1, self._readahead_bytes // 65536
    ):
      return True
    return False

  def _in_buf(self, position: int) -> bool:
    """Returns True if position falls within the active readahead range."""
    return self._buf_range is not None and position in self._buf_range

  def read_record(self, position: int) -> bytes:
    """Reads a single record using adaptive stateful or stateless access."""
    is_in_buf = self._in_buf(position)
    is_contiguous = (
        self._last_pos is not None and position == self._last_pos + 1
    )
    is_initial = self._last_pos is None
    if is_contiguous:
      self._seq_mode = True

    if (
        self._is_likely_small_shard()
        or is_in_buf
        or is_contiguous
        or is_initial
        or self._seq_mode
    ):
      if not is_contiguous and not is_in_buf and not is_initial:
        # Requiring the next record to be contiguous to keep _seq_mode active
        # ensures random single-record access immediately switches to stateless.
        self._seq_mode = False
      self._reader.seek(position)
      data = self._reader.read()
      self._init_shard_stats(len(data))
      if not is_in_buf:
        if self._whole_shard_fits:
          self._buf_range = range(0, self._num_records or self._records_per_buf)
        else:
          aligned_start = (
              position // self._records_per_buf
          ) * self._records_per_buf
          self._buf_range = range(
              aligned_start, aligned_start + self._records_per_buf
          )
    else:
      data = self._reader.read([position])[0]
      self._init_shard_stats(len(data))
    self._last_pos = position
    return data

  def read_records(self, positions: Sequence[int]) -> list[bytes]:
    """Reads a batch of records using adaptive readahead or parallel point reads."""
    if not positions:
      return []
    has_seq_locality = len(positions) > 1 and any(
        positions[i + 1] == positions[i] + 1 for i in range(len(positions) - 1)
    )
    single_in_buf_or_seq = len(positions) == 1 and (
        self._in_buf(positions[0])
        or (self._last_pos is not None and positions[0] == self._last_pos + 1)
        or (self._last_pos is None and positions[0] == 0)
    )
    if (
        self._is_likely_small_shard()
        or has_seq_locality
        or single_in_buf_or_seq
    ):
      if has_seq_locality and not self._in_buf(positions[0]):
        self._last_pos = positions[0] - 1
      return [self.read_record(p) for p in positions]

    res = list(self._reader.read(list(positions)))
    if res:
      self._init_shard_stats(len(res[0]))
    self._last_pos = positions[-1]
    return res

  def __getattr__(self, name: str) -> Any:
    return getattr(self._reader, name)


def _create_gcs_reader(filename: str, additional_reader_options: str) -> Any:
  """Returns an ArrayRecordReader with GCS readahead defaults for `gs://`."""
  user_opts = _parse_options_string(additional_reader_options)
  readahead_bytes = _get_flag_value(
      _ARRAY_RECORD_GCS_READAHEAD_BUFFER_SIZE_BYTES
  )
  readahead_str = user_opts.get("readahead_buffer_size", str(readahead_bytes))
  # Default max_parallelism to 1 instead of the C++ thread-pool default (16).
  # With max_parallelism=16, every buffer miss in ReadAheadFromBuffer schedules
  # 16 parallel 4 MiB Range GETs (64 MiB), which are discarded on non-sequential
  # seeks (causing up to 16x GCS QPS and memory amplification).
  #
  # IMPORTANT: GCS readahead relies on batched read pushdown (`__getitems__`
  # wired to PyGrain's `SupportsBatchedReadRandomAccessDataSource._getitems`).
  # On workloads with large shards (shard_size > readahead_buffer_size),
  # multi-threaded prefetching without batched read pushdown borrows and returns
  # a pooled reader for one record at a time (`__getitem__`), causing concurrent
  # threads reading different batch offsets in the same shard to overwrite each
  # other's `buffer_idx` and thrash the readahead buffer. With batched read
  # pushdown, `read_records` in `__getitems__` holds the borrowed reader for the
  # entire batch of records in that shard so the readahead buffer is consumed
  # before the reader returns to `_BoundedReaderPool`. Do not remove batched
  # read pushdown while GCS readahead is enabled.
  merged_opts = {
      "readahead_buffer_size": readahead_str,
      "max_parallelism": "1",
  }
  merged_opts.update(user_opts)
  reader = array_record_module.ArrayRecordReader(
      filename,
      options=_format_options_dict(merged_opts),
  )
  if readahead_str == "0":
    return reader
  try:
    parsed_readahead = int(readahead_str)
  except ValueError:
    parsed_readahead = readahead_bytes
  return _StatefulArrayRecordReader(reader, parsed_readahead)


def _create_reader(filename: epath.PathLike, additional_reader_options: str):
  """Returns an ArrayRecordReader for the given filename."""
  filename_str = os.fspath(filename)
  if filename_str.startswith("gs://"):
    return _create_gcs_reader(filename_str, additional_reader_options)
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


class _BoundedReaderPoolBorrowContext:
  """Context manager for borrowing a reader safely from a _BoundedReaderPool.

  Ensures that the borrowed reader is always returned to the pool, even if
  exceptions are raised within the borrowing thread's critical section.
  """

  def __init__(self, pool: "_BoundedReaderPool", sticky: bool = True):
    self._pool = pool
    self._sticky = sticky
    self._reader = None

  def __enter__(self) -> Any:
    if not self._sticky:
      self._pool._batch_lock.acquire()  # pylint: disable=protected-access
    try:
      self._reader = self._pool.get(sticky=self._sticky)
      return self._reader
    except Exception:
      if not self._sticky:
        self._pool._batch_lock.release()  # pylint: disable=protected-access
      raise

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    try:
      if self._reader is not None:
        self._pool.put(self._reader, sticky=self._sticky)
    finally:
      if not self._sticky:
        self._pool._batch_lock.release()  # pylint: disable=protected-access


class _BoundedReaderPool:
  """A semaphore-throttled thread-safe connection pool for a single shard.

  This pool maintains and recycles expensive, non-thread-safe reader instances
  (such as `ArrayRecordReader`) to enable parallel reads without lock
  contention.

  This is a private class. Since it is not RAII, directly calling `get()` and
  `put()` is subject to a risk of deadlock upon exception handling. Callers
  MUST use the context-manager based borrowing pattern instead:
    with pool.borrow() as reader:
      # Perform read operations

  Concurrency Model (Permit/Ownership Flow):
    1. A thread calls `get()` to acquire a reader. This blocks if the number
       of active readers has reached `max_size` (acquires a semaphore permit).
    2. The calling thread is now the exclusive owner of the reader and can
       safely perform non-thread-safe read operations on it.
    3. Once reading is complete, the thread MUST call `put(reader)` to return
       the reader. This recycles the reader and releases the connection slot
       (releases the semaphore permit).

    To guarantee safe lease return, callers are strongly encouraged to use the
    context-manager based borrowing pattern:
      with pool.borrow() as reader:
        # Perform read operations

    WARNING: Failing to return a borrowed reader via `put()` will permanently
    leak a semaphore permit, eventually causing all subsequent `get()` calls
    to deadlock when the cap is reached.

  Teardown & Lifecycle:
    Calling `close_all()` marks the pool as closed and immediately closes all
    idle readers. Any outstanding borrowed readers will be closed immediately
    upon their return via `put()`, ensuring zero file descriptor leaks during
    concurrent shutdown sequences.
  """

  def __init__(self, filename: str, options_string: str, max_size: int = 1):
    self._filename = filename
    self._options_string = options_string
    self._max_size = max_size
    self._readers = collections.deque()
    self._readers_by_tid: dict[int, Any] = {}
    self._total_created = 0
    # Use BoundedSemaphore to strictly enforce the max_size cap
    self._semaphore = threading.BoundedSemaphore(max_size)
    self._batch_lock = threading.Lock()
    self._lock = threading.Lock()
    self._group_size_checked = False
    self._closed = False

  def get(self, sticky: bool = True) -> Any:
    """Acquires a reader from the pool, blocking if the active reader cap is reached.

    If the pool is empty but the cap has not been reached, a new reader is
    instantiated. If the pool already has idle readers, one is returned
    instantly without blocking.

    Args:
      sticky: Whether to prefer thread-sticky reader affinity.

    Returns:
      A reader instance. Callers must use the borrow() context manager.

    Raises:
      RuntimeError: If the reader pool is already closed.
    """
    self._semaphore.acquire()
    tid = threading.get_ident()

    with self._lock:
      if self._closed:
        self._semaphore.release()
        raise RuntimeError(
            f"Cannot get reader from closed pool: {self._filename}"
        )
      if tid in self._readers_by_tid:
        return self._readers_by_tid.pop(tid)
      if self._readers:
        return self._readers.popleft()
      if not sticky and self._readers_by_tid:
        _, reader = self._readers_by_tid.popitem()
        return reader
      if self._total_created >= self._max_size and self._readers_by_tid:
        _, reader = self._readers_by_tid.popitem()
        return reader
      self._total_created += 1

    # Create a new reader outside lock so concurrent threads open in parallel.
    reader = None
    try:
      reader = _create_reader(self._filename, self._options_string)
      with self._lock:
        if self._closed:
          if reader and hasattr(reader, "close"):
            reader.close()
          self._total_created -= 1
          raise RuntimeError(
              f"Cannot get reader from closed pool: {self._filename}"
          )
        if not self._group_size_checked:
          _check_group_size(self._filename, reader)
          self._group_size_checked = True
      return reader
    except Exception:
      with self._lock:
        self._total_created -= 1
      if reader and hasattr(reader, "close"):
        reader.close()
      self._semaphore.release()
      raise

  def put(self, reader: Any, sticky: bool = True) -> None:
    """Returns a reader to the pool, recycling it for future operations.

    If the pool has been closed in the interim, the reader is closed
    immediately.

    Args:
      reader: The reader instance previously obtained from `get()`.
      sticky: Whether to store the reader under the calling thread's ID.
    """
    with self._lock:
      if self._closed:
        # If the pool was closed while the reader was borrowed, close it
        # immediately.
        if reader and hasattr(reader, "close"):
          reader.close()
        self._semaphore.release()
        return

      tid = threading.get_ident()
      if sticky and tid not in self._readers_by_tid:
        self._readers_by_tid[tid] = reader
      else:
        self._readers.append(reader)
    self._semaphore.release()

  def borrow(self, sticky: bool = True) -> _BoundedReaderPoolBorrowContext:
    """Returns a context manager to borrow a reader safely.

    Usage:
      with pool.borrow() as reader:
        # Perform read operations

    Args:
      sticky: Whether to use thread-sticky reader affinity.
    """
    return _BoundedReaderPoolBorrowContext(self, sticky=sticky)

  def close_all(self) -> None:
    """Closes all pooled readers and prevents future allocations."""
    with self._lock:
      self._closed = True
      readers_to_close = list(self._readers) + list(
          self._readers_by_tid.values()
      )
      self._readers.clear()
      self._readers_by_tid.clear()

    for reader in readers_to_close:
      if reader and hasattr(reader, "close"):
        reader.close()

  def peek_readers(self) -> List[Any]:
    """Returns the list of readers (for testing only)."""
    with self._lock:
      return list(self._readers) + list(self._readers_by_tid.values())


class ArrayRecordDataSource:
  """Datasource for ArrayRecord files using a Lock-Free Connection Pool."""

  def __init__(
      self,
      paths: Union[
          PathLikeOrFileInstruction, Sequence[PathLikeOrFileInstruction]
      ],
      reader_options: dict[str, str] | None = None,
      reader_pool_size: int | None = None,
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
      reader_pool_size: The maximum number of readers to keep open per shard.
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
      self._reader_options_string = ",".join(
          [f"{k}:{v}" for k, v in reader_options.items()]
      )
    self._read_instructions = _get_read_instructions(paths)
    self._paths = [ri.filename for ri in self._read_instructions]
    default_pool_size = (
        16 if any(p.startswith("gs://") for p in self._paths) else 1
    )
    self._reader_pool_size = (
        reader_pool_size
        or _get_flag_value(_ARRAY_RECORD_READER_POOL_SIZE)  # pyrefly: ignore[bad-argument-type]
        or default_pool_size
    )

    # Lock-free connection pool per shard
    self._shard_pools = [
        _BoundedReaderPool(
            ri.filename, self._reader_options_string, self._reader_pool_size
        )
        for ri in self._read_instructions
    ]

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
    for pool in self._shard_pools:
      pool.close_all()

  def __len__(self) -> int:
    return self._num_records

  def __iter__(self) -> Iterator[bytes]:
    for index in range(self._num_records):
      yield self[index]

  def _reader_idx_and_position(
      self, record_key: SupportsIndex
  ) -> Tuple[int, int]:
    """Computes reader idx and position of given record key."""
    record_key = record_key.__index__()
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
      self, record_keys: Sequence[SupportsIndex]
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

  def _read_record(self, reader: Any, position: int) -> bytes:
    """Helper to read a record using the best available method."""
    if hasattr(reader, "read_record"):
      return reader.read_record(position)
    if hasattr(reader, "read"):
      return reader.read([position])[0]
    return reader[position]

  def __getitem__(self, record_key: SupportsIndex) -> bytes:
    pool_idx, position = self._reader_idx_and_position(record_key)
    with self._shard_pools[pool_idx].borrow(sticky=True) as reader:
      return self._read_record(reader, position)

  def __getitems__(
      self, record_keys: Sequence[SupportsIndex]
  ) -> Sequence[bytes]:

    def read_records(
        pool_idx: int, reader_positions_and_indices: Sequence[Tuple[int, int]]
    ) -> Sequence[Tuple[Any, int]]:
      """Reads records using the given reader keeping track of the indices."""
      # Holding the borrowed reader for the entire batch of positions in this
      # shard is required for GCS readahead (`_StatefulArrayRecordReader`) on
      # large shards so concurrent threads do not interleave per-record seeks
      # and evict the readahead buffer before the batch finishes reading it.
      with self._shard_pools[pool_idx].borrow(sticky=False) as reader:
        positions = [position for position, _ in reader_positions_and_indices]
        if hasattr(reader, "read_records"):
          records = reader.read_records(positions)
        else:
          records = [self._read_record(reader, pos) for pos in positions]
        indices = [idx for _, idx in reader_positions_and_indices]
        return list(zip(records, indices))

    positions_and_indices = self._split_keys_per_reader(record_keys)
    num_threads = _get_flag_value(_GRAIN_NUM_THREADS_FETCHING_RECORDS)
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
    return state

  def __setstate__(self, state):
    logging.debug("__setstate__ for ArrayRecordDataSource is called.")
    self.__dict__.update(state)
    # We open readers lazily when we need to read from them. Thus, we don't
    # need to re-open the same files as before pickling.
    default_pool_size = (
        16
        if any(p.startswith("gs://") for p in getattr(self, "_paths", []))
        else 1
    )
    self._shard_pools = [
        _BoundedReaderPool(
            ri.filename,
            self._reader_options_string,
            getattr(self, "_reader_pool_size", default_pool_size),
        )
        for ri in self._read_instructions
    ]

  def __repr__(self) -> str:
    """Storing a hash of paths since paths can be a very long list."""
    h = hashlib.sha1()
    for p in self._paths:
      h.update(p.encode())
    return f"ArrayRecordDataSource(hash_of_paths={h.hexdigest()})"

  def _peek_readers(self) -> List[Any]:
    """Returns a list of readers (one per shard) or None (for testing only)."""
    readers = []
    for pool in self._shard_pools:
      pooled_readers = pool.peek_readers()
      readers.append(pooled_readers[-1] if pooled_readers else None)
    return readers


def _get_flag_value(flag: flags.FlagHolder[int]) -> int:
  """Retrieves the flag value or the default if run outside of absl."""
  try:
    return flag.value
  except flags.UnparsedFlagAccessError:
    return flag.default
