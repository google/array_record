# Installation

## Requirements

ArrayRecord requires Python 3.10 or later and is available for the following platforms:

| Platform | x86_64 | aarch64 |
|----------|--------|---------|
| Linux    | ✓      | ✓       |
| macOS    | ✗      | ✓       |
| Windows  | ✗      | ✗       |

## Basic Installation

Install ArrayRecord from PyPI:

```bash
pip install array-record-python
```

## Optional Dependencies

### Apache Beam Integration

For large-scale data processing with Apache Beam:

```bash
pip install array-record-python[beam]
```

This includes:
- `apache-beam[gcp]>=2.53.0`
- `google-cloud-storage>=2.11.0`
- `tensorflow>=2.20.0`

### Development and Testing

For development and testing:

```bash
pip install array-record-python[test]
```

This includes:
- `jax`
- `grain`
- `tensorflow>=2.20.0`

## Building from Source

### Prerequisites

- Python 3.10+
- C++17 compatible compiler
- Bazel (for building C++ components)

### Clone and Build

```bash
git clone https://github.com/google/array_record.git
cd array_record
pip install -e .
```

### Building with Bazel

```bash
# Build all targets
bazel build //...

# Run tests
bazel test //...
```

## Verification

Verify your installation:

```python
import array_record.python.array_record_module as ar

# Create a simple test
writer = ar.ArrayRecordWriter('/tmp/test.array_record')
writer.write(b'Hello, ArrayRecord!')
writer.close()

reader = ar.ArrayRecordReader('/tmp/test.array_record')
print(f"Records: {reader.num_records()}")
reader.close()
```

## Common Issues

### Import Errors

If you encounter import errors, ensure you have the correct Python version and platform:

```python
import sys
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")
```

### Missing Dependencies

If you're using ArrayRecord with other libraries like TensorFlow or JAX, install them separately:

```bash
pip install tensorflow>=2.20.0
pip install jax
```

### Platform Compatibility

ArrayRecord currently supports limited platforms. If you're on an unsupported platform, you may need to build from source or use a compatible environment like Docker.
