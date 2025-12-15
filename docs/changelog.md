# Changelog

All notable changes to ArrayRecord will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive Sphinx documentation
- Performance optimization guides
- Extended examples for machine learning workflows
- Multi-modal data storage examples

### Changed
- Improved documentation structure and navigation
- Enhanced API reference documentation

### Fixed
- Documentation formatting and cross-references

## [0.8.1] - 2024-01-15

### Added
- Support for Python 3.13
- Enhanced error handling in Python bindings
- Improved memory management in C++ components

### Changed
- Updated dependencies to latest versions
- Optimized default buffer sizes for better performance

### Fixed
- Memory leaks in certain edge cases
- Compatibility issues with newer TensorFlow versions
- Build issues on macOS with Apple Silicon

### Security
- Updated dependencies to address security vulnerabilities

## [0.8.0] - 2023-12-01

### Added
- New `index_storage_option` for memory optimization
- Support for offloaded index storage to reduce memory usage
- Enhanced parallel reading capabilities
- Improved random access performance

### Changed
- **BREAKING**: Changed default group size from 32768 to 65536
- Improved compression ratio with better chunk organization
- Enhanced Apache Beam integration with better error handling

### Fixed
- Race conditions in parallel writing scenarios
- Incorrect record counts in certain file configurations
- Memory usage spikes during large file processing

### Deprecated
- Old-style option parsing (will be removed in v1.0.0)

## [0.7.2] - 2023-10-15

### Added
- Support for transposition with custom bucket sizes
- New writer option `transpose_bucket_size`
- Enhanced debugging capabilities

### Fixed
- Transposition issues with variable-length records
- Performance regression in sequential access
- Build failures on certain Linux distributions

## [0.7.1] - 2023-09-01

### Fixed
- Critical bug in record indexing for large files
- Memory corruption in multi-threaded scenarios
- Incorrect file size calculations

### Changed
- Improved error messages for better debugging
- Enhanced logging in debug builds

## [0.7.0] - 2023-08-15

### Added
- Apache Beam integration for large-scale data processing
- Support for Google Cloud Storage via Beam DoFns
- Pre-built pipelines for TFRecord conversion
- Command-line utilities for data conversion

### Changed
- Restructured project layout for better modularity
- Improved build system with better dependency management

### Fixed
- Threading issues in concurrent read scenarios
- File handle leaks in error conditions

## [0.6.3] - 2023-07-01

### Added
- Support for macOS on Apple Silicon (aarch64)
- Enhanced protocol buffer support
- New parallel reading methods with index support

### Fixed
- Build issues on newer macOS versions
- Performance degradation in certain access patterns
- Memory alignment issues on ARM processors

## [0.6.2] - 2023-06-01

### Fixed
- Critical bug in chunk boundary calculations
- Data corruption issues in specific compression scenarios
- Build compatibility with newer Bazel versions

### Changed
- Improved error handling and reporting
- Enhanced validation of writer options

## [0.6.1] - 2023-05-15

### Added
- Support for Python 3.12
- Enhanced compression options with Zstd support
- New writer option `saturation_delay_ms`

### Changed
- Improved default compression settings
- Better memory usage patterns in large file scenarios

### Fixed
- Compatibility issues with newer Python versions
- Memory leaks in certain error scenarios

## [0.6.0] - 2023-04-01

### Added
- **New Feature**: Parallel writing capabilities
- Support for custom thread pools
- Enhanced random access performance
- New reader options for optimization

### Changed
- **BREAKING**: Modified ArrayRecordReader API for better performance
- Improved chunk indexing for faster seeks
- Enhanced compression efficiency

### Fixed
- Race conditions in multi-threaded environments
- Incorrect record ordering in parallel scenarios

### Deprecated
- Legacy reader initialization methods

## [0.5.2] - 2023-03-01

### Fixed
- Critical data corruption bug in specific compression scenarios
- Build issues on CentOS and RHEL systems
- Memory usage optimization for large files

### Changed
- Improved error messages and diagnostics
- Enhanced validation of input parameters

## [0.5.1] - 2023-02-15

### Added
- Support for Python 3.11
- Enhanced debugging and profiling capabilities
- New utility functions for file inspection

### Fixed
- Performance regression in sequential reading
- Build compatibility with newer GCC versions
- Memory alignment issues on certain architectures

## [0.5.0] - 2023-01-15

### Added
- **Major Feature**: Random access by record index
- Support for multiple compression algorithms (Brotli, Zstd, Snappy)
- Configurable group sizes for performance tuning
- Enhanced Python API with context manager support

### Changed
- **BREAKING**: New file format with improved indexing
- Restructured C++ API for better performance
- Improved memory usage patterns

### Fixed
- Data integrity issues in certain edge cases
- Build system improvements for better portability

### Migration Guide
- Files created with v0.4.x need to be recreated with v0.5.0+
- Update code to use new ArrayRecordDataSource API
- Review compression settings for optimal performance

## [0.4.3] - 2022-12-01

### Fixed
- Memory leaks in long-running processes
- Compatibility with TensorFlow 2.11+
- Build issues on Ubuntu 22.04

### Changed
- Improved error handling in Python bindings
- Enhanced logging for debugging

## [0.4.2] - 2022-11-01

### Added
- Support for Linux aarch64 (ARM64)
- Enhanced error reporting and diagnostics

### Fixed
- Segmentation faults in certain error conditions
- Build compatibility with newer Python versions

## [0.4.1] - 2022-10-15

### Fixed
- Critical bug in record boundary detection
- Performance issues with small records
- Build system improvements

### Changed
- Optimized buffer management for better performance
- Improved documentation and examples

## [0.4.0] - 2022-10-01

### Added
- **Initial public release**
- Core ArrayRecord format implementation
- Python bindings with basic read/write functionality
- C++ API for high-performance applications
- Support for Linux x86_64
- Basic compression support with Brotli

### Features
- Sequential reading and writing
- Configurable compression levels
- Thread-safe operations
- Integration with Riegeli format

---

## Version Support

| Version | Python Support | Platform Support | Status |
|---------|----------------|------------------|---------|
| 0.8.x   | 3.10, 3.11, 3.12, 3.13 | Linux (x86_64, aarch64), macOS (aarch64) | Active |
| 0.7.x   | 3.10, 3.11, 3.12 | Linux (x86_64, aarch64), macOS (aarch64) | Security fixes only |
| 0.6.x   | 3.10, 3.11, 3.12 | Linux (x86_64, aarch64), macOS (aarch64) | End of life |
| 0.5.x   | 3.10, 3.11 | Linux (x86_64, aarch64) | End of life |
| < 0.5   | 3.8, 3.9, 3.10 | Linux (x86_64) | End of life |

## Migration Notes

### Upgrading from 0.7.x to 0.8.x

- No breaking changes
- New features available but optional
- Recommended to update for performance improvements

### Upgrading from 0.6.x to 0.7.x

- Apache Beam integration requires separate installation: `pip install array_record[beam]`
- New command-line tools available
- Existing code continues to work without changes

### Upgrading from 0.5.x to 0.6.x

- Python 3.12 support added
- New compression options available
- Performance improvements in random access

### Upgrading from 0.4.x to 0.5.x

- **File format changed**: Files need to be recreated
- **API changes**: Update to new ArrayRecordDataSource
- **Performance**: Significant improvements in random access
- **Compression**: New algorithms available

## Contributing

See [CONTRIBUTING.md](contributing.md) for information about contributing to ArrayRecord.

## License

ArrayRecord is licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
