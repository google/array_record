# Contributing to ArrayRecord

We welcome contributions to ArrayRecord! This document provides guidelines for contributing to the project.

## Getting Started

### Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/google/array_record.git
   cd array_record
   ```

2. **Set up development environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e .[test,beam]
   ```

3. **Install build dependencies**:
   ```bash
   # Install Bazel (for C++ components)
   # Follow instructions at https://bazel.build/install
   
   # Verify installation
   bazel version
   ```

### Building from Source

```bash
# Build all targets
bazel build //...

# Build specific components
bazel build //python:array_record_module
bazel build //cpp:array_record_reader
bazel build //beam:all
```

### Running Tests

```bash
# Run all tests
bazel test //...

# Run specific test suites
bazel test //python:array_record_module_test
bazel test //cpp:array_record_reader_test
bazel test //beam:all

# Run Python tests with pytest
pytest python/ -v
```

## Code Style and Standards

### Python Code Style

We follow PEP 8 with some modifications. Use the following tools:

```bash
# Install formatting tools
pip install black isort pylint mypy

# Format code
black .
isort .

# Check style
pylint array_record/
mypy array_record/
```

### C++ Code Style

We follow the Google C++ Style Guide. Use clang-format:

```bash
# Format C++ code
find cpp/ -name "*.cc" -o -name "*.h" | xargs clang-format -i
```

### Documentation Style

- Use Google-style docstrings for Python
- Use Doxygen-style comments for C++
- Write clear, concise documentation
- Include code examples where helpful

## Types of Contributions

### Bug Reports

When reporting bugs, please include:

1. **Clear description** of the issue
2. **Steps to reproduce** the problem
3. **Expected vs. actual behavior**
4. **Environment information** (OS, Python version, ArrayRecord version)
5. **Minimal code example** that demonstrates the issue

**Bug Report Template**:
```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Create ArrayRecord file with...
2. Read using...
3. See error...

**Expected behavior**
A clear description of what you expected to happen.

**Environment:**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.10.8]
- ArrayRecord version: [e.g., 0.8.1]
- Installation method: [e.g., pip, source]

**Additional context**
Add any other context about the problem here.
```

### Feature Requests

For feature requests, please include:

1. **Clear description** of the proposed feature
2. **Use case** and motivation
3. **Proposed API** (if applicable)
4. **Implementation considerations**

### Code Contributions

#### Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the coding standards
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Run tests** to ensure everything works
6. **Submit a pull request**

#### Pull Request Guidelines

- **One feature per PR**: Keep changes focused and atomic
- **Clear commit messages**: Use descriptive commit messages
- **Add tests**: All new code should have corresponding tests
- **Update docs**: Update documentation for API changes
- **Follow style guides**: Ensure code follows project standards

#### Example Workflow

```bash
# 1. Fork and clone your fork
git clone https://github.com/yourusername/array_record.git
cd array_record

# 2. Create feature branch
git checkout -b feature/my-new-feature

# 3. Make changes and commit
git add .
git commit -m "Add new feature: description of changes"

# 4. Run tests
bazel test //...

# 5. Push to your fork
git push origin feature/my-new-feature

# 6. Create pull request on GitHub
```

## Development Guidelines

### Adding New Features

#### Python Features

1. **Add implementation** in appropriate module
2. **Write comprehensive tests**:
   ```python
   # python/my_feature_test.py
   import unittest
   from array_record.python import my_feature
   
   class MyFeatureTest(unittest.TestCase):
       def test_basic_functionality(self):
           # Test implementation
           pass
   ```

3. **Update BUILD files**:
   ```python
   # python/BUILD
   py_test(
       name = "my_feature_test",
       srcs = ["my_feature_test.py"],
       deps = [":my_feature"],
   )
   ```

4. **Add documentation**:
   ```python
   def my_function(arg1: str, arg2: int) -> bool:
       """Brief description of the function.
       
       Args:
           arg1: Description of arg1.
           arg2: Description of arg2.
           
       Returns:
           Description of return value.
           
       Raises:
           ValueError: When invalid arguments are provided.
           
       Example:
           >>> result = my_function("test", 42)
           >>> print(result)
           True
       """
   ```

#### C++ Features

1. **Add header file**:
   ```cpp
   // cpp/my_feature.h
   #ifndef ARRAY_RECORD_CPP_MY_FEATURE_H_
   #define ARRAY_RECORD_CPP_MY_FEATURE_H_
   
   namespace array_record {
   
   class MyFeature {
    public:
     // Public interface
   };
   
   }  // namespace array_record
   
   #endif  // ARRAY_RECORD_CPP_MY_FEATURE_H_
   ```

2. **Add implementation**:
   ```cpp
   // cpp/my_feature.cc
   #include "cpp/my_feature.h"
   
   namespace array_record {
   
   // Implementation
   
   }  // namespace array_record
   ```

3. **Add tests**:
   ```cpp
   // cpp/my_feature_test.cc
   #include "cpp/my_feature.h"
   #include "gtest/gtest.h"
   
   namespace array_record {
   
   TEST(MyFeatureTest, BasicFunctionality) {
     // Test implementation
   }
   
   }  // namespace array_record
   ```

4. **Update BUILD files**:
   ```python
   # cpp/BUILD
   cc_library(
       name = "my_feature",
       srcs = ["my_feature.cc"],
       hdrs = ["my_feature.h"],
       deps = [
           # Dependencies
       ],
   )
   
   cc_test(
       name = "my_feature_test",
       srcs = ["my_feature_test.cc"],
       deps = [
           ":my_feature",
           "@googletest//:gtest_main",
       ],
   )
   ```

### Documentation

#### API Documentation

- **Python**: Use Google-style docstrings
- **C++**: Use Doxygen-style comments
- **Include examples** in docstrings
- **Document all public APIs**

#### User Documentation

- **Update relevant guides** when adding features
- **Add examples** to the examples section
- **Update performance guide** if applicable
- **Keep documentation up to date** with code changes

### Testing

#### Test Categories

1. **Unit tests**: Test individual components
2. **Integration tests**: Test component interactions
3. **Performance tests**: Benchmark critical paths
4. **Compatibility tests**: Test across Python versions and platforms

#### Test Guidelines

- **Comprehensive coverage**: Test normal and edge cases
- **Clear test names**: Describe what is being tested
- **Independent tests**: Each test should be self-contained
- **Fast execution**: Keep tests fast and reliable

#### Example Test Structure

```python
class ArrayRecordFeatureTest(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test.array_record')
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_normal_case(self):
        """Test normal operation."""
        # Test implementation
        pass
    
    def test_edge_case(self):
        """Test edge case handling."""
        # Test implementation
        pass
    
    def test_error_handling(self):
        """Test error conditions."""
        with self.assertRaises(ValueError):
            # Code that should raise ValueError
            pass
```

## Release Process

### Version Management

ArrayRecord uses semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality additions
- **PATCH**: Backward-compatible bug fixes

### Release Checklist

1. **Update version numbers** in:
   - `setup.py`
   - `docs/conf.py`
   - Version constants in code

2. **Update CHANGELOG.md** with:
   - New features
   - Bug fixes
   - Breaking changes
   - Performance improvements

3. **Run full test suite**:
   ```bash
   bazel test //...
   ```

4. **Build and test packages**:
   ```bash
   python setup.py sdist bdist_wheel
   twine check dist/*
   ```

5. **Create release tag**:
   ```bash
   git tag -a v0.8.2 -m "Release version 0.8.2"
   git push origin v0.8.2
   ```

## Community Guidelines

### Code of Conduct

We follow the [Google Open Source Community Guidelines](https://opensource.google/conduct/). Please:

- **Be respectful** and inclusive
- **Be collaborative** and constructive
- **Be mindful** of cultural differences
- **Focus on the issue**, not the person

### Communication

- **GitHub Issues**: For bug reports and feature requests
- **Pull Requests**: For code contributions
- **Discussions**: For general questions and design discussions

### Getting Help

- **Documentation**: Check the documentation first
- **Search existing issues**: Your question might already be answered
- **Create new issue**: If you can't find an answer
- **Provide context**: Include relevant details when asking for help

## Advanced Topics

### Performance Optimization

When contributing performance improvements:

1. **Benchmark before and after** changes
2. **Profile code** to identify bottlenecks
3. **Consider different use cases** (sequential vs. random access)
4. **Document performance implications**

### Cross-Platform Compatibility

- **Test on multiple platforms** when possible
- **Consider platform-specific optimizations**
- **Use appropriate build configurations**
- **Document platform limitations**

### Security Considerations

- **Validate inputs** thoroughly
- **Handle errors gracefully**
- **Avoid buffer overflows** in C++ code
- **Consider security implications** of new features

## Resources

- [ArrayRecord GitHub Repository](https://github.com/google/array_record)
- [Bazel Documentation](https://bazel.build/docs)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [PEP 8 Python Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Apache Beam Documentation](https://beam.apache.org/documentation/)

Thank you for contributing to ArrayRecord! Your contributions help make high-performance data storage accessible to everyone.
