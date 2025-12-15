# ArrayRecord Documentation

This directory contains the Sphinx documentation for ArrayRecord.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -r requirements.txt
```

### Build HTML Documentation

```bash
# Using Sphinx directly
sphinx-build -b html . _build/html

# Or using the Makefile
make html
```

The generated HTML documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser to view it.

### Development Workflow

For development with live reload:

```bash
# Install additional development dependencies
pip install sphinx-autobuild

# Start live reload server
sphinx-autobuild . _build/html
```

This will start a local server (usually at http://127.0.0.1:8000) that automatically rebuilds and reloads when you make changes.

### Other Formats

Build PDF documentation (requires LaTeX):

```bash
make latexpdf
```

Check for broken links:

```bash
make linkcheck
```

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation index
├── installation.md      # Installation guide
├── quickstart.md        # Quick start guide
├── core_concepts.md     # Core concepts explanation
├── performance.md       # Performance optimization guide
├── examples.md          # Comprehensive examples
├── beam_integration.md  # Apache Beam integration guide
├── contributing.md      # Contribution guidelines
├── changelog.md         # Version changelog
├── python_reference.rst # Python API reference
├── beam_reference.rst   # Beam API reference
├── cpp_reference.rst    # C++ API reference
├── requirements.txt     # Documentation dependencies
├── Makefile            # Build automation
└── README.md           # This file
```

## Writing Documentation

### Markdown vs reStructuredText

- Use **Markdown (.md)** for user guides, tutorials, and narrative documentation
- Use **reStructuredText (.rst)** for API references and when you need advanced Sphinx features

### Style Guidelines

1. **Use clear, concise language**
2. **Include code examples** for all features
3. **Provide context** for when to use different options
4. **Link between related sections**
5. **Keep examples up-to-date** with the current API

### Code Examples

Always include working code examples:

```python
from array_record.python import array_record_module

# Create a writer
writer = array_record_module.ArrayRecordWriter('example.array_record')
writer.write(b'Hello, ArrayRecord!')
writer.close()
```

### Cross-References

Use Sphinx cross-references to link between sections:

- `:doc:`installation`` - Link to installation.md
- `:ref:`section-label`` - Link to a labeled section
- `:class:`array_record.python.array_record_module.ArrayRecordWriter`` - Link to class

### API Documentation

API documentation is auto-generated from docstrings. Ensure all public APIs have comprehensive docstrings:

```python
def my_function(param1: str, param2: int = 0) -> bool:
    """Brief description of the function.
    
    Longer description with more details about the function's behavior,
    use cases, and any important considerations.
    
    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter. Defaults to 0.
        
    Returns:
        Description of what the function returns.
        
    Raises:
        ValueError: Description of when this exception is raised.
        
    Example:
        >>> result = my_function("test", 42)
        >>> print(result)
        True
    """
    pass
```

## Contributing to Documentation

1. **Check existing documentation** before adding new content
2. **Follow the style guidelines** above
3. **Test your changes** by building the documentation locally
4. **Update the table of contents** if adding new pages
5. **Check for broken links** using `make linkcheck`

### Adding New Pages

1. Create the new file (`.md` or `.rst`)
2. Add it to the appropriate `toctree` in `index.rst`
3. Update cross-references as needed
4. Test the build

### Updating API Documentation

API documentation is automatically generated from docstrings. To update:

1. Modify the docstrings in the source code
2. Rebuild the documentation
3. Check that the changes appear correctly

## Troubleshooting

### Common Issues

1. **Import errors during build**: Ensure ArrayRecord is installed in your environment
2. **Missing dependencies**: Install requirements with `pip install -r requirements.txt`
3. **Build warnings**: Address warnings as they often indicate real issues
4. **Broken links**: Use `make linkcheck` to identify and fix broken links

### Clean Builds

If you encounter issues, try a clean build:

```bash
make clean
make html
```

### Debugging Sphinx

Enable verbose output to debug issues:

```bash
sphinx-build -v -b html . _build/html
```

## Deployment

The documentation is typically built and deployed automatically via CI/CD. For manual deployment:

1. Build the documentation: `make html`
2. Upload the `_build/html` directory to your web server
3. Ensure proper permissions and web server configuration

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [MyST Parser (Markdown)](https://myst-parser.readthedocs.io/)
- [Read the Docs Sphinx Theme](https://sphinx-rtd-theme.readthedocs.io/)
