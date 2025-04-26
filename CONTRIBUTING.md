# Contributing to Multidimensional Neural Networks

Thank you for your interest in contributing to the Multidimensional Neural Networks (MNN) project! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## How to Contribute

There are many ways to contribute to this project:

1. **Report bugs**: If you find a bug, please create an issue with a detailed description of the problem, steps to reproduce it, and your environment details.

2. **Suggest enhancements**: If you have ideas for new features or improvements, please create an issue describing your suggestion.

3. **Improve documentation**: Help us improve the documentation by fixing typos, adding examples, or clarifying explanations.

4. **Submit code changes**: Implement new features, fix bugs, or improve existing code.

## Development Process

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:

   ```bash
   git clone https://github.com/mohamed-services/nn.git
   cd nn
   ```

3. Create a new branch for your changes:

   ```bash
   git checkout -b feature/your-feature-name
   ```

### Code Style Guidelines

Please follow these guidelines when writing code:

- Use 4 spaces for indentation (no tabs)
- Follow PEP 8 style guide for Python code
- Write clear, descriptive variable and function names
- Add docstrings to all functions and classes
- Include type hints where appropriate

### Testing

- Add tests for new features or bug fixes
- Ensure all tests pass before submitting a pull request
- For TensorFlow implementation, use `tf.test`
- For PyTorch implementation, use `unittest` or `pytest`

### Documentation

- Update documentation to reflect your changes
- Add examples for new features
- Ensure code comments are clear and helpful

## Pull Request Process

1. Update your fork to the latest version of the main repository
2. Make your changes in a new branch
3. Test your changes thoroughly
4. Update documentation as needed
5. Submit a pull request with a clear description of the changes
6. Address any feedback from code reviews

## Architectural Decisions

When contributing to the core architecture, please consider:

1. **Compatibility**: Ensure changes work with both TensorFlow and PyTorch implementations
2. **Performance**: Consider the computational efficiency of your implementation
3. **Flexibility**: Maintain the configurable nature of the architecture
4. **Clarity**: Keep the code readable and well-documented

## Research Contributions

If you're extending the theoretical foundations:

1. Clearly explain the mathematical basis for your changes
2. Provide empirical evidence or theoretical justification
3. Reference relevant academic papers or resources
4. Consider adding your findings to the paper.md document

## License

By contributing to this project, you agree that your contributions will be licensed under the project's license.

## Questions?

If you have any questions about contributing, please open an issue for discussion.

Thank you for helping improve the Multidimensional Neural Networks project!
