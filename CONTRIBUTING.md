# Contributing to Atlas

Thank you for your interest in contributing to Atlas! This document provides guidelines and instructions for contributing.

## 🚀 Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/longmem.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `python -m pytest` (if tests exist)
6. Commit: `git commit -m "Add: your feature description"`
7. Push: `git push origin feature/your-feature-name`
8. Open a Pull Request

## 📋 Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks (recommended)
pip install pre-commit
pre-commit install

# Run the demo to verify setup
./run_demo.sh
```

## 🎯 Areas for Contribution

### High Priority
- [ ] **Streaming support** - Add async/streaming response generation
- [ ] **Memory decay** - Implement LRU-style expiration based on `last_used_turn`
- [ ] **Multi-user support** - Add session IDs for multi-user scenarios
- [ ] **Web UI** - Create FastAPI backend + React frontend
- [ ] **More eval scenarios** - Add additional test cases

### Medium Priority
- [ ] **Memory conflict resolution** - Handle conflicting memories gracefully
- [ ] **Import/export** - Backup and restore memory databases
- [ ] **Performance optimizations** - Cache embeddings, batch operations
- [ ] **Better error handling** - More informative error messages
- [ ] **Documentation** - Expand docstrings and add architecture diagrams

### Low Priority
- [ ] **Alternative LLM providers** - Add support for Anthropic, Cohere, etc.
- [ ] **Custom embedding models** - Allow user-specified embedding models
- [ ] **Memory visualization** - Graph-based memory relationship viewer
- [ ] **Memory compression** - Summarize old memories to save space

## 🧪 Testing Guidelines

When adding new features:

1. **Add tests** in the `eval/` directory
2. **Test with multiple models** (Groq, OpenAI, local)
3. **Verify spec compliance** - Run `eval/evaluate.py`
4. **Check edge cases** - Empty conversations, very long conversations, etc.

Example test structure:
```python
def test_memory_retrieval():
    agent = LongMemAgent(db_path="test.db")
    agent.chat("My name is Alice")
    agent.manual_distill()
    
    response = agent.chat("What's my name?")
    assert "Alice" in response['response']
    assert len(response['active_memories']) > 0
```

## 📝 Code Style

- **Python 3.11+** syntax
- **Type hints** for all function signatures
- **Docstrings** for all modules, classes, and public functions
- **Black** for code formatting (line length: 88)
- **isort** for import sorting

Example:
```python
def retrieve_memories(
    self, 
    query: str, 
    top_k: int = 5
) -> list[RetrievalResult]:
    """
    Retrieve relevant memories using hybrid search.
    
    Args:
        query: User query string
        top_k: Number of memories to retrieve
        
    Returns:
        List of RetrievalResult objects sorted by relevance
    """
    # Implementation here
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Python version**: `python --version`
2. **Package versions**: `pip list`
3. **Error message**: Full traceback
4. **Minimal reproduction**: Smallest code to reproduce the issue
5. **Expected behavior**: What should happen
6. **Actual behavior**: What actually happens

## 💡 Feature Requests

When requesting features:

1. **Use case**: Why is this feature needed?
2. **Proposed solution**: How would it work?
3. **Alternatives**: What alternatives have you considered?
4. **Spec impact**: Does it affect the specification compliance?

## 📦 Pull Request Process

1. **Update documentation** if you change APIs
2. **Add tests** for new functionality
3. **Update CHANGELOG.md** with your changes
4. **Ensure tests pass** before submitting
5. **Keep PRs focused** - One feature per PR
6. **Write clear commit messages**

### Commit Message Format

```
Type: Brief description (50 chars max)

Detailed explanation (if needed)
- Bullet points for multiple changes
- Reference issues with #issue_number

Closes #123
```

Types: `Add`, `Fix`, `Update`, `Remove`, `Refactor`, `Docs`, `Test`

## 🔍 Code Review Criteria

Your PR will be reviewed for:

- ✅ **Correctness** - Does it work as intended?
- ✅ **Completeness** - Are tests and docs included?
- ✅ **Code quality** - Is it readable and maintainable?
- ✅ **Performance** - Is it efficient?
- ✅ **Compatibility** - Does it break existing functionality?
- ✅ **Spec compliance** - Does it maintain 100/100 compliance?

## 📚 Resources

- **Specification**: `eval/SPEC_COMPLIANCE_ANALYSIS.md`
- **Architecture**: `README.md` Architecture section
- **Implementation details**: `LAST_USED_TURN_IMPLEMENTATION.md`
- **Evaluation**: `eval/evaluate.py`

## 🤝 Community

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code contributions

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Atlas! 🎉**
