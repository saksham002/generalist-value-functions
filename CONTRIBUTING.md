# Contributing

Bug reports, questions and pull requests are welcome through GitHub issues. This is a
research code release maintained by a small team, so we cannot promise to review every
request quickly, but we will do our best.

When filing an issue, please include your OS and Python version, the exact command you
ran, and the full traceback. When opening a pull request, run `ruff check .` and
`pytest --strict-markers -m "not manual"` first.

The code is released under the [Apache 2.0 license](LICENSE); the Gemma weights it loads
are covered by the [Gemma terms of use](LICENSE_GEMMA.txt).
