# Voico

Voico is a modular voice conversion engine that transforms speech through pitch, formant, and spectral processing. The project includes a CLI workflow, a Python API, and an optional FastAPI server.

## Core Architecture

- Analysis pipeline: pitch, formant, and spectral feature extraction.
- Matching pipeline: source-to-target profile alignment.
- DSP pipeline: pitch shifting, optional formant shifting, phase reconstruction.
- Quality pipeline: profile validation and conversion diagnostics.
- Storage pipeline: profile persistence with SQLite.

## Installation

```bash
pip install -e .
```

With optional extras:

```bash
pip install -e ".[full,server,dev]"
```

## CLI Usage

```bash
voico input.wav -o output.wav -p 2.0 -f 1.1 -q balanced -b 16
```

With a target voice:

```bash
voico input.wav -t target.wav -o output.wav -q high
```

Show metadata only:

```bash
voico input.wav --info
```

## API Server

```bash
voico-server --host 0.0.0.0 --port 8000
```

See API contract details in [docs/API_SPEC.md](docs/API_SPEC.md).

## Development Automation

- Verify quality gates: `poe check`
- Apply automated fixes: `poe fix`
- Run tests with coverage: `poe test`
- Type checks: `poe typecheck`
- Lint: `poe lint`
- Format verification: `poe format`

Windows helpers:

- `scripts/install.bat`
- `scripts/check.bat`

## Quality Gates

The quality workflow includes:

- Ruff formatting and lint checks.
- Mypy type checks on source modules.
- Pytest execution with coverage reporting.
- Vulture dead-code detection.

## License Summary

This project is licensed under UCSL-1.0. Read the full legal text in [LICENSE](LICENSE).

Allowed:

- Private use.
- Commercial use.
- Modification and redistribution under UCSL-1.0 terms.

Not allowed:

- Re-licensing derivatives as closed or proprietary.
- Removing required attribution.
- Restricting downstream recipients from source access required by UCSL-1.0.

## Repository Workflows

- [check.yml](.github/workflows/check.yml) runs quality gates on pull requests, pushes to `main`, and manual dispatch.
- [package.yml](.github/workflows/package.yml) builds distributable artifacts on version tags and manual dispatch.
