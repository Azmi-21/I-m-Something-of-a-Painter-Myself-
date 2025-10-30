# Contributing Guidelines

Please follow these guidelines to maintain code quality and collaboration efficiency.

## Branch Naming Convention

Use descriptive branch names with the following prefixes:

- `feature/` - New features or experiments (e.g., `feature/dcgan-implementation`)
- `bugfix/` - Bug fixes (e.g., `bugfix/training-loop-error`)
- `experiment/` - Experimental changes (e.g., `experiment/new-loss-function`)
- `docs/` - Documentation updates (e.g., `docs/update-readme`)

## Pull Request Rules

1. **Create a PR for all changes** - Never push directly to `main`
2. **Descriptive titles** - Use clear, concise PR titles
3. **Description required** - Explain what changes were made and why
4. **Review required** - At least one team member must review and approve
6. **Resolve conflicts** - Keep your branch up to date with `main`

## Commit Style

Follow conventional commit format:

```
<type>: <subject>

<body (optional)>
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, no logic change)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

**Examples:**
```
feat: implement DCGAN generator architecture
fix: correct discriminator loss calculation
docs: add training instructions to README
```
