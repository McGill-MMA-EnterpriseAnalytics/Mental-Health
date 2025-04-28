Contributing Guidelines

Thank you for considering contributing to this project!
We follow a structured contribution workflow to maintain code quality and consistency.

Contribution Workflow

1. Implement Your Feature or Fix
Start by creating a new branch from main:
git checkout -b feature/your-feature-name
Make your code changes.
Follow the existing coding style and conventions.
2. Write Tests
Add or update tests under the tests/ directory to cover your new functionality.
Ensure that your code is fully tested before submitting a PR.
3. Create a Pull Request (PR)
Push your branch:
git push origin feature/your-feature-name
Open a Pull Request (PR) against the main branch.
Include a clear title and description explaining the purpose of the changes.
4. Review Process
Your PR will be reviewed by project maintainers or teammates.
Please be responsive to feedback and requested changes.
5. Merge
After approval and successful passing of all CI/CD checks, your PR will be merged into main.

Notes

Run pytest before pushing to ensure all tests pass:
pytest tests/
Make sure the CI/CD workflow passes (GitHub Actions will automatically check your PR).
Keep commits small and focused; use meaningful commit messages.
