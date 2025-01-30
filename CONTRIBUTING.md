# LLMstudio Contribution Guide

## Contribution Guidelines

Thank you for expressing your interest in contributing to LLMstudio. To ensure that your contribution aligns with our guidelines, please carefully review the following considerations before proceeding:

1. For feature requests, we recommend creating a [GitHub Issue](https://github.com/tensoropsai/llmstudio/issues) on our repository. If it's your first time contributing, maybe start with a [good first issue](https://github.com/tensoropsai/llmstudio/labels/good%20first%20issue)
2. Clone the repo and make your changes on a local branch
3. Follow our repo guidelines
   - Ensure that you update any relevant docstrings and comments within your code
   - Run `pre-commit run --all-files` to lint your code
4. Sign your commits. Without signed commits, your changes will not be accepted for main.

## Branches

- All development happens in per-feature branches prefixed by contributor's
  initials. For example `feat/feature_name`.
- Approved PRs are merged to the `main` branch.

## Alpha releases:
You need to have your changes in the `develop` branch in order to push a new alpha version of any library `(llmstudio, llmstudio-proxy, llmstudio-tracker)`. Therefore, first guarantee that you feature branch is reviewed and working before merging to develop.

Process:
- Ensure the `feature/**` you worked is passing the tests and has the approvals necessary.
- Merge to `develop`
- Ensure the changes are in the develop branch
- Use GitHub Actions to initiate the pre-release process: [PyPI pre-release any module](https://github.com/TensorOpsAI/LLMstudio/actions/workflows/upload-pypi-dev.yml)
- Select the target library `(llmstudio, llmstudio-proxy, llmstudio-tracker)` and the target version for the final release (e.g., 1.1.0). Consult main branch and PyPI for current versions.
- Run the workflow.
- The workflow will automatically bump the version and create an alpha release of the library/module specified
- The workflow will automatically push changes back (bump version) to the develop branch

Repeat the process in case your `development` branch contains changes in multiple libraries.

## Final releases:
Once you're happy with the versions, create the Release notes on the PR between `develop` and `main` and merge to main branch when ready for full release. The workflow will automatically remove any `alpha` tag in your libraries and push the versions for every library/module that suffered changes.


