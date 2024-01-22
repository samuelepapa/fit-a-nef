# Contributing

Thank you for your interest in contributing to this repository! We welcome contributions from the community.

## Getting Started

To get started with contributing, please follow these steps:

1. Fork the repository.
2. Clone the forked repository to your local machine.
3. Install and configure `pre-commit` (see below).
4. Create a new branch for your changes.
5. Make your changes and commit them.
6. Push the changes to your forked repository.
7. Open a pull request to the main repository.

## Guidelines

The repository follows the "make it simple, not easy" philosophy ([see here](https://www.entropywins.wtf/blog/2017/01/02/simple-is-not-easy/)).

We prioritize extensibility, and strong independence between packages.
This means that we prefer to have several simple components that have a small set of functionalities and leave the
onus of building powerful software to the end user.

The code does not need to be as concise as it could be.
However, it must always be easy to add new tasks, new neural fields, and new datasets.
Additionally, the dataset format must be standardized across tasks.
Finally, always provide clear documentation and error messages.

Before committing, please ensure to have `pre-commit` installed. This will ensure that the code is formatted correctly and that the tests pass. To install it, run:

```bash
pip install pre-commit
pre-commit install
```

Please follow these guidelines when contributing:

- Make sure your code follows the coding style and conventions used in the project.
- Write clear and concise commit messages.
- Include tests for your changes, if applicable.
- Document any new features or changes in the project's documentation.

## Code of Conduct

Please note that this project has a Code of Conduct. By participating in this project, you agree to abide by its terms.

## Contact

If you have any questions or need further assistance, please contact the project maintainers at [s.papa@uva.nl](mailto:s.papa@uva.nl).

Happy contributing!
