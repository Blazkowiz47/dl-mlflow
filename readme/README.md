# dl-mlflow Docs

Current public release: `deep-learning-mlflow==0.0.16`, requiring
`deep-learning-core>=0.1.8,<0.2`.

## What's New in 0.0.16?

- local sweep runs nest under their parent, and invalid scalar metrics are
  omitted from MLflow logging
- the package uses dl-core 0.1.8 for runtime extension registration

- [Release History](../RELEASES.md)
- [`dl-core`](https://github.com/Blazkowiz47/dl-core)
- [`dl-azure`](https://github.com/Blazkowiz47/dl-azure)
- [`dl-wandb`](https://github.com/Blazkowiz47/dl-wandb)

- [TLDR: Install And Enable](./tldr/1_install_and_enable.md)
- [Guide: Wiring MLflow](./guide/1_wiring_mlflow_into_an_experiment_repo.md)
- [Technical: Callback And Scaffold](./technical/1_callback_and_scaffold.md)
