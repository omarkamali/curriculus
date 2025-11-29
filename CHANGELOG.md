## [0.1.6] - 2025-11-29

### Fixed
- Prime the column union cache on initialization by discovering columns from all datasets before sampling.

## [0.1.5] - 2025-11-29

### Fixed
- Preserve the union of columns when mixing datasets with differing schemas. The previous fix did not account for dataset and column counts >= 3.

## [0.1.4] - 2025-11-29

### Added
- Optional `shuffled` flag and `shuffle_seed` controls to randomize dataset order deterministically before curriculum sampling.

### Fixed
- Preserve the union of columns when mixing datasets with differing schemas, ensuring downstream transforms and metadata retain optional fields.

## [0.1.3] - 2025-11-29

### Added
- `Curriculus` splits now expose `to_hf_dataset()` alongside the `to_hf_iterable_dataset()` helper for Hugging Face interoperability.
- Column metadata helpers (`columns`, `shape`, `num_columns`) and lazy transform utilities (`remove_column`, `rename_column(s)`, `map`) to better mirror Hugging Face Dataset ergonomics.

### Changed
- Renamed `CurriculusIterableDataset` to `Curriculus`, returning a `CurriculusSplits` mapping with richer preview and repr support.
- Updated documentation and examples to use the simplified `Curriculus` API and highlight new helpers.

## [0.1.2] - 2025-11-29

### Changed
- Drop support for Python 3.8

## [0.1.1] - 2025-11-29

### Added
- Support for train/test split via `train_ratio` and `split_names` parameters
- `CurriculusIterableDataset` now returns a `CurriculusIterableDatasetDict` mapping

### Changed
- Updated documentation for split support

## [0.1.0] - 2025-11-29

### Added
- Initial public release of `curriculus` with progressive curriculum planning utilities.
