# Changelog

## [Unreleased](https://github.com/btschwertfeger/python-cmethods/tree/HEAD)

[Full Changelog](https://github.com/btschwertfeger/python-cmethods/compare/v1.0.2...HEAD)

**Fixed bugs:**

- The tool fails when input time series include np.nan for distribution-based methods [\#41](https://github.com/btschwertfeger/python-cmethods/issues/41)
- Fix error when time series includes nan values [\#40](https://github.com/btschwertfeger/python-cmethods/pull/40) ([btschwertfeger](https://github.com/btschwertfeger))

## [v1.0.2](https://github.com/btschwertfeger/python-cmethods/tree/v1.0.2) (2023-06-18)

[Full Changelog](https://github.com/btschwertfeger/python-cmethods/compare/v1.0.1...v1.0.2)

**Merged pull requests:**

- Clarified difference between stochastic and non-stochastic climate variables in doc and readme [\#36](https://github.com/btschwertfeger/python-cmethods/pull/36) ([btschwertfeger](https://github.com/btschwertfeger))
- Fix typos [\#38](https://github.com/btschwertfeger/python-cmethods/pull/38) ([btschwertfeger](https://github.com/btschwertfeger))

## [v1.0.1](https://github.com/btschwertfeger/python-cmethods/tree/v1.0.1) (2023-04-17)

[Full Changelog](https://github.com/btschwertfeger/python-cmethods/compare/v1.0.0...v1.0.1)

**Breaking changes:**

- Split Quantile Mapping into Quantile Mapping and Detrended Quantile Mapping [\#18](https://github.com/btschwertfeger/python-cmethods/issues/18)
- Split Quantile Mapping into Quantile Mapping and Detrended Quantile Mapping [\#34](https://github.com/btschwertfeger/python-cmethods/pull/34) ([btschwertfeger](https://github.com/btschwertfeger))

**Fixed bugs:**

- Multiplicative Quantile Delta Mapping is not applying scaling where the delta is infinite [\#32](https://github.com/btschwertfeger/python-cmethods/issues/32)
- Fixed PyPI repository URL [\#16](https://github.com/btschwertfeger/python-cmethods/pull/16) ([btschwertfeger](https://github.com/btschwertfeger))
- Fixed bug where division lead to nan or inf values [\#33](https://github.com/btschwertfeger/python-cmethods/pull/33) ([btschwertfeger](https://github.com/btschwertfeger))

**Closed issues:**

- Create a changelog [\#19](https://github.com/btschwertfeger/python-cmethods/issues/19)

**Merged pull requests:**

- Create a Changelog [\#21](https://github.com/btschwertfeger/python-cmethods/pull/21) ([btschwertfeger](https://github.com/btschwertfeger))
- Extended the description of quantile mapping with text and images [\#20](https://github.com/btschwertfeger/python-cmethods/pull/20) ([btschwertfeger](https://github.com/btschwertfeger))
- Prepare release [\#35](https://github.com/btschwertfeger/python-cmethods/pull/35) ([btschwertfeger](https://github.com/btschwertfeger))

## [v1.0.0](https://github.com/btschwertfeger/python-cmethods/tree/v1.0.0) (2023-04-10)

[Full Changelog](https://github.com/btschwertfeger/python-cmethods/compare/v0.6.3...v1.0.0)

**Breaking changes:**

- Remove the unnecessary `CMethods.py` module to access the CMethods class more easily [\#10](https://github.com/btschwertfeger/python-cmethods/issues/10)

**Implemented enhancements:**

- Create a release workflow for dev and production releases [\#8](https://github.com/btschwertfeger/python-cmethods/issues/8)
- Move from setup.py to pyproject.toml [\#7](https://github.com/btschwertfeger/python-cmethods/issues/7)
- Adjusted and improved the examples [\#15](https://github.com/btschwertfeger/python-cmethods/pull/15) ([btschwertfeger](https://github.com/btschwertfeger))
- Improving workflows - adding release workflow [\#12](https://github.com/btschwertfeger/python-cmethods/pull/12) ([btschwertfeger](https://github.com/btschwertfeger))

**Closed issues:**

- Create a documentation [\#9](https://github.com/btschwertfeger/python-cmethods/issues/9)

**Merged pull requests:**

- Moved the content of `CMethods.py` to `__init__.py` and adjusted the imports [\#14](https://github.com/btschwertfeger/python-cmethods/pull/14) ([btschwertfeger](https://github.com/btschwertfeger))
- Create the documentation [\#13](https://github.com/btschwertfeger/python-cmethods/pull/13) ([btschwertfeger](https://github.com/btschwertfeger))
- Move from setup.py to pyproject.toml [\#11](https://github.com/btschwertfeger/python-cmethods/pull/11) ([btschwertfeger](https://github.com/btschwertfeger))

## [v0.6.3](https://github.com/btschwertfeger/python-cmethods/tree/v0.6.3) (2023-03-22)

[Full Changelog](https://github.com/btschwertfeger/python-cmethods/compare/v0.6.2...v0.6.3)

**Breaking changes:**

- adjust_3d forces group to be "time.month" if group is set to the default \(None\) [\#5](https://github.com/btschwertfeger/python-cmethods/issues/5)

**Implemented enhancements:**

- Add pre-commit [\#2](https://github.com/btschwertfeger/python-cmethods/issues/2)

**Fixed bugs:**

- Removed forced 'time.month' grouping in `adjust_3d` when no group was specified [\#4](https://github.com/btschwertfeger/python-cmethods/pull/4) ([btschwertfeger](https://github.com/btschwertfeger))

## [v0.6.2](https://github.com/btschwertfeger/python-cmethods/tree/v0.6.2) (2023-03-14)

[Full Changelog](https://github.com/btschwertfeger/python-cmethods/compare/v0.6.1...v0.6.2)

**Fixed bugs:**

- KeyError when Quantile\* Mapping and group != None [\#3](https://github.com/btschwertfeger/python-cmethods/issues/3)

**Merged pull requests:**

- Added pre-commit workflow to standardize the code base [\#1](https://github.com/btschwertfeger/python-cmethods/pull/1) ([btschwertfeger](https://github.com/btschwertfeger))

## [v0.6.1](https://github.com/btschwertfeger/python-cmethods/tree/v0.6.1) (2022-12-02)

[Full Changelog](https://github.com/btschwertfeger/python-cmethods/compare/v0.6...v0.6.1)

## [v0.6](https://github.com/btschwertfeger/python-cmethods/tree/v0.6) (2022-11-28)

[Full Changelog](https://github.com/btschwertfeger/python-cmethods/compare/v0.5.4.2...v0.6)

## [v0.5.4.2](https://github.com/btschwertfeger/python-cmethods/tree/v0.5.4.2) (2022-11-14)

[Full Changelog](https://github.com/btschwertfeger/python-cmethods/compare/0.5.4.1...v0.5.4.2)

## [0.5.4.1](https://github.com/btschwertfeger/python-cmethods/tree/0.5.4.1) (2022-11-09)

[Full Changelog](https://github.com/btschwertfeger/python-cmethods/compare/v0.5.4...0.5.4.1)

## [v0.5.4](https://github.com/btschwertfeger/python-cmethods/tree/v0.5.4) (2022-11-05)

[Full Changelog](https://github.com/btschwertfeger/python-cmethods/compare/v0.5.3...v0.5.4)

## [v0.5.3](https://github.com/btschwertfeger/python-cmethods/tree/v0.5.3) (2022-10-26)

[Full Changelog](https://github.com/btschwertfeger/python-cmethods/compare/v0.5.2...v0.5.3)

## [v0.5.2](https://github.com/btschwertfeger/python-cmethods/tree/v0.5.2) (2022-10-14)

[Full Changelog](https://github.com/btschwertfeger/python-cmethods/compare/v0.5.1...v0.5.2)

## [v0.5.1](https://github.com/btschwertfeger/python-cmethods/tree/v0.5.1) (2022-08-19)

[Full Changelog](https://github.com/btschwertfeger/python-cmethods/compare/v0.5...v0.5.1)

## [v0.5](https://github.com/btschwertfeger/python-cmethods/tree/v0.5) (2022-08-19)

[Full Changelog](https://github.com/btschwertfeger/python-cmethods/compare/41c1837e5d23c300656c8ee2ce0079d6a8baac2f...v0.5)

\* _This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)_
