# Legacy script adapters

Former flat `scripts/*.py` adapters live here, grouped explicitly as compatibility
inputs. Their first line identifies the canonical implementation. Running a file
here forwards to that implementation with its actual source identity and globals.

Use canonical experiment/shared-tool commands for new work. The source packager
also emits these bytes at declared legacy names in a new release, with manifest
hashes. See the [compatibility contract](../README.md) and
[mapping](../../../experiments/folder-layout.json). This folder launches nothing
by itself and grants no new experiment/retry authority.
