#!/bin/bash
# Compatibility entry point; implementation: experiments/brd4/chemistry-5000/hpc/submit_chemistry_readiness.sh
_mosaic_compat_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  source "$_mosaic_compat_root/experiments/brd4/chemistry-5000/hpc/submit_chemistry_readiness.sh"
else
  exec bash -l "$_mosaic_compat_root/experiments/brd4/chemistry-5000/hpc/submit_chemistry_readiness.sh" "$@"
fi
