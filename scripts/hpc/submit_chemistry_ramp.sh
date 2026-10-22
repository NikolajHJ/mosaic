#!/bin/bash
# Compatibility entry point; implementation: experiments/brd4/chemistry-ramp-10000/hpc/submit_chemistry_ramp.sh
_mosaic_compat_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  source "$_mosaic_compat_root/experiments/brd4/chemistry-ramp-10000/hpc/submit_chemistry_ramp.sh"
else
  exec bash -l "$_mosaic_compat_root/experiments/brd4/chemistry-ramp-10000/hpc/submit_chemistry_ramp.sh" "$@"
fi
