#!/bin/bash
# Source from one allocated launcher. PyTorch AF_UNIX listeners require short paths.
if [[ ! "${LSB_JOBID:-}" =~ ^[0-9]{1,20}$ ]]; then
  printf '%s\n' 'Boltz node temporary directory requires a numeric LSB_JOBID.' >&2
  return 1
fi
if [[ -n "$(trap -p EXIT)" ]]; then
  printf '%s\n' 'Refusing to replace an existing EXIT cleanup handler.' >&2
  return 1
fi
MOSAIC_BOLTZ_NODE_TMPDIR=$(mktemp -d -- "/tmp/mosaic-boltz.${LSB_JOBID}.XXXXXX") || return 1
readonly MOSAIC_BOLTZ_NODE_TMPDIR
export TMPDIR="$MOSAIC_BOLTZ_NODE_TMPDIR"

mosaic_boltz_cleanup_tmpdir() {
  local status=$?
  # Never remove an inherited path, symlink, or directory owned by another user.
  if [[ -e "$MOSAIC_BOLTZ_NODE_TMPDIR" || -L "$MOSAIC_BOLTZ_NODE_TMPDIR" ]]; then
    if [[ "$MOSAIC_BOLTZ_NODE_TMPDIR" =~ ^/tmp/mosaic-boltz\.[0-9]{1,20}\.[[:alnum:]]{6}$ \
          && -d "$MOSAIC_BOLTZ_NODE_TMPDIR" \
          && ! -L "$MOSAIC_BOLTZ_NODE_TMPDIR" \
          && -O "$MOSAIC_BOLTZ_NODE_TMPDIR" ]]; then
      if ! rm -rf -- "$MOSAIC_BOLTZ_NODE_TMPDIR"; then
        printf '%s\n' 'Failed to clean owned Boltz node temporary directory.' >&2
        return 1
      fi
    else
      printf '%s\n' 'Refusing unsafe Boltz temporary-directory cleanup.' >&2
      return 1
    fi
  fi
  return "$status"
}
trap mosaic_boltz_cleanup_tmpdir EXIT
