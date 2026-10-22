# Reading this notebook with an agent

Only this reading guide is distributed with the repository organization changes.
The exported notes, attachments and raw notebook data are optional local inputs;
they are not included in this commit or supplied by a fresh clone.

> **Imported reference snapshot:** these are exported lab notes, not a current
> portable Mosaic runbook. Named users' paths, module/queue suggestions and
> example storage allowances are not guaranteed for another account. Use the
> [maintained HPC guide](../HPC_GUIDE.md) and actual quota/access checks.
> Export warnings and original note/attachment contents remain preserved; this
> repository audit does not certify external lab, HR, grant or financial policy.

Start with INDEX.md, then read the linked notes under notes/. Directory names follow the original section groups and sections, with short suffixes to prevent filename collisions. Page filenames start with their order in the section. INDEX.md and manifest.json preserve subpage levels and original titles.

Images and attachments are under each section's assets/ directory, linked from the corresponding Markdown note. Existing OCR text is included when OneNote exposes it. Tables and inline formatting are converted to Markdown when supported; a fragment may remain as HTML to avoid discarding it.

Read EXPORT_REPORT.md before assuming the export is complete. manifest.json records each source section, page, original IDs, original dates, output paths, warnings, and failures. raw_xml/ preserves original page XML and the temporary notebook hierarchy. Raw XML uses binary callbacks; the binary files are stored as assets rather than duplicated as base64.

The notebook was exported from a separate working copy. The script never calls OneNote content-editing methods. It leaves the temporary native copy on disk so there is no risk of deleting files that OneNote still has open. WorkingCopyPath in manifest.json shows its location. The supplied source files are checked by SHA-256 before and after the export.

Free-form layout, handwriting recognition, embedded application objects, and some OneNote formatting may need manual review. Links to pages included in this export are rewritten to local notes when their OneNote page ID is recognizable; other links retain the original target.
