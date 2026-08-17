# Bundled assets

DSLE includes boss scenario saves and screenshot templates needed to reset and
verify the supported encounters. `checksums.sha256` records the expected bytes
for those runtime assets and the container build verifies the manifest.

The commercial Dark Souls: Remastered installation and executable are not
included. Users supply their own legally obtained installation at runtime; it
is attached to the container read-only and excluded from Git and Docker build
contexts.

The GPL license in the repository covers DSLE's source code. It does not by
itself grant redistribution rights for third-party game-derived saves or
screenshots.
