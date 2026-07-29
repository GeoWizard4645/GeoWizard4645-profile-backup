# Publish this GitHub profile

GitHub profiles are customized through one special public repository whose name exactly matches your username.

1. Go to [github.com/new](https://github.com/new).
2. Set the owner to **GeoWizard4645**.
3. Name the repository **GeoWizard4645** (matching capitalization exactly).
4. Set it to **Public** and create it without adding a README, `.gitignore`, or license.
5. Upload `README.md` and the entire `assets` folder from this package.
6. Commit to `main`, then return to [your profile](https://github.com/GeoWizard4645).

The repository is public, correctly named, and its README renders normally when
opened directly. If the README still does not appear on the profile overview,
GitHub is serving a stale profile cache. Push one new README commit (this
portrait update does that), then open `README.md` on GitHub, click the pencil,
make a harmless one-character edit, and commit it to `main` if the profile still
has not refreshed after a few minutes.

## Regenerate the portrait

The committed `assets/ascii-portrait.svg` is self-contained; the source photo is
deliberately not stored in this public repository. To rebuild it from a local
photo:

```bash
python3 scripts/generate_portrait.py /path/to/photo.png
```

The script detects the face, crops to a head-and-shoulders composition, removes
the background, maps the result through a 13-character brightness ramp, and
builds the row-by-row SVG typing animation. It requires `opencv-python` and
`numpy`.

## Suggested pin order

1. `caduceus`
2. `sprig-dashboard`
3. `PersonalPortfolio`
4. `debate101`
5. `BlotInator`
6. `homebrew-caduceus`

Also unpin the `FitFo` fork unless you want it to represent your current role; a strong profile usually prioritizes repositories where your ownership is immediately clear.

## Easy edits

- Change the hero copy in `assets/header.svg`.
- Change the current-status line near the top of `README.md`.
- Replace any featured card as your next major project ships.
- Keep the page focused: four strong projects beat fifteen equal-weight links.
