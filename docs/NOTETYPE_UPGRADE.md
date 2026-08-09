# Adding fields to a notetype that is already in Anki

**Verified 2026-08-08. The answer is: safe — append and re-import.** An `.apkg`
carrying the same `model_id` with fields *appended* upgrades the live notetype in
place, matches existing notes by GUID, and leaves their review history untouched.
No remap, no duplicate notes, no scheduling reset.

DESIGN_V2 ("Schema changes") assumed this and marked it unproven; M5.5 exists to
settle it before M5.4 ships three new Anki-visible fields. This is what M5.4
implements against.

## What was tested

`anki` Python library 26.8.1 (the same Rust backend the desktop app runs;
desktop here is **25.09**), against a fresh throwaway collection:

1. Build `data/decks/verbs.yaml` as it stands — notetype `1607392313`,
   19 fields — and import it. → 3 notes, 6 cards.
2. Give one card a review history by hand: interval 21 days, 4 reps, ease 2500,
   one `revlog` row.
3. Rebuild the *same* deck with `PitchPattern`, `PitchDiagram` and
   `ExampleAudio` **appended** to `FIELD_NAMES` — same `model_id`, same notetype
   name, same templates — and import that.

Import options were `merge_notetypes=True`, `with_scheduling=True`,
`with_deck_configs=False`.

## Result

| | before | after |
| --- | --- | --- |
| notetype id | `1607392313` | `1607392313` — same |
| declared fields | 19 | **22**, the three appended at the end |
| notes | 3 | 3 — no duplicates |
| cards | 6 | 6 |
| note GUIDs | `N5-CgB5\|p8`, … | identical |
| reviewed card | ivl 21, reps 4, 1 revlog row | **ivl 21, reps 4, 1 revlog row** |

The new fields arrive empty on existing notes, which is what an appended field
should do.

## The conditions this depends on

Append is safe. These are the parts that make it so, and none of them should be
treated as incidental:

- **Fields are appended, never inserted or reordered.** A note's values are
  positional; inserting a field in the middle shifts every value after it.
- **`model_id` and the notetype name are unchanged.** The id is what makes this
  an upgrade rather than a second notetype sitting beside the first.
- **Note GUIDs are unchanged**, which for janki means record ids are unchanged —
  the same invariant `stable_record_id` already protects.
- **Templates may reference the new fields**, but existing templates must keep
  working with them empty, since that is the state of every pre-existing note.

## Two traps worth writing down

**A stale in-memory view will tell you this does not work.** Reading
`col.models.all()` in the same session that performed the import returned the
*old* 19-field notetype while notes already carried 22 values — an apparently
broken half-upgraded state. It is an artifact of the cached view, not the
collection: re-opening the collection shows 22 fields everywhere. The first
version of this document said the opposite because of it. Re-open before
concluding anything.

**A scratch script named `inspect.py` shadows the standard library.** The
script's own directory leads `sys.path`, so `import inspect` inside `anki`
imported the scratch file and the whole library failed to load. Name spike
scripts so they cannot collide.

## Still to confirm

This used the library's import API. **The desktop app's File → Import applies its
own defaults**, and `merge_notetypes` is exactly the sort of option a GUI may set
differently. Before M5.4 relies on this, repeat it once through the app:

1. Anki → Profiles → **Add**, name it something like `janki-test` (leave
   `User 1` alone).
2. Import the current deck, study one card so it has real history.
3. Import a build carrying the appended fields.
4. Check Browse: the note should show the new empty fields, and the studied card
   should keep its interval rather than returning to New.

Record the outcome here. If the GUI behaves differently, the GUI is what counts —
it is how the deck is actually used.
