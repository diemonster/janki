# Adding fields to a notetype that is already in Anki

**Verified 2026-08-08, in the desktop app as well as the library. The answer is
conditional, and the condition is not the default.**

Appending fields to a notetype that already exists in a collection works — the
notetype upgrades in place, notes are matched by GUID, review history survives —
**only if the import has "Merge Notetypes" enabled.** Anki's default is *off*,
and with it off the import silently does something else: it keeps every existing
note on the old schema and files the new one as a separate, empty notetype with
a `+` appended to its name.

DESIGN_V2 ("Schema changes") assumed the in-place upgrade and marked it
unproven. It is real, but it is opt-in, and a user who does not know to tick a
box gets the broken outcome with no error.

## The two outcomes

Same `.apkg`, same `model_id`, three fields appended, imported into a collection
holding three notes with a review history on two cards:

| | Merge Notetypes **on** | Merge Notetypes **off** (the default) |
| --- | --- | --- |
| `Japanese Study (recognition+production)` | **22 fields, 3 notes** | 19 fields, 3 notes |
| `Japanese Study (recognition+production)+` | — | 22 fields, **0 notes** |
| values per note | 22 | 19 — the new fields never arrive |
| review history | preserved | preserved |
| duplicates | none | none |

Review history is safe either way, which is why the failure is quiet: nothing is
lost or corrupted, the new fields simply never reach a note, and the collection
grows a `+` notetype nobody asked for.

## How this was established

1. **Library, `merge_notetypes=True`** — notetype `1607392313` went 19 → 22
   fields, all three notes matched by GUID, the reviewed card kept ivl 21 /
   reps 4 / its revlog row.
2. **Desktop app 25.09, default options** — the owner studied two cards and
   imported the appended build through File → Import. Inspecting the resulting
   profile found the split above: 3 notes still on the 19-field notetype, an
   empty 22-field `…+` beside it. Both studied cards kept their scheduling
   (`Due 2026-08-08` rather than New).
3. **Library, `merge_notetypes=False`** — reproduced the GUI result exactly,
   which identifies the option as the whole difference rather than something
   else about the GUI path.

## What this means for M5.4

Appending the fields is correct; shipping it without saying this is not. M5.4
has to deal with the fact that the good path is opt-in:

- **Document the checkbox** wherever the README tells someone to import a rebuilt
  deck. It is one tick, and without it the upgrade quietly does not happen.
- **Detect the failure.** A notetype whose name ends in `+`, or an existing
  notetype with fewer fields than `FIELD_NAMES`, means a previous import took the
  default. janki can see both — `janki status` is the natural place to say so,
  since it already reports on things being out of step.
- **Do not renumber `model_id` to force a fresh notetype.** That orphans every
  existing card's scheduling, which is the outcome this whole exercise exists to
  avoid.

## Conditions the in-place upgrade depends on

Beyond the checkbox:

- **Fields are appended, never inserted or reordered.** Note values are
  positional; inserting in the middle shifts everything after it.
- **`model_id` and the notetype name are unchanged.** The id is what makes this
  an upgrade rather than a second notetype.
- **Note GUIDs are unchanged** — for janki that is the record id, which
  `stable_record_id` already protects.
- **Existing templates keep working with the new fields empty**, since that is
  the state of every pre-existing note.

## Two traps for whoever repeats this

**A stale in-memory view will tell you the upgrade failed.** Reading
`col.models.all()` in the same session as the import returns the *old* notetype
while notes already carry the new values — an apparently broken half-upgrade
that is only a cached view. The first version of this document reported exactly
that, wrongly. Re-open the collection before concluding anything.

**A scratch script named `inspect.py` shadows the standard library.** The
script's directory leads `sys.path`, so `import inspect` inside the `anki`
package picks up the scratch file and the library fails to load. Name spike
scripts so they cannot collide.
