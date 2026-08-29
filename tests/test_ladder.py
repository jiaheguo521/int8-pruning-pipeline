"""Contracts in int8_pruning.prune.ladder that fail SILENTLY when broken.

That module's own docstring says it: nothing it declares is enforceable from
inside it, and every way of getting it wrong produces a run that finishes, writes
files, and reports numbers. This file is the enforcement.

Stdlib only, on purpose. scripts/check.sh runs it in the tier that passes on a
fresh clone with no venv (`--clean-clone`, which is what CI runs), and the module
under test imports nothing but pathlib and typing, so there is no reason for these
to need an environment. Run them alone with:

    python3 -m unittest discover -s tests

Two contracts from ladder.py's docstring are NOT here, because they are properties
of a caller and not of this module: passing `initial_params=<dense count>` so a
ratio is measured against the dense model, and reusing one pruner for a whole
trajectory. Both still fail silently. Neither is reachable without torch and a
checkpoint, which is exactly why they are hard to guard.
"""
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from int8_pruning.prune.ladder import MODE_SUFFIX, mode_suffix, resume_point  # noqa: E402

# One real checkpoint name per family that declares `iterative` in its family.yaml,
# composed the way families/*/prune.py composes it:
#   {baseline}_pruned{cp}pct_{importance}{mode_sfx}{seed_sfx}.pt
# Each is checked against that family's own filename_pattern below, so a stale entry
# fails loudly instead of quietly testing nothing.
ITERATIVE_FAMILIES = {
    "imagenet_backbones": "mobilenetv2_imagenet_pruned50pct_magnitude_l2",
    "clip_rn50": "clip_rn50_pruned50pct_magnitude_l2",
    "relu_clip": "relu_clip_pruned50pct_magnitude_l2",
}


def _patterns(family):
    """The convert-side filename_pattern regexes declared by one family.

    Read with a regex rather than with pyyaml: they are single-line scalars, and
    keeping this file stdlib-only keeps it in check.sh's clean-clone tier.
    """
    hits = sorted((REPO / "families").glob(f"*/{family}/family.yaml"))
    assert len(hits) == 1, f"expected one families/*/{family}/family.yaml, got {hits}"
    text = hits[0].read_text()
    return re.findall(r"^\s*filename_pattern:\s*(\S+)\s*$", text, re.M)


class TestModeSuffix(unittest.TestCase):

    def test_independent_adds_nothing(self):
        """The default mode must contribute an EMPTY string.

        Every artifact on disk, every filename_pattern in families/*/family.yaml
        and all 202 hashes in results/deliverables.sha256 were produced in
        independent mode. Give it a marker of its own and none of those names is
        reachable any more -- and nothing would raise: convert.sh would simply
        match zero files and report success on an empty set.
        """
        self.assertEqual(mode_suffix("independent"), "")

    def test_iterative_is_marked(self):
        """The non-default mode must NOT be empty, or it overwrites a ladder.

        Same (model, ratio, importance) in the two modes means the same .pt and
        the same .json. An unmarked iterative run silently replaces a published
        independent rung, and the artifact does not record which made it.
        """
        self.assertNotEqual(mode_suffix("iterative"), "")

    def test_the_two_modes_do_not_collide(self):
        self.assertEqual(len(set(MODE_SUFFIX.values())), len(MODE_SUFFIX))

    def test_unknown_mode_is_rejected_and_says_what_is_known(self):
        with self.assertRaises(ValueError) as cm:
            mode_suffix("lottery")
        message = str(cm.exception)
        self.assertIn("lottery", message)
        for known in MODE_SUFFIX:
            self.assertIn(known, message)

    def test_every_marker_is_absorbable_by_the_convert_patterns(self):
        """A marker may only use characters `(?:_\\w+)?` can swallow.

        The convert patterns end in one optional `(?:_\\w+)?` group that already
        has to cover `_magnitude_l2`. `_iter` rides along only because `\\w`
        includes the underscore. A marker with a hyphen or a dot would make
        convert.sh skip every iterative checkpoint -- silently, since a
        non-matching file is simply not a candidate.
        """
        for mode, suffix in MODE_SUFFIX.items():
            with self.subTest(mode=mode):
                self.assertRegex(suffix, r"\A(_\w+)?\Z")

    def test_iterative_names_still_match_their_family_pattern(self):
        """End to end, on the three families that implement the mode.

        The base name is asserted first: if that stops matching, the table above
        is stale and the rest of this test would be checking nothing.
        """
        for family, stem in ITERATIVE_FAMILIES.items():
            patterns = _patterns(family)
            self.assertTrue(patterns, f"no filename_pattern in {family}/family.yaml")
            with self.subTest(family=family):
                base = [p for p in patterns if re.match(p, stem + ".pt")]
                self.assertTrue(base, f"{stem}.pt matches no pattern -- table is stale")
                marked = stem + mode_suffix("iterative")
                self.assertTrue([p for p in patterns if re.match(p, marked + ".pt")])
                # ...and with a non-default seed, which appends after the marker.
                self.assertTrue([p for p in patterns if re.match(p, marked + "_seed7.pt")])

    def test_declared_modes_are_implementable(self):
        """Every mode a family.yaml declares must be a mode this module knows.

        family.yaml is the manifest scripts/pruning.sh gates on; a mode named
        there but missing from MODE_SUFFIX raises only once a worker reaches
        mode_suffix(), after the job has started.
        """
        for yaml_path in sorted((REPO / "families").glob("*/*/family.yaml")):
            text = yaml_path.read_text()
            declared = re.search(r"^\s*prune_modes:\s*\[([^\]]*)\]", text, re.M)
            if not declared:
                continue
            with self.subTest(family=yaml_path.parent.name):
                for mode in (m.strip() for m in declared.group(1).split(",")):
                    self.assertIn(mode, MODE_SUFFIX)


class TestResumePoint(unittest.TestCase):
    """`resume_point(checkpoints, path_for, force)` -> (start_idx, resume_path)."""

    @staticmethod
    def _present(*existing):
        """A path_for whose files exist exactly for the given checkpoints."""
        done = set(existing)
        class _P:
            def __init__(self, cp): self.cp = cp
            def exists(self): return self.cp in done
            def __eq__(self, other): return isinstance(other, _P) and other.cp == self.cp
            def __repr__(self): return f"<rung {self.cp}%>"
        return _P

    def test_empty_ladder(self):
        self.assertEqual(resume_point([], self._present()), (0, None))

    def test_nothing_done_starts_from_the_dense_baseline(self):
        path_for = self._present()
        self.assertEqual(resume_point([10, 20, 30], path_for), (0, None))

    def test_everything_done_returns_the_end_and_no_path(self):
        """start_idx == len(checkpoints) is how a caller detects "nothing to do".

        resume_path stays None there: there is no next rung to continue into.
        """
        path_for = self._present(10, 20, 30)
        self.assertEqual(resume_point([10, 20, 30], path_for), (3, None))

    def test_leading_run_only_and_the_path_is_the_last_finished_rung(self):
        path_for = self._present(10)
        start, resume = resume_point([10, 20, 30], path_for)
        self.assertEqual(start, 1)
        self.assertEqual(resume, path_for(10))

    def test_a_hole_in_the_middle_stops_at_the_hole(self):
        """The documented case: 10 and 20 done, 30 missing, 40 present.

        Resume from 20 and REWRITE 40. Rung 40 was produced by a different
        trajectory, so continuing from it would silently publish a rung whose
        history no artifact records. Counting it would also skip rung 30 entirely.
        """
        path_for = self._present(10, 20, 40)
        start, resume = resume_point([10, 20, 30, 40], path_for)
        self.assertEqual(start, 2)
        self.assertEqual(resume, path_for(20))

    def test_force_ignores_every_finished_rung(self):
        path_for = self._present(10, 20, 30)
        self.assertEqual(resume_point([10, 20, 30], path_for, force=True), (0, None))

    def test_unsorted_is_rejected(self):
        """Contract 2. `progressive_pruning_to_target` loops `while count > target`,
        so a rung below the model's current size does nothing at all and emits a
        duplicate checkpoint. Descending input is the way that happens."""
        with self.assertRaises(ValueError):
            resume_point([30, 20, 10], self._present())

    def test_duplicates_are_rejected(self):
        """Strictly ascending, not merely non-decreasing: a repeated rung is the
        same no-op as a descending one."""
        with self.assertRaises(ValueError):
            resume_point([10, 20, 20, 30], self._present())

    def test_the_ladder_is_validated_before_force_short_circuits(self):
        """force=True returns early, but not early enough to accept a bad ladder:
        the run that follows would still walk it in the given order."""
        with self.assertRaises(ValueError):
            resume_point([30, 10], self._present(), force=True)

    def test_path_for_is_called_with_the_checkpoint_not_the_index(self):
        seen = []
        class _P:
            def exists(self): return False
        def path_for(cp):
            seen.append(cp)
            return _P()
        resume_point([10, 20, 30], path_for)
        self.assertEqual(seen[0], 10)
        self.assertNotIn(0, seen)

    def test_the_caller_may_pass_any_sequence(self):
        """The signature says Sequence[int]; a tuple must behave like a list."""
        path_for = self._present(10)
        self.assertEqual(resume_point((10, 20), path_for)[0], 1)


if __name__ == "__main__":
    unittest.main()
