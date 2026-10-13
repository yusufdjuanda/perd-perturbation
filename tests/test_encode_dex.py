import numpy as np
import pandas as pd
import pytest

from perd_perturbation import encode_dex


class FakeBigWig:
    """Minimal pyBigWig stand-in: one chromosome with a few intervals, gaps between them."""

    def __init__(self):
        self.intervals_ = [(100, 150, 2.0), (150, 300, 1.0), (500, 600, 4.0)]
        self.length = 1000

    def chroms(self):
        return {"chr1": self.length}

    def values(self, chrom, start, end, numpy=True):
        out = np.full(end - start, np.nan)
        for s, e, v in self.intervals_:
            out[max(s, start) - start : min(e, end) - start] = v
        return out

    def intervals(self, chrom, start, end):
        return [(s, e, v) for s, e, v in self.intervals_ if e > start and s < end]


def test_bigwig_window_means_match_interval_method_and_treat_gaps_as_zero():
    bw = FakeBigWig()
    grid = pd.DataFrame({"chrom": ["chr1", "chr1", "chr1"], "bin": [0, 1, 2]})
    windows = encode_dex.grid_windows(grid)
    means = encode_dex.bigwig_window_means(bw, windows)
    # bin 0 [0,200): 50*2 + 50*1 = 150 / 200; bin 1 [200,400): 100*1 / 200; bin 2 [400,600): 100*4 / 200
    np.testing.assert_allclose(means, [0.75, 0.5, 2.0])
    for row, value in zip(windows.itertuples(), means):
        assert encode_dex.interval_mean(bw, row.chrom, row.start, row.end) == pytest.approx(value)


def test_window_past_chromosome_end_uses_nominal_length():
    bw = FakeBigWig()
    windows = pd.DataFrame({"chrom": ["chr1"], "start": [900], "end": [1100]})
    assert encode_dex.bigwig_window_means(bw, windows)[0] == 0.0


def test_read_encode_gene_fpkm_strips_versions_merges_par_and_drops_spike_ins(tmp_path):
    path = tmp_path / "genes.tsv"
    pd.DataFrame(
        {
            "gene_id": ["ENSG01.3", "ENSG01.3_PAR_Y", "ENSG02.1", "gSpikein_ERCC-00002"],
            "FPKM": [2.0, 1.0, 5.0, 100.0],
        }
    ).to_csv(path, sep="\t", index=False)
    fpkm = encode_dex.read_encode_gene_fpkm(path)
    assert fpkm.to_dict() == {"ENSG01": 3.0, "ENSG02": 5.0}
    symbols = pd.Series({"ENSG01": "A", "ENSG02": "A"})
    values, dropped = encode_dex.encode_log2fpkm_by_symbol(path, symbols)
    assert dropped == 1
    assert values["A"] == pytest.approx(np.log2(4.0))  # lowest Ensembl ID kept


def test_replicate_halves_by_position():
    assert encode_dex.replicate_halves([5, 1, 3, 2, 4]) == ([1, 3, 5], [2, 4])
    assert encode_dex.replicate_halves([2, 3, 4, 5]) == ([2, 4], [3, 5])
    assert encode_dex.replicate_halves([1, 2, 4, 5]) == ([1, 4], [2, 5])


def test_scale_constant_maps_median():
    k = encode_dex.scale_constant(3.0, 0.5)
    assert np.log2(k * 0.5 + 1) == pytest.approx(3.0)
