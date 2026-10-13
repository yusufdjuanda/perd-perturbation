import numpy as np
import pandas as pd
import pytest

from perd_perturbation import atac, rna


# ---- RNA -------------------------------------------------------------------


def test_parse_gencode_header_strips_version():
    header = ">ENST00000456328.2|ENSG00000290825.1|-|-|DDX11L2-202|DDX11L2|1657|lncRNA|"
    parsed = rna.parse_gencode_header(header)
    assert parsed["transcript_id"] == "ENST00000456328.2"
    assert parsed["gene_id"] == "ENSG00000290825"
    assert parsed["gene_name"] == "DDX11L2"
    assert rna.strip_version("ENSG00000182378.14_PAR_Y") == "ENSG00000182378"


def test_aggregate_to_genes_uses_tpm_weighted_length_and_fpkm():
    quant = pd.DataFrame(
        {
            "Name": ["t1", "t2", "t3"],
            "Length": [1000, 3000, 500],
            "EffectiveLength": [800.0, 2800.0, 400.0],
            "TPM": [30.0, 10.0, 0.0],
            "NumReads": [60.0, 40.0, 0.0],
        }
    )
    tx2gene = pd.DataFrame({"transcript_id": ["t1", "t2", "t3"], "gene_id": ["G1", "G1", "G2"], "gene_name": ["A", "A", "B"]})
    genes = rna.aggregate_to_genes(quant, tx2gene)
    expected_len = (30 * 800 + 10 * 2800) / 40
    assert genes.loc["G1", "effective_length"] == pytest.approx(expected_len)
    assert genes.loc["G1", "fpkm"] == pytest.approx(1e9 * 100 / (expected_len * 100))
    # Zero-TPM gene falls back to the unweighted mean length and has FPKM 0.
    assert genes.loc["G2", "effective_length"] == pytest.approx(400.0)
    assert genes.loc["G2", "fpkm"] == 0.0


def test_aggregate_to_genes_rejects_unknown_transcripts():
    quant = pd.DataFrame({"Name": ["tx"], "Length": [1], "EffectiveLength": [1.0], "TPM": [1.0], "NumReads": [1.0]})
    with pytest.raises(ValueError):
        rna.aggregate_to_genes(quant, pd.DataFrame({"transcript_id": ["other"], "gene_id": ["G"], "gene_name": ["X"]}))


def test_collapse_to_symbols_keeps_lowest_ensembl_id_regardless_of_value():
    values = pd.Series({"ENSG2": 100.0, "ENSG1": 1.0, "ENSG3": 5.0})
    symbols = pd.Series({"ENSG1": "DUP", "ENSG2": "DUP", "ENSG3": "UNIQ"})
    collapsed, dropped = rna.collapse_to_symbols(values, symbols)
    assert collapsed["DUP"] == 1.0
    assert collapsed["UNIQ"] == 5.0
    assert dropped == 1


# ---- ATAC ------------------------------------------------------------------


def test_tn5_shift_plus_and_minus_strand():
    reads = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [100, 100], "end": [136, 136], "strand": ["+", "-"]})
    cuts = atac.tn5_cut_sites(reads)
    assert cuts["pos"].tolist() == [104, 135 - 5]


def test_filter_reads_drops_noncanonical_and_blacklisted():
    reads = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chrM", "chrY", "chrUn_x", "chr2"],
            "start": [0, 1000, 5, 5, 5, 50],
            "end": [36, 1036, 41, 41, 41, 86],
            "strand": ["+"] * 6,
        }
    )
    blacklist = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [1030, 1040], "end": [1045, 2000]})
    kept, counts = atac.filter_reads(reads, blacklist)
    assert kept["start"].tolist() == [0, 50]
    assert counts["reads_chrM"] == 1
    assert counts["reads_noncanonical_removed"] == 3
    assert counts["reads_blacklist_removed"] == 1
    assert counts["reads_final"] == 2


def test_bin_counts_uses_floor_of_position():
    cuts = pd.DataFrame({"chrom": ["chr1"] * 4 + ["chr2"], "pos": [0, 199, 200, 401, 10]})
    counts = atac.bin_counts(cuts)
    assert counts.values.tolist() == [["chr1", 0, 2], ["chr1", 1, 1], ["chr1", 2, 1], ["chr2", 0, 1]]


def test_normalize_matches_bird_formula():
    assert atac.normalize([0], 1000)[0] == 0.0
    assert atac.normalize([10], atac.BIRD_N)[0] == pytest.approx(np.log2(11))
    assert atac.normalize([10], 2 * atac.BIRD_N)[0] == pytest.approx(np.log2(6))


def test_bin_interval_matches_perd_representation():
    starts, ends = atac.bin_interval(np.array([0, 3]))
    assert starts.tolist() == [0, 600]
    assert ends.tolist() == [199, 799]


def test_contained_grid_bins_use_closed_bins():
    enhancers = pd.DataFrame({"chrom": ["chr1", "chr1", "chr1"], "start": [150, 400, 400], "end": [650, 599, 598]})
    grid = atac.contained_grid_bins(enhancers)
    # [150, 650] contains bins 1 and 2; [400, 599] contains bin 2 exactly; [400, 598] contains none.
    assert grid["bin"].tolist() == [1, 2]
    assert len(atac.contained_grid_bins(enhancers.iloc[[2]])) == 0


def test_enhancer_openness_max_rule_zero_fill_and_exclusions():
    enhancers = pd.DataFrame(
        {
            "enhancer_id": ["chr1:200-599", "chr1:1000-1399", "chrY:200-599", "chr1:2001-2150"],
            "chrom": ["chr1", "chr1", "chrY", "chr1"],
            "start": [200, 1000, 200, 2001],
            "end": [599, 1399, 599, 2150],
        }
    )
    values = pd.DataFrame({"chrom": ["chr1", "chr1", "chr1"], "bin": [0, 1, 2], "value": [9.0, 1.5, 3.0]})
    openness = atac.enhancer_openness(enhancers, {"runA": values})
    assert openness.loc["chr1:200-599", "runA"] == 3.0  # bins 1-2 contained; bin 0 (value 9) is not
    assert openness.loc["chr1:1000-1399", "runA"] == 0.0  # contains bins with no cut sites
    assert np.isnan(openness.loc["chrY:200-599", "runA"])  # chrY reads discarded
    assert np.isnan(openness.loc["chr1:2001-2150", "runA"])  # no complete bin inside
    assert openness.index.tolist() == enhancers["enhancer_id"].tolist()


def test_bins_roundtrip(tmp_path):
    counts = pd.DataFrame({"chrom": ["chr1", "chrX"], "bin": [3, 7], "count": [2, 5]})
    path = tmp_path / "run.npz"
    atac.save_bins(path, counts, {"reads_final": 7})
    loaded, meta = atac.load_bins(path)
    pd.testing.assert_frame_equal(loaded, counts, check_dtype=False)
    assert meta == {"reads_final": 7}


def test_tss_profile_is_strand_aware():
    cuts = pd.DataFrame({"chrom": ["chr1", "chr1"], "pos": [1010, 4990]})
    tss = pd.DataFrame({"chrom": ["chr1", "chr1"], "pos": [1000, 5000], "strand": ["+", "-"]})
    profile = atac.tss_profile(cuts, tss, flank=50)
    # Both cuts are 10 bp downstream of their TSS.
    assert profile[50 + 10] == 2
    assert profile.sum() == 2


def test_promoter_counts_window():
    cuts = pd.DataFrame({"chrom": ["chr1"] * 3, "pos": [0, 500, 2500]})
    tss = pd.DataFrame({"chrom": ["chr1", "chr1"], "pos": [400, 2400], "strand": ["+", "+"], "gene_name": ["A", "B"]})
    counts = atac.promoter_counts(cuts, tss, window=500)
    assert counts.to_dict() == {"A": 2, "B": 1}


def test_downsample_fastq_records_keeps_whole_records():
    from perd_perturbation.runs import downsample_fastq_records

    lines = [f"@r{i}\nACGT\n+\nIIII\n" for i in range(10)]
    flat = [line + "\n" for record in lines for line in record.strip().split("\n")]
    kept = list(downsample_fastq_records(flat, total=10, n=4, seed=1))
    assert len(kept) == 16
    assert all(kept[i].startswith("@r") for i in range(0, 16, 4))
    assert kept == list(downsample_fastq_records(flat, total=10, n=4, seed=1))
    with pytest.raises(ValueError):
        list(downsample_fastq_records(flat, total=11, n=4, seed=1))


def test_five_prime_ends():
    reads = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [100, 100], "end": [136, 136], "strand": ["+", "-"]})
    assert atac.five_prime_ends(reads)["pos"].tolist() == [100, 135]


def test_cut_sites_roundtrip_and_bins(tmp_path):
    cuts = pd.DataFrame({"chrom": ["chr2", "chr1", "chr1", "chrX", "chr1"], "pos": [5, 400, 10, 7, 10]})
    path = tmp_path / "cuts.npz"
    atac.save_cut_sites(path, cuts, {"reads_final": 5})
    loaded, meta = atac.load_cut_sites(path)
    assert loaded.values.tolist() == [["chr1", 10, 2], ["chr1", 400, 1], ["chr2", 5, 1], ["chrX", 7, 1]]
    assert meta == {"reads_final": 5}
    expanded, _ = atac.load_cut_sites(path, expand=True)
    assert len(expanded) == 5
    pd.testing.assert_frame_equal(atac.bin_counts(expanded), atac.bin_counts(cuts))
