#!/usr/bin/env python

import re
import yaml
import numpy
import pandas
import warnings
import requests
from copy import copy

from Bio.KEGG import REST

from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from skbio.stats.distance import DistanceMatrix, permanova

# ---------------------------------------------------------------------------------#


def read_data(fileName, index=False, datatype=False):
    """Consolidate read in steps"""
    sep = "\t" if ".tsv" in fileName else ","

    if index != False:
        if index.isnumeric():
            data = pandas.read_csv(fileName, sep=sep, low_memory=False, header=0, index_col=index)
        else:
            data = pandas.read_csv(fileName, sep=sep, low_memory=False, header=0)
            for x in data.columns: data.rename(columns={str(x): sanitize_str(x)}, inplace=True)
            data.index = data[sanitize_str(index)]
            data = data.drop(sanitize_str(index), axis=1)
    else:
        data = pandas.read_csv(fileName, sep=sep, low_memory=False, header=0)

    if datatype != False:
        data = data.astype(datatype)

    return data


def random_splitter(data, grouping, size=0.2, alg="manhattan", p_max=0.05, perm=99, seed=42):
    """Recursive distance check to generate random training/test sets"""

    for i in range(0, perm):
        X_train, X_test, y_train, y_test = train_test_split(
            data, grouping, test_size=size, random_state=seed, stratify=grouping
        )
        reordered = list(X_train.index) + list(X_test.index)
        groups = list(numpy.repeat(0, X_train.shape[0])) + list(numpy.repeat(1, X_test.shape[0]))
        distance_matrix = pairwise_distances(data.reindex(reordered), metric=alg)
        pval = permanova(DistanceMatrix(distance_matrix), groups, permutations=perm)["p-value"]
        if pval > p_max: break

    return X_train, X_test, y_train, y_test


def read_config(yml, key=None):
    """Read in config dictionary"""
    with open(yml) as stream:

        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return

        if key is not None:
            config = config[key]

        return config


def fix_nan(data):
    """Update malformed NA strings"""
    for x in ["NaN", "nan", "Nan", "NA", "Na", "na", "none", "None", "n/a", "N/A", "N/a", "n/A"]:
        data.replace(x, numpy.nan, inplace=True)
    return data


def sanitize_str(raw, verbose=False):
    """Correct name strings to be more automation friendly"""
    try:
        raw = int(raw)
    except:
        pass
    partial = re.split("[^a-zA-Z0-9]", str(raw))
    cleaned = "_".join([x.lower().strip() for x in partial if len(x.strip()) > 0])
    if verbose and raw != cleaned:
        print(f"{raw} ---> {cleaned}")
    return cleaned


def rarefy_abundances(table, target_reads=1e9):
    """Returns a subsampled table of abundances to a given level for all columns"""
    rarefied_table = pandas.DataFrame(columns=table.columns)

    for column in table.columns:
        column_total = table[column].sum()

        if column_total <= target_reads:
            rarefied_table[column] = table[column]
        else:
            rarefaction_factors = target_reads / column_total
            rarefied_table[column] = round(table[column] * rarefaction_factors)

    return rarefied_table


def aggregate_subset_sum(data, group, annotations):
    """Group catagories of interest"""
    data[group] = [sanitize_str(y, verbose=False) for y in data[group]]
    data = fix_nan(data)
    grouped = data.dropna(subset=group)
    rm_cols = copy(annotations)
    rm_cols.remove(group)
    grouped = grouped.drop(rm_cols, axis=1)
    grouped = grouped.groupby(group).sum()
    return grouped.T


def clean_annotation(names, hard=False):
    """Refomat IUPAC (or otherwise) chemical names"""
    counts = {}
    cleaned = []

    for raw in names:
        name = "_".join(raw.split()).replace(",", ".")
        name = name.replace("'", "prime")
        for x in "[]<>":
            name = name.replace(x, "")
        for y in "()":
            name = name.replace(y, "-")
        name = "-".join([z.lower() for z in name.split("-") if len(z) > 0])
        name = name.replace("._", "").replace("_.", "")

        if hard:
            partial = re.split("[^a-zA-Z0-9]", str(name))
            name = "_".join([x.lower().strip() for x in partial if len(x.strip()) > 0])

        try:
            counts[name] += 1
            cleaned.append(f"{name}_{counts[name]}")
        except KeyError:
            counts[name] = 1
            cleaned.append(name)

    return cleaned


def calc_corr_threshold(data, limit=500):
    """Calculate minimum R value based on features included"""
    cutoff = 0.0
    test = 10000
    data = data.abs()
    while test > limit:  # maximum comparisons to keep in correlation matrix
        cutoff += 0.01
        test_threshold = numpy.nanquantile(data, cutoff)
        df = data[data > test_threshold]
        df = df[df != 1.0]
        test = int(sum(df.count()) / 2)

    return round(test_threshold, 2)


def get_uniprot_name(uniprotID):
    """Use the Uniprot REST APi to get protein names"""
    x = uniprotID.split(".")[0]
    info = requests.get(
        f"https://rest.uniprot.org/uniprotkb/search?query={x}&fields=protein_name"
    ).text

    replaceDict = {":": "", "{": "", "}": "", "}": "", ",": "", "[": "", "]": ""}
    for key, value in replaceDict.items():
        info = info.replace(key, value)

    max_len = -1
    for ele in info.split('""'):
        if "UniProtKB" in ele or "ANTAGONIST" in ele or "SWOLLEN" in ele:
            continue
        elif ele in [
            "proteinDescription",
            "recommendedName",
            "primaryAccession",
            "entryType",
            "shortNames",
            "Precursor",
            "results",
        ]:
            continue
        elif len(ele) > max_len:
            max_len = len(ele)
            prot_name = ele.strip('"')

    return prot_name


def get_kos(path_id):
    """Get component KOs of a KEGG pathway"""
    try:
        path_data = REST.kegg_get(path_id).read()
        kos = (
            path_data.split("ORTHOLOGY")[-1].split("COMPOUND")[0].split("REFERENCE")[0].split("\n")
        )
        kos = [x.strip() for x in kos if len(x) > 0]
    except:
        kos = []

    return kos


def get_kegg_data_from_uniprot(uniprot, uniprot2kegg_dict):
    """Translate Uniprot gene IDs to KEGG genes + orthologies & pathways"""

    try:
        kegg_genes = uniprot2kegg_dict[uniprot]
    except KeyError:
        return [], [], []

    pathways = set()
    orthologies = set()
    for kegg_gene in kegg_genes:
        try:
            gene_data = REST.kegg_get(kegg_gene).read()  # query and read each pathway
        except:
            return [], [], []

        prev = ""
        for line in gene_data.rstrip().split("\n"):
            section = line[:12].strip()  # section names are within 12 columns
            if not section == "":

                if section == "ORTHOLOGY":
                    orthology = line[12:].split()[0]
                    orthologies |= set([orthology])
                    prev = section

                elif section == "PATHWAY":
                    pathway = line[12:].split()[0]
                    pathway = "ko" + "".join(filter(str.isdigit, pathway))
                    pathways |= set([pathway])
                    prev = section

                elif prev == "PATHWAY":
                    break

            elif prev == "PATHWAY":
                pathway = line[12:].split()[0]
                pathway = "ko" + "".join(filter(str.isdigit, pathway))
                pathways |= set([pathway])

    return kegg_genes, list(orthologies), list(pathways)


def translate_uniprot(hits, keggdict):
    """Translate Uniprot hits to KEGG information"""
    gene_kegg_hits = []
    fail = 0
    total = 0
    for x in hits["uniprot"]:
        total += 1
        genes, kos, maps = get_kegg_data_from_uniprot(x.split(".")[0], keggdict)
        if len(kos) > 0:
            gene_kegg_hits.append([x, ",".join(genes), ",".join(kos), ",".join(maps)])
        else:
            fail += 1
    gene_kegg_hits = pandas.DataFrame(gene_kegg_hits, columns=["uniprot", "kegg", "ko", "pathway"])
    success = total - fail
    success_perc = round((success / total) * 100, 2)
    print(f"Successful gene annotations: {success} of {total} ({success_perc}%)")

    return gene_kegg_hits
