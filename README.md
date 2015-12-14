![PaDuA](PaDuA.png)

A Python package for Proteomic Data Analysis, offering processing and analysis of the output of proteomics software [MaxQuant](http://maxquant.org).

# Installation

PaDuA is available via the Python package index at [PyPi](http://pypi.org) and can be installed in the usual way with:

    pip install padua
    
Once installed the package is available for import using:

    import padua
    
The package is organised into multiple submodules for different purposes, eg.

1. `io` for reading and writing both MaxQuant and Perseus format files (input/output)
1. `filters` for filtering data by quality and features
1. `process` incorporating experimental design, labels to index, expand-side-table (Perseus) and more
1. `normalization` for performing normalisation methods, e.g. remove column median
1. `annotations` adding annotation metadata for quantified proteins
1. `analysis` performing simple analyses, e.g. column correlations
1. `plots` standard plot outputs for overviews of data

# What is it for?

The goal is to provide a simple scripting approach to replicate many of the common steps for interacting with the output
of MaxQuant. Many of the steps implemented are based on similar steps used in the MaxQuant sister software 
[Perseus](http://141.61.102.17/perseus_doku/). While currently Perseus has more features, it has stability issues with
the larger datasets we are currently using. Having the processing steps implemented in Python allows for simple 
processing workflow scripts to be created and re-used.

# Examples

An example Phosphoproteomic label-free-quantification workflow would be as follows:

    import padua
    df = padua.io.read_maxquant('Phospho (STY)Sites.txt')

    df = padua.filter.filter_localization_probability(df)

    df = padua.filter.remove_reverse(df)
    df = padua.filter.remove_only_identified_by_site(df)
    df = padua.filter.remove_potential_contaminants(df)

    # Use standard Pandas dataframe manipulations to set an index
    df.set_index('Proteins', inplace=True)
    df = df.filter(regex='Intensity ')

    df = df.process.expand_side_table(df)

    # Remove the multiplicity column
    df = df.filter(regex='Intensity ')

    df = padua.process.apply_experimental_design(df, 'experimentalDesignTable.txt')

    # The result of this step will be a multilevel index Class, Replicate
    # built by matching sample labels using regex
    indices = [
        ('Class': '^(.*)_',
        ('Replicate': '_(\d)', 
    ]
    df = padua.process.build_index_from_labels(df, indices)

# Future

Provided functions are based on our current requirements, but will be expanded in future. 

# License

PaDuA is open source software and available under the BSD 2-clause (Simplified) license.
