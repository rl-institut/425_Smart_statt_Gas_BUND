### Get Start

```
conda env list
conda env remove --name pypsa-eur
conda env create -f envs/environment.yaml
conda env create -f envs/macos-pinned.yaml
conda activate pypsa-eur


snakemake -call all --configfile work/base/seprateEU.yaml --cores all
snakemake -call all --configfile work/base/de27Node2045.yaml --cores all
snakemake -call all --configfile config/config.yaml --cores all


snakemake prepare_sector_networks --configfile research/config/transmission_mean.yaml --cores all


// scenario
snakemake -call all --configfile research/config/transmission_best.yaml --cores all

snakemake -call all --configfile research/config/h2_mean.yaml --cores all

// smart
snakemake -call all --configfile research/config/smart/smart-flex.yaml --cores all




```


### other information

DE 39



EU 节点限制为只有 DE
config_provider("countries")
