### Get Start

```
conda env list
conda env remove --name pypsa-eur
conda env create -f envs/environment.yaml
conda activate pypsa-eur


snakemake -call all --configfile work/base/seprateEU.yaml --cores all
snakemake -call all --configfile work/base/de27Node2045.yaml --cores all
snakemake -call all --configfile config/config.yaml --cores all


snakemake prepare_sector_networks --configfile config/config.yaml --cores all


// scenario
snakemake -call all --configfile work/scenario/biogas_best.yaml --cores all

snakemake -call make_summary --configfile work/scenario/biogas_mean.yaml --cores all



```


### other information

DE 39



EU 节点限制为只有 DE
config_provider("countries")
