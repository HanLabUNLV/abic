# Network Generation Process

This script describes the processing of HiC data in various ways to generate chromatin interaction networks. The basic outline looks like this

1. Generate base HiC networks
2. Run fithic2 to call significant loops
3. Filter networks based on fithic loops and presence in gasperini
4. Label the roles of enhancers in the network

We begin by generating the base networks from the hic data. This is achieved by running the script `generate_base_networks.py`. The script takes the name of a gene as the only command line argument and is made to be used with linux parallel to efficiently produce the networks. An example of how to use the script is as follows, first in serial, then in parallel. 

`python src/network/generate_base_networks.py ADAMTS14`

`cat gene.list | parallel --max-args 1 -j 8 -I% python src/network/generate_base_networks.py %`

This will generate the network objects saved as python pickle objects, which themselves are instances of `networkx` objects into a directory called `gene_networks/`.

To filter the networks based on only significant edges, we run fithic2 (you will need a GPU optimized machine)

`bash src/network/call_loops.sh`

From those significant loop calls which are housed in the directory `fithic_loops`, we can filter the edges of the networks we generated previously. We also want to filter the nodes of the graphs to only contain experimentally validated enhancers in Gasperini et al. The script for this task is `generate_validated_networks.py` and also takes the name of a gene as the command line argument.

`cat gene.list | parallel --max-args 1 -j 8 -I% python src/network/generate_validated_networks.py %`

This script will generate new network objects in a directory called `gene_networks_filtered` which houses the networks filtered up to this point. 

The next step is to label the enhancers based on their distance on the network from the promoter of the gene. You can run the script in the same way as the others.

`cat gene.list | parallel --max-args 1 -j 8 -I% python src/network/add_roles.py %`

The networkx objects now have the roles as attributes attached to each node in the graph. These roles can be harvested and merged with the main training matrix using the following command

`python src/network/append_e_roles.py`
