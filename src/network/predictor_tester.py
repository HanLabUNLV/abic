#this file serves to stop frontend data from needing to be processed, a buffer file with the contents of enhancers was made 
#as a stop gap to save the progress of the pipeline. This should ensure seamless integration into the ABC model
from predictor import *
import pandas as pd
from re import split as resplit
from scipy.stats import gmean

#args mostly taken from predict.py defaults
hic_dir = 'example_chr22/input_data/HiC/raw/'
hic_resolution = 5000
hic_type = 'juicebox'
tss_hic_contribution = 100
window = 5000000 #size of window around gene TSS to search for enhancers
hic_gamma = .87
args = {'hic_dir':hic_dir,'hic_resolution':hic_resolution,'hic_type':hic_type,'tss_hic_contribution':tss_hic_contribution,'window':window,'hic_gamma':hic_gamma}
chromosome = 'chr22'

def add_attr_network(network, enhancers):
    vs = network.vs['name']
    for i in vs:
        local_enhancer_attr = {'local_enhancers':node_2_coord(i, enhancers)}
        activity = []
        for enh in local_enhancer_attr['local_enhancers']:
            chrm, start, end = enh
            if start!=end:
                activity.append(enhancers.loc[(enhancers['chr']==chrm) & (enhancers['start']==start) & (enhancers['end']==end),'activity_base'].mean())
            else: #is a promoter, activity not taken into account?? could use promoter activity quantile to get estimate
                activity.append(0)
        local_enhancer_attr['activity'] = activity
        network.vs.find(i)['enhancers'] = local_enhancer_attr

def contact_network_to_df(network):
    pairs = pd.DataFrame(columns = ['chr', 'viewStart', 'viewEnd', 'contactStart', 'contactEnd', 'view_node','contact_node'])
    for i in network.vs['name']:
        node = network.vs.find(i)
        viewpoint_enh = node['enhancers']['local_enhancers']      
        for vp in viewpoint_enh:
            for nb in node.neighbors():
                for con in nb['enhancers']['local_enhancers']:
                    #add the values to the pairs dataframe, then pass that into the HIC calculator
                    pairs = pairs.append({'chr':vp[0], 'viewStart':vp[1], 'viewEnd':vp[2], 'contactStart':con[1], 'contactEnd':con[2], 'view_node':node['name'],'contact_node':nb['name']}, ignore_index=True)
    return pairs

def calculate_distal_contact(enh, prm, path, network):
    contacts = []
    for i in range(len(path)-1):
        view_node = path[i]   
        contact_node = path[i+1]
        edge = network.es.select(_source=view_node, _target=contact_node)
        contacts.append(edge['avg_contact'][0])
    if ((len(contacts)>0) and (sum(contacts)>0)):
        return(gmean(contacts))
    else:
        return(0)
#enhancers with activity scores calculated
enhancers = pd.read_csv('enhancers_abcd.csv')

#create network, find communities
network = populate_network('loop_anchors.bed','merged_loops.bedpe')
communities_ig = network.community_fastgreedy()
communities = igraph_community_membership(communities_ig, network.vs['name'])
#print(network.get_shortest_paths(to='chr22_40920000_40930000', v='chr22_40830000_40840000'))

#check how many enhancer-promoter connections can be found
ep_cnxn = 0
no_cnxn = 0
ep_multi_cnxn = 0
def valid_connections(enhancers, network):
    enhancers = deepcopy(enhancers.loc[enhancers['class']!='promoter',])
    enhancers['enhID'] = enhancers.chr.str.cat([enhancers.start.astype(str), enhancers.end.astype(str)], sep='_')
    enhancers['promoterID'] = enhancers.chr.str.cat([enhancers.TargetGeneTSS.astype(str), enhancers.TargetGeneTSS.astype(str)], sep='_')
    enhIDs = enhancers.enhID.tolist()
    promoterIDs = enhancers.promoterID.tolist()
    valid_connections = []
    checked_enh = []
    for i in range(len(enhIDs)):
         if enhIDs[i] not in checked_enh:
           enh_coord = coord_2_node(enhIDs[i])
           if len(enh_coord) >0:
               prom_coord = coord_2_node(promoterIDs[i])
               if len(prom_coord) > 0:
                   enh_coord = enh_coord.as_df()
                   prom_coord = prom_coord.as_df()
                   view_node = list(set(enh_coord.Chromosome.str.cat([enh_coord.Start.astype(str),enh_coord.End.astype(str)], sep='_').tolist()))[0]
                   contact_node = list(set(prom_coord.Chromosome.str.cat([prom_coord.Start.astype(str),prom_coord.End.astype(str)], sep='_').tolist()))[0]
                   for community in communities:
                       if ((view_node in community) and (contact_node in community)):
                           path = network.get_shortest_paths(v=view_node, to=contact_node)
                           if len(path[0])>0:
                               valid_connections.append([enhIDs[i], promoterIDs[i]])
                               ep_cnxn += 1
                               if len(path[0])>2:
                                   ep_multi_cnxn +=1
           else:
               checked_enh.append(enhIDs[i])
    return(valid_connections)
    #with open('valid_connections','w') as f:
    #    f.writelines(['\t'.join(c)+'\n' for c in valid_connections])
    #ex22 was [231, 0, ,75] -- took forever, don't run again

valid_connections = []
with open('valid_connections','r') as f:
    for line in f:
        valid_connections.append(line.strip().split('\t'))

#add activity and enhancers to network
#add_attr_network(network, enhancers)

#map contact to edges
#pairs = contact_network_to_df(network)
#hic_file, hic_norm_file, hic_is_vc = get_hic_file(chromosome, args['hic_dir'], hic_type = args['hic_type'])
#pairs = add_hic_ABCD(pairs, hic_file, hic_norm_file, hic_is_vc, chromosome, args)
#pairs.to_csv('pairs.csv')
pairs = pd.read_csv('pairs.csv')
pairs['edgeID'] = pairs.view_node.str.cat(pairs.contact_node, sep=':')  #pairs['view_node'] + ':' pairs['contact_node']
pairs['viewID'] = pairs.chr.str.cat([pairs.viewStart.astype(str), pairs.viewEnd.astype(str)], sep='_')
pairs['contactID'] = pairs.chr.str.cat([pairs.contactStart.astype(str), pairs.contactEnd.astype(str)], sep='_')
pairs['connectionID'] = pairs.viewID.str.cat(pairs.contactID, sep=':')
cmpnt = [resplit('_|:', x) for x in pairs.connectionID.tolist()]

#connectionID has no promoters identifiable by start==end (promoters are given a start-end coordinate of their TSS, but I remember defining them as 500bp around it at one point)
#print([(x[1]==x[2]) or (x[4]==x[5]) for x in cmpnt])
#iter through edges
def add_contact_network(network, pairs):
    for e in network.es:
        v1 = network.vs.find(e.source)['name']
        v2 = network.vs.find(e.target)['name']
        edge_data = pairs.loc[((pairs['view_node']==v1)&(pairs['contact_node']==v2))]# & pairs['contact_node']==v2) | (pairs['view_node']==v2 & pairs['contact_node']==v1)]
        connections = {'connectionID':edge_data['connectionID'].tolist(), 'contact':edge_data['hic_contact'].tolist()}
        e['connections'] = connections
        if len(connections['contact'])>0:
            e['avg_contact'] = sum(connections['contact'])
        else:
            e['avg_contact'] = 0
    #contact data successfully loaded into network
def add_ABCD_score(enhancers, network, valid_connections):
     enhancers['hic.ABCD'] = 0
     for ep in valid_connections:
         enh, prm = ep
         chrm, enh_start, enh_end = enh.split('_')
         tss = prm.split('_')[1]
         enh_df = coord_2_node(enh).as_df()
         enh_node = enh_df.Chromosome.str.cat([enh_df.Start.astype(str), enh_df.End.astype(str)], sep='_').tolist()[0]  
         prm_df = coord_2_node(prm).as_df()
         prm_node = prm_df.Chromosome.str.cat([prm_df.Start.astype(str), prm_df.End.astype(str)], sep='_').tolist()[0]  
         path = network.get_shortest_paths(v=enh_node, to=prm_node)
         cnx1 = ':'.join([prm,enh])
         cnx2 = ':'.join([enh,prm])
         #what I need is to walk the path of the network and get all avg_contact for each edge along the way
         if len(path[0])>1:
             dc = calculate_distal_contact(enh, prm, path[0],network)
             #then find the entry in enhancers and add the distal contact to those who need it
             enhancers.loc[(enhancers['chr']==chrm) & (enhancers['start']==int(enh_start)) & (enhancers['end']==int(enh_end)) & (enhancers['TargetGeneTSS']==int(tss)),'hic.ABCD'] = dc
#enhancers.to_csv('enhancers_abcd_hic.csv')
print(enhancers.loc[(enhancers['hic.ABCD'] - enhancers['hic_contact_pl_scaled_adj'])>=0,])
