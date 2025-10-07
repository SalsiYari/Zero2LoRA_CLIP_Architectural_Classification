SPIEGAZIONE DEL CONTENUTO DELLA CARTELLA:

coop_cluster_perClass.py e coop_cluster_shared.py -->sono lo stesso file, quest'ultimo è commentato profondamente. 
                                                  --->differiscono per setting di default (per_class o shared prompt learning)

coop_cluster_promptLearning --> si sarebbe dovuto chiamare coop_cluster_promptEnsamble ma mi sono sbagliato (pacca cambiare nome). 
                            -->differisce dai precedenti in quanto allena un ensamble di prompts

-->! risultati nel file *.log

-->! per comprendere i'ordine dei  risultati GUARDARE PRRIMA il file *.sh che ha permesso l'esecuzione dei vari modelli 

nota:
Funziona senza modifiche (a parte batch/VRAM) con i checkpoint già supportati da OpenCLIP:
ViT-L/14, ViT-H/14, ViT-g/14, ViT-bigG/14 con pesi laion2b/, datacomp1b, metaclip_fullc/metaclip2_worldwide (se disponibili).
Dove NON è drop-in:
SigLIP2, PE-Core bigG/L (meta), CLIPA, VITamin-XL, gopt-16: spesso non hanno pesi OpenCLIP o hanno text tower/tokenizer diversi


