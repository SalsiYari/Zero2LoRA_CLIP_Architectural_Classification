DA NOTARE CHE:
i modelli supervisionati BATTONO zero-shot essendo allenati su esempi etichettati.

se si confrontano LR/KN vs LR/KNN+ZS con pesi banali come i miei (tipo media 50/50) è normale che il contributo dell'ensamble zero-shot peggiori leggermente l'accuracy.

- ZeroShot non dipende dal train set
-->può aiutare nelle classi poco rappresentate o dove LR/KNN overfittano
-->può ridurre errori gravi su classi rare
--> rende il modello più flessibile (introduzione nuove classi in futuro)
-->oltre alla conoscenza visva porta la conoscenza semantica



per correggere la media la media 50/50 usa una griglia pesi con (xLR , yKNN, zXS) lasciando a zero-shot un peso piccolo ma positivo)



---> se obiettivo è massimizzare accuracy su dataset attuale -> spento
---> se obirttivo è classificatore + robusto,generalizzabile e interpretabile ->acceso 
