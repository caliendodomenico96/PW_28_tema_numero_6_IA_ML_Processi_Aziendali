# Customer Segmentation — Streamlit (Simulazione multi-cliente)

Un'unica applicazione Streamlit con le pagine **Analisi dei cluster** e
**Nuovi clienti**. Entrambe condividono training e numero di cluster K.
La seconda pagina mantiene una **simulazione multi-cliente** modificabile durante
la sessione, senza salvare i nuovi record nei CSV di training.

## Avvio

Dalla root del progetto e nello stesso ambiente virtuale del training:

```bash
pip install -r requirements.txt
pip install -r requirements_frontend.txt
streamlit run frontend/app.py
```

Apri `http://localhost:8501`.

## Refresh automatico del catalogo

L'entrypoint `frontend/app.py` usa `st.fragment(run_every="10s")` per leggere un
**catalogo non cached** dei dataset in `data/` e dei run in `results/kmeans/`.

- Quando appare un nuovo training completo (`training_report.json` scritto per
  ultimo), il tag si aggiorna e viene selezionato automaticamente il nuovo run.
- Quando viene rimossa una cartella in `data/`, la lista dataset viene
  aggiornata automaticamente. Se un training dipende da quel dataset, il run
  diventa temporaneamente **non selezionabile** e la sidebar ne segnala il motivo.
- Quando viene cancellata una cartella di training, il tag scompare e
  l'app seleziona il run utilizzabile più recente, se esiste.
- Il controllo *non* lancia il training: dopo aver generato un nuovo dataset è
  necessario eseguire `python src/2_kmeans_training.py` per ottenere un nuovo
  modello. La sidebar segnala la differenza fra ultimo dataset e training.
- Non ridisegna grafici e tabelle ogni 10 secondi: esegue il rerun completo
  **soltanto quando cambia lo stato del catalogo**.

## Pagina Analisi dei cluster

Visualizza esclusivamente le **assegnazioni storiche originali** dei cluster,
con scatter plot, profili medi, filtri (zona, categoria, periodo) e grafici
salvati durante il training. Non include i nuovi clienti classificati.

## Pagina Nuovi clienti: simulazione multi-cliente

- **Aggiungi cliente** con i campi anagrafici, zona, categoria, iscrizione,
  numero richieste, spesa e recency. Le variabili descrittive non entrano in K-Means.
- **Importa CSV** per aggiungere più clienti insieme (massimo 1000 nella
  sessione, file massimo 2 MB), anche ad una simulazione già avviata.
- Ogni inserimento riceve un ID stabile e progressivo (`NEW_CUSTOMER_001`, ...).
  Modificare/eliminare un cliente non cambia gli ID rimanenti e non ricicla il numero.
- **Clicca una riga nella tabella** per selezionare il cliente; dalla scheda
  *Modifica selezionato* puoi aggiornare i dati e ricalcolare la predizione.
- **Elimina selezionato** oppure **Svuota simulazione** con conferma.
- Lo scatter Plotly può mostrare soltanto il cliente selezionato oppure tutti
  i simulati con stelle; la stella più grande identifica il selezionato.
- I grafici disponibili sono **Richieste e spesa** e **PCA 2D** con centroidi.
  La PCA viene stimata esclusivamente sui clienti storici e non cambia i cluster.
- La tabella *Distribuzione dei cluster · scenario temporaneo* mette a confronto
  i clienti storici, quelli simulati e i conteggi aggregati. Il modello NON si aggiorna.
- Il pulsante **Esporta tutti i clienti simulati (.csv)** produce le predizioni
  con ID, anagrafica, cluster e distanza dal centroide.

### Durata dei dati e cambio del modello

I clienti sono conservati **nella sessione Streamlit del browser**, separati
per training. Sono mantenuti durante la navigazione tra le pagine e durante i
refresh dell'app, ma non dopo la chiusura/reset della sessione o il riavvio.

Quando si cambia `K` all'interno del medesimo training, i clienti vengono
**riclassificati** dal modello selezionato, senza riaddestrare K-Means. Quando
si cambia training, la pagina usa la simulazione associata a quel training.
Le date sono interpretate rispetto alla data di riferimento del training.

La pagina **Analisi dei cluster** include anche il pannello **Validazione dei cluster**:
legge l'ultima validation completa disponibile per il training e il K selezionati,
mostra `silhouette_k<K>.png` e, in un riquadro espandibile, le metriche K confrontate
(Davies-Bouldin, silhouette e Calinski-Harabasz). Quando una validation viene
aggiunta o rimossa sotto `results/kmeans/<run>/validations/`, il monitor ogni
10 secondi aggiorna automaticamente la pagina. Le silhouette originali e le
metriche sono calcolate sul dataset completo, **non sui dati filtrati**. Non
sono necessari nuovi training o una nuova validazione se gli artefatti esistono.

La pagina **Analisi dei cluster** mostra sempre e solo i clienti del CSV
originale e le sue statistiche; i nuovi clienti restano in una simulazione
separata. Il catalogo dei training continua a controllare cartelle aggiunte e
rimosse ogni 10 secondi.

### CSV di esempio

```csv
id_cliente;nome;cognome;email;zona;categoria_preferita;data_iscrizione;numero_richieste;importo_totale_speso;giorni_ultima_richiesta
E01;Mario;Rossi;new.customer.001@example.com;Roma;Configurazione backup;01/06/2026;2;100,00;14
```

Sono obbligatori `data_iscrizione`, `numero_richieste`, `importo_totale_speso`
e `giorni_ultima_richiesta`. I campi derivati `importo_medio_speso` e
`data_ultima_richiesta` vengono ricalcolati e, se forniti, verificati.
Gli ID importati vengono conservati in `id_cliente_origine`.
Il backend valida gli importi, le date e le richieste prima di accettare un
batch: in caso di errore nessun record del batch viene aggiunto.

La classificazione utilizza `build_feature_frame()` e gli oggetti locali
`preprocessor.joblib`/`kmeans_k<K>.joblib` mediante `transform()` e `predict()`;
nessun `fit()` o scrittura sui dati storici.

## Esecuzione dei test

```bash
python -m unittest discover -s tests -v
```

I test coprono lettura e rimozione delle cartelle, inserimento multiplo,
ID stabili, modifiche/eliminazioni, importazione atomica, ricalcolo al cambio
di K, grafici Plotly (singolo e tutti i clienti) e integrità dei dati storici.

**Nota sicurezza:** leggere soltanto i modelli `joblib` creati e salvati dal
proprio training, usando versioni compatibili di scikit-learn.
