# Dataset sintetico — segmentazione clienti di una startup informatica

**Project Work L-31.** Il progetto simula una startup italiana B2C che offre assistenza informatica da remoto: configurazione software, account, scansione diagnostica e backup. I dati sono **interamente sintetici**; nomi, cognomi e indirizzi `example.com` servono solo per la presentazione.

## Requisiti di sistema e librerie

Ambiente di riferimento per l'esecuzione locale (Linux x86_64):

| Componente | Versione / requisito |
|---|---|
| Sistema operativo | **Ubuntu 24.04 LTS** (64 bit) |
| Interprete Python | **Python 3.12.2** (versione di riferimento; altre patch 3.12.x possono essere compatibili) |
| `pip` e `venv` | Necessari per installare le librerie in un ambiente isolato |
| Browser | Browser moderno per visualizzare la dashboard Streamlit e i report HTML |
| Accesso a Internet | Necessario per il download iniziale delle dipendenze PyPI |

Le librerie Python sono dichiarate nei file di progetto; **non è necessario installarle singolarmente**:

| File | Librerie principali | Impiego |
|---|---|---|
| `requirements.txt` | NumPy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1, Matplotlib 3.11.2, joblib 1.6.0, fg-data-profiling 4.20.0 | Generatore, profiling, preprocessing, training, validation e grafici |
| `requirements_frontend.txt` | Streamlit 1.67.0, Plotly 7.1.0| Dashboard web e interazione con i cluster |



## Installazione

Eseguire i comandi in un terminale Bash di Ubuntu 24.04.

### 1. Verificare Python e installare gli strumenti di sistema

```bash
python3.12 --version  # versione di riferimento: Python 3.12.2
```

Se Python 3.12 o il supporto `venv` non sono disponibili:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip
```

### 2. Entrare nella cartella root del progetto

La cartella è quella che contiene `src/`, `libs/`, `config/`, `frontend/` e i due file `requirements`:

```bash
cd /percorso/alla/data_customer_segmentation_application
```

### 3. Creare e attivare l'ambiente virtuale

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
```

Il terminale dovrebbe mostrare `(.venv)` all'inizio del prompt. L'ambiente si riattiva con `source .venv/bin/activate` a ogni nuova sessione terminale; per uscirne si usa `deactivate`.

### 4. Installare tutte le dipendenze

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements_frontend.txt
```

`requirements.txt` è necessario anche per l'EDA: lo script di generazione importa `data_profiling` e produce il report statistico HTML.

### 5. Verificare l'ambiente

```bash
python -m pip check
python -c "import numpy, pandas, sklearn, matplotlib, joblib, streamlit, plotly, data_profiling; print('Ambiente pronto')"
```

Se il controllo restituisce errori di dipendenza, risolverli **prima** di eseguire il training.

## Esecuzione (running)

Tutti i comandi si lanciano **dalla root del progetto**, con il virtual environment attivo. In ogni nuovo terminale:

```bash
cd /percorso/alla/data_customer_segmentation_application
source .venv/bin/activate
export PYTHONPATH="$PWD"
```

**Attenzione:** Bash distingue maiuscole e minuscole: il nome corretto della variabile è `PYTHONPATH` e la variabile che contiene la directory corrente è `$PWD` (non `$pwd`). L'export vale soltanto per il terminale corrente.

### Step 1 — Generazione del dataset

```bash
python src/1_dataset_tool_generator.py
```

La configurazione si modifica in `config/dataset_generator_config.ini`. Il generatore salva CSV, JSON e profiling HTML in una nuova sottocartella UTC, senza sovrascrivere le generazioni precedenti:

```text
data/
└── <timestamp_UTC>/
    ├── customer_history_dataset.csv
    ├── customer_history_dataset.metadati.json
    ├── customer_history_dataset.recap.json
    ├── dataset_profiling.html
    └── richieste_sintetiche.csv   # solo se esporta_richieste = yes
```

### Step 2 — Preprocessing e training K-Means

```bash
python src/2_kmeans_training.py
```

Le impostazioni si trovano in `config/kmeans_config.ini`. Per usare automaticamente il dataset appena generato, impostare:

```ini
[KMEANS]
dataset_csv = latest
```

Il training salva modelli, assegnazioni, metriche e grafici in `results/kmeans/<timestamp_UTC>/`.

### Step 3 — Validation K-Means

```bash
python src/3_kmeans_validation.py
```

Per validare l'ultimo training completato, impostare:

```ini
[VALIDATION]
training_run = latest
```

La validation **non riaddestra** il modello: verifica integrità, riproducibilità e metriche, e salva le silhouette dettagliate in una sottocartella `validations/<timestamp_UTC>/` del training selezionato.

### Step 4 — Avvio della dashboard Streamlit

```bash
streamlit run frontend/app.py
```

Aprire <http://localhost:8501> nel browser. Dalla stessa GUI puoi scegliere **Analisi dei cluster** oppure **Nuovi clienti**. Il catalogo di dataset/training/validation si aggiorna ogni 10 secondi, ma **un nuovo CSV richiede comunque una nuova esecuzione dello Step 2** per ottenere il relativo modello.

Per arrestare Streamlit, usare `Ctrl+C` nel terminale.

### Step 5 — Test automatici (consigliati)

```bash
python -m unittest discover -s tests -v
```

È possibile ripetere gli step 1–3 con configurazioni differenti, mantenendo tutti gli esperimenti in cartelle distinte identificate da timestamp UTC.

---

## Requisiti della traccia e colonne

**Un record per cliente**, tra 300 e 600 clienti secondo la traccia (configurazione corrente: 600). Il file contiene:

| Campo | Significato |
|---|---|
| `id_cliente` | Identificativo cliente univoco, `C000001` ecc. |
| `zona` | Città italiana simulata, estratta da `[ZONE]` |
| `data_iscrizione` | Data di iscrizione in `GG/MM/AAAA` |
| `numero_richieste` | Numero di interventi completati e pagati **negli ultimi 90 giorni** |
| `importo_totale_speso` | Somma delle richieste **nella stessa finestra di osservazione**, in euro |
| `importo_medio_speso` | Spesa media per richiesta nella finestra; `0.00` se non ci sono richieste |
| `giorni_ultima_richiesta` | Giorni dalla richiesta più recente **nell'intera storia** |
| `categoria_preferita` | Categoria più frequente nell'intera storia; parità risolta alfabeticamente |
| `data_ultima_richiesta` | Data della richiesta più recente in `GG/MM/AAAA`, campo descrittivo |
| `nome`, `cognome`, `email` | Campi descrittivi facoltativi con dati sintetici, non usati nel clustering |

Una richiesta è **un intervento remoto completato e pagato**, non una telefonata o un semplice messaggio di assistenza. I prezzi sono positivi e arrotondati a centesimi; gli importi sono aggregati usando centesimi interi e `Decimal` per la media. Nel periodo senza richieste, `numero_richieste`, totale e media valgono zero, ma ogni cliente ha almeno un acquisto nella sua storia simulata.

**Attenzione:** `data_riferimento` nel file INI usa `GG-MM-AAAA`, mentre le date CSV/JSON descrittive usano `GG/MM/AAAA`. Con `26-10-2026` e `giorni_osservazione = 90` la finestra inclusiva è **29/07/2026–26/10/2026**. Il timestamp UTC di generazione è invece ISO 8601 per identificare univocamente gli output.

## Contesto realistico e aderenza alla traccia

La traccia richiede clienti **molto attivi**, **occasionali**, **recenti** e **inattivi**, con differenze plausibili per richieste e importi. Per non confondere la frequenza con l'anzianità o la capacità di spesa, il simulatore utilizza **quattro profili latenti di frequenza/continuità**, con un'ulteriore fascia osservata di attività intermedia:

| Profilo latente nell'INI | Peso | Richieste medie/mese durante la fase attiva | Ticket base medio |
|---|---:|---:|---:|
| `molto_attivi` | 0,12 | 2,70 | 38 € |
| `abituali` | 0,30 | 1,10 | 35 € |
| `occasionali` | 0,43 | 0,20 | 28 € |
| `inattivi` | 0,15 | 0,55 prima della sospensione | 35 € |

Il profilo latente `inattivi` ha una **cessazione simulata degli acquisti** tra 100 e 160 giorni prima della data di riferimento: conserva acquisti nella storia, ma non nella finestra osservata di 90 giorni. **Non tutti gli inattivi osservati appartengono però a quel profilo latente**: anche un cliente occasionale può non effettuare richieste durante la finestra.

I **recenti** non sono un profilo alternativo: si estraggono clienti con iscrizione da meno di 30 giorni tramite `quota_nuovi_clienti = 0.15` tra i profili non interrotti. Un recente può essere occasionale, attivo o molto attivo. Il 15% è la probabilità di simulazione tra i clienti eleggibili, non la percentuale garantita sull'intero dataset.

I **ticket premium** sono una caratteristica economica indipendente dalla frequenza, impostata in `[VALORE_ECONOMICO]`: `quota_clienti_ticket_premium = 0.15`, `moltiplicatore_ticket_premium = 2.50`. Questo consente a un cliente con poche richieste di avere importi elevati, senza introdurre un quinto profilo esclusivo. La quota premium riguarda il ticket base latente e non coincide necessariamente con il conteggio osservato di ticket medi almeno 70 €.

All'interno di ogni profilo non tutti i clienti sono identici: il tasso di richieste e il ticket base sono estratti con distribuzioni lognormali a variabilità controllata; gli intervalli tra richieste successive seguono una legge esponenziale. La prima richiesta avviene entro 7 giorni dall'iscrizione, perché il dataset rappresenta una coorte di clienti che hanno già acquistato. La categoria delle singole richieste dipende da una preferenza individuale probabilistica e i prezzi sono modulati dal tipo di servizio.

**I profili latenti non vengono esportati nel CSV.** Sono ipotesi di generazione e non etichette da ricostruire mediante K-Means. I pesi sono probabilità, non quote rigide di clienti per fascia osservata.

### Fasce di attività osservabili

Nel JSON, `distribuzione_comportamenti_osservati` è calcolata **solo dalle colonne CSV**, senza conoscere il profilo che ha originato il cliente.

| Fascia esclusiva | Definizione (90 giorni) |
|---|---|
| Inattivi | 0 richieste, ultima richiesta da almeno 90 giorni |
| Occasionali | 1–2 richieste |
| Attivi | 3–5 richieste |
| Molto attivi | Almeno 6 richieste |

Le fasce esclusive comprendono ogni cliente **una volta sola**. Le due caratteristiche trasversali sono **recente** (iscrizione da meno di 30 giorni) e **ticket medio elevato** (almeno 70 €, con richieste nel periodo): sono conteggi che possono sovrapporsi alle quattro fasce, e non vanno sommati a esse. Nel modello attuale *inattivi* e *nessuna richiesta nel periodo* sono lo stesso criterio osservato, quindi gli inattivi non sono più ripetuti fra le caratteristiche trasversali.

Le soglie descrittive `richieste_min_attivi`, `richieste_min_molto_attivi` e `ticket_medio_elevato_euro` in `[ANALISI_COMPORTAMENTI]` non modificano il dataset, ma solo il riepilogo osservato. Rivalutare le soglie se si modifica la finestra di osservazione.

### Risultati verificati con la configurazione predefinita

`n_clienti=600`, `seed=1`, riferimento `26/10/2026`, finestra inclusiva `29/07/2026–26/10/2026`.

| Fascia osservata | Clienti | Percentuale |
|---|---:|---:|
| Inattivi (0 richieste) | 199 | 33,2% |
| Occasionali (1–2) | 255 | 42,5% |
| Attivi (3–5) | 94 | 15,7% |
| Molto attivi (6+) | 52 | 8,7% |

In aggiunta, **66 recenti (11,0%)** e **39 con ticket medio almeno 70 € (6,5%)**. In totale, **1.187 interventi nel periodo**, **2.259 nella storia** e **51.708,10 € di spesa nel periodo**. Questi valori cambiano con `seed` e configurazione e **non sono dati osservati sul mercato reale**.

## Impostazioni del file INI

- `[DATASET_GENERATOR]`: `n_clienti`, `seed`, `data_riferimento`, `giorni_osservazione`, `giorni_attivita_azienda`, `quota_nuovi_clienti`, `giorni_cliente_nuovo`, variabilità di frequenze/ticket, probabilità della categoria preferita.
- `[ZONE]`: città e pesi illustrativi di estrazione; la startup offre assistenza da remoto su tutto il territorio nazionale.
- `[CATEGORIA:...]`: categorie, pesi e fattori di prezzo. Sono disponibili installazione/configurazione software, configurazione posta/account, scansione anomalie/vulnerabilità e backup.
- `[PROFILO:...]`: profili latenti di frequenza/continuità con `peso`, `richieste_mese`, `ticket_medio_euro` e, per gli inattivi, intervallo di sospensione.
- `[VALORE_ECONOMICO]`: quota e moltiplicatore dei ticket premium, indipendenti dalla frequenza.
- `[ANALISI_COMPORTAMENTI]`: soglie descrittive per il JSON, indipendenti dalla simulazione.
- `[OUTPUT]`: nomi dei file, esportazione facoltativa delle richieste, anagrafica e separatore CSV.
- `[LOG]`: livello ed esportazione facoltativa dei log; `[GEOGRAFIA]` riporta la nazione di riferimento.

### Suggerimenti per il clustering

Usare come punto di partenza **`numero_richieste`, `importo_medio_speso`, `giorni_ultima_richiesta`** (analisi RFM semplificata), applicando una scalatura appropriata. Valutare anche `importo_totale_speso`, considerando che è quasi determinato dal numero di richieste e dal ticket medio. La data di iscrizione permette di distinguere un cliente con poca storia da un inattivo anziano; eventualmente usarla per derivare l'anzianità o una frequenza normalizzata per i giorni osservabili. Tenere `id_cliente`, nome, cognome, email e date grezze fuori dal modello. Le categorie e la zona possono essere usate per interpretare i cluster oppure codificate con cura in successivi esperimenti.

**I profili latenti non sono esportati come label nel CSV.** Le fasce riassuntive del JSON non sono classi vere da recuperare né definiscono il numero di cluster: confrontare più valori di `K` con metriche e interpretazione dei risultati. Le statistiche provengono da dati inventati e non dimostrano automaticamente che le distribuzioni rappresentino un mercato reale.

## Riproducibilità e verifica

Con stesso codice, INI, seed e ambiente Python, si riproducono i medesimi record; cambiano soltanto timestamp UTC, cartella e metadato temporale. Il JSON registra configurazione, versione del generatore, data di osservazione, statistiche descrittive e hash SHA-256 del CSV.

I test verificano ID univoci, date, somme/importi, coerenza con le richieste, riproducibilità e distribuzioni osservabili. Per controllare le richieste riga per riga, impostare `esporta_richieste=yes` e confrontare il CSV facoltativo delle transazioni con quello dei clienti.

## Riepilogo JSON a fine generazione

Al termine di `python src/1_dataset_tool_generator.py`,
`store_dataset_recap()` in `libs/utils/helpers.py` salva
`customer_history_dataset.recap.json` nella stessa sottocartella UTC del CSV.
La funzione non stampa nulla a video ed è basata sui clienti effettivamente generati.

Il JSON espone **in primo piano i quattro comportamenti della traccia**
(`comportamenti_richiesti_dalla_traccia`):

- `molto_attivi`: almeno 6 richieste negli ultimi 90 giorni;
- `occasionali`: 1–2 richieste negli ultimi 90 giorni;
- `recenti`: iscrizione da meno di 30 giorni;
- `inattivi`: nessuna richiesta negli ultimi 90 giorni.

Le soglie variano in base alla configurazione INI. La sezione
`analisi_aggiuntive` include `attivi_intermedi` (3–5 richieste) e
`ticket_medio_elevato`. **Recenti è una caratteristica trasversale**:
le quattro categorie elencate nella traccia non sono una partizione,
perciò le rispettive percentuali non si sommano. Le quattro fasce
*di frequenza*, invece, rimangono mutuamente esclusive.

I metadati JSON mantengono anche `distribuzione_comportamenti_osservati`
nel formato analitico originale, per compatibilità con gli script che lo leggono.
Il CSV clienti e il processo di generazione non sono stati modificati.

---

---

## Punto 2 — Machine Learning non supervisionato: segmentazione K-Means

Il training e la validazione sono due processi indipendenti, con logica riutilizzabile in `libs/clustering/`. Si avviano seguendo la sezione **Esecuzione** (passi 2 e 3): gli script non richiedono argomenti CLI.

### Preparazione delle feature e preprocessing

Le quattro variabili di base, scelte per lo scenario di assistenza informatica, sono:

| Feature | Significato | Preprocessing |
| --- | --- | --- |
| `numero_richieste` | frequenza interventi nel periodo | mediana + z-score |
| `importo_totale_speso` | spesa nel periodo | log(1+x), mediana + z-score |
| `giorni_ultima_richiesta` | tempo trascorso dall'ultimo intervento | mediana + z-score |
| `giorni_da_iscrizione` | anzianità rispetto alla data di riferimento del dataset | derivata da `data_iscrizione`, mediana + z-score |

- Lettura del CSV con separatore acquisito dai metadati, ID obbligatori e univoci.
- Le date sono interpretate in formato **GG/MM/AAAA**, senza dipendere dal calendario di sistema.
- I valori non interpretabili, negativi o infiniti diventano mancanti. La mediana è calcolata sul dataset utilizzato per il training.
- Se una feature contiene **oltre il 25% di valori mancanti** o è tutta mancante, la procedura si interrompe con un errore.
- `SimpleImputer(strategy="median")` e `StandardScaler` sono addestrati e salvati insieme a K-Means.
- Il passaggio logaritmico sulla spesa è facoltativo nell'INI: riduce la dominanza di pochi interventi costosi.
- `nome`, `cognome`, `email`, `zona`, `categoria_preferita` e `id_cliente` **non entrano nel modello**. La categoria preferita e la città possono però aiutare a interpretare i cluster dopo il training.
- `importo_medio_speso` non entra nel primo esperimento perché deriva da spesa totale e frequenza; potrà essere oggetto di un esperimento successivo.

### Scelta di K e confronto

La configurazione predefinita è:

```ini
[KMEANS]
dataset_csv = latest
output_dir = results/kmeans
features = numero_richieste, importo_totale_speso, giorni_ultima_richiesta, giorni_da_iscrizione
k_min = 2
k_max = 8
k_confronto = 3, 4
random_state = 42
n_init = 20
log1p_spesa = yes

[VALIDATION]
training_run = latest
```

Per ogni K da `k_min` a `k_max` vengono calcolati: **inertia**, **silhouette media**,
**Davies-Bouldin** e **Calinski-Harabasz**. Il metodo del gomito individua il K a
massima distanza dalla retta che collega i due estremi della curva di inertia
(assi normalizzati). Viene inoltre individuato il K con silhouette massima.
Se il K del gomito ha una silhouette inferiore al massimo di non più di **0,02 punti**, si
preferisce il gomito; altrimenti si preferisce la silhouette massima. La regola è
esplicita e documentata, ma rimane un'euristica, **non prova dell'esistenza di K gruppi veri**.
Il confronto fra almeno due K specificati in `k_confronto` è garantito.

Ogni run è salvato in una cartella separata:

```text
results/kmeans/<timestamp_UTC>/
├── training_report.json             # metriche, configurazione, metodo scelta di K
├── preprocessing_report.json        # pulizia, valori mancanti, mediane, feature
├── metriche_k.csv                   # confronto su tutti i K studiati
├── features_scalate.csv             # matrice z-score con id cliente (no label)
├── grafico_gomito.png
├── grafico_silhouette.png
├── heatmap_centroidi_k3.png        # centroidi standardizzati
├── scatter_pca_k3.png              # proiezione PCA 2D con centroidi
├── boxplot_feature_k3.png          # distribuzioni delle feature per cluster
├── models/
│   ├── preprocessor.joblib
│   ├── kmeans_k3.joblib
│   └── kmeans_k4.joblib
├── cluster_k3.csv                   # dati originali + cluster scoperto
├── cluster_k4.csv
├── profili_cluster_k3.json          # interpretazione a posteriori
└── profili_cluster_k4.json
```

Per ogni K salvato (quelli di confronto, il gomito e il K raccomandato) vengono esportati anche heatmap dei centroidi, scatter PCA 2D e boxplot delle feature.
Gli ID numerici dei cluster sono **arbitrari** e cambiano al variare di K o del seed:
non bisogna interpretarli come etichette prestabilite (`0=inattivi`, ecc.).
I quattro comportamenti della traccia sono categorie descrittive del dataset,
**non etichette di addestramento**; "recenti" può sovrapporsi ad altri comportamenti.

### Validazione separata dei modelli

Lo script seleziona il training completo più recente (oppure quello specificato in
`[VALIDATION] training_run`), carica lo **stesso dataset e la stessa pipeline salvata**,
verifica l'hash SHA-256 del CSV e dei metadati, ricostruisce le feature e controlla che
le assegnazioni ottenute dai modelli siano identiche a quelle del training.
**Non riaddestra né imputa nuovamente i modelli**, limitandosi a verificare i risultati.

Salva sotto `results/kmeans/<run>/validations/<timestamp_UTC>/`:

- `validation_report.json`: confronto K=3/K=4 con silhouette, Davies-Bouldin e Calinski-Harabasz, dimensioni dei cluster e verifiche di integrità.
- `confronto_k.csv`: tabella sintetica delle metriche.
- `silhouette_k3.png`, `silhouette_k4.png`: distribuzioni dei coefficienti silhouette per cluster.

**Nota sulla separazione delle responsabilità:** la validazione genera soltanto la silhouette dettagliata,
le metriche di confronto e il report di verifica. Heatmap, PCA, boxplot e profili dei cluster
sono prodotti e mantenuti **esclusivamente** nella cartella del training, senza duplicazioni.
La dashboard utilizza già questi artefatti del training.

**Limite metodologico:** trattandosi di apprendimento non supervisionato, l'indice di
silhouette è una **metrica interna** calcolata sugli stessi dati su cui K-Means è stato
addestrato. La fase di validazione controlla riproducibilità, compattezza e separazione
dei gruppi, ma non costituisce un test su etichette reali o dati indipendenti.
È necessario interpretare i cluster prima di trarre conclusioni commerciali.

### Test di verifica

I nuovi test del clustering verificano scaling, gestione dei valori mancanti,
identificativi, feature proibite, training e validazione dei modelli salvati.

### Sicurezza dei modelli

Il formato `joblib` usa la serializzazione Python (pickle): **caricare solo modelli
prodotti localmente o provenienti da fonti attendibili**. Non è sicuro caricare
un `.joblib` ricevuto da sconosciuti.

---

## Frontend Streamlit — dashboard e simulazione multi-cliente

L’applicazione è unica e multipagina (`frontend/app.py`), avviabile con il comando della sezione **Esecuzione**. Non sono richiesti server backend separati e non viene riaddestrato il modello a ogni interazione.

- **Analisi dei cluster:** scatter plot, KPI, profili medi, filtri per zona, periodo e categoria preferita; heatmap dei centroidi, PCA 2D, boxplot e silhouette dettagliata disponibili dagli artefatti di training/validation.
- **Nuovi clienti:** inserimento manuale o caricamento CSV multiplo, tabella selezionabile, modifica/eliminazione, ID stabili `NEW_CUSTOMER_001` e successivi, predizione con preprocessor e K-Means salvati, rappresentazione dei clienti con simboli evidenziati nello scatter e nella PCA, esportazione dei risultati.
- **Catalogo aggiornato:** controllo automatico delle cartelle di dati, training e validation ogni 10 secondi; in caso di nuovo training completo, il menu viene aggiornato. La semplice generazione di un CSV non avvia K-Means automaticamente.

**Separazione dei dati:** i clienti aggiunti nella GUI sono temporanei e rimangono nella sessione Streamlit, associati al training selezionato. Non modificano i 600 clienti storici, i centroidi o le metriche. Il passaggio da K=3 a K=4 riclassifica i clienti simulati, ma non esegue `fit()`.

**Validation nella GUI:** l'ultima validation disponibile per il training corrente viene letta da `results/kmeans/<run>/validations/<timestamp_UTC>/`. Il pannello mostra `silhouette_k<K>.png`, metriche di confronto e, quando disponibile, l'Adjusted Rand Index; l'assenza della validation è segnalata senza rigenerare il modello.

Per la descrizione dei campi accettati, gli esempi di CSV e la durata delle simulazioni consulta [`frontend/README.md`](frontend/README.md).


## Authors
***Domenico Caliendo***

email: caliendo.domenico96@gmail.com