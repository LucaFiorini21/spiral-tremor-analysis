# Dati

Le immagini **non sono incluse** nel repository: provengono dal dataset pubblico HandPD (Botucatu Medical School, Brasile), che va scaricato dalla fonte originale rispettandone le condizioni d'uso.

Per eseguire la pipeline, mettere le scansioni in `data/raw/` (cartella ignorata da git) con nomi del tipo:

```
V01HE01.png   # soggetto 01, sano (HE), disegno 01
V07PE02.png   # soggetto 07, paziente (PE), disegno 02
```

- `V##`: codice del soggetto
- `HE` / `PE`: sano / paziente
- numero finale: numero progressivo del disegno

Il gruppo di ogni immagine viene letto dal nome del file, quindi lo schema va rispettato. Formati accettati: `.png`, `.jpg`, `.jpeg`.
